"""Self-contained synthetic model fixtures migrated from historical unit tests."""
from copy import deepcopy
import torch
from torch import nn
from spear_current.role_context import build_role_context
from spear_current.role_generator import PairRoleResidualGenerator
from spear_current.attention_context import attention_context
from spear_current.attention_generator import ContentAttentionGenerator
from spear_current.support_generator import fixed_support
from spear_current.support_schema import SELECTED_COORDINATES


class ToyShadow(nn.Module):
    """A deterministic eval-time one-hop shadow, including a mode-sensitive layer."""

    def __init__(self, input_dim=8, hidden_dim=4):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(.75)
        self.register_buffer('scale', torch.tensor(.3))

    def get_h(self, features, edge_index):
        source, target = edge_index
        aggregate = torch.zeros_like(features).index_add(0, target, features[source])
        degree = torch.bincount(target, minlength=len(features)).to(features.dtype).clamp_min(1)
        return self.dropout(self.linear(features + self.scale * aggregate / degree[:, None]))


class ToyGenerator(nn.Module):
    def __init__(self, hidden_dim=4, output_dim=3):
        super().__init__()
        self.dropout = nn.Dropout(.75)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.register_buffer('offset', torch.tensor(.2))

    def forward(self, features, threshold):
        return self.linear(self.dropout(features)) + threshold + self.offset


def fixture(input_dim=8):
    pairs = torch.tensor([[0, 1], [0, 2], [1, 2], [1, 3], [0, 3], [2, 3]])
    edges = torch.tensor([(i, j) for i in range(len(pairs)) for j in range(len(pairs))
                          if i != j and len(set(pairs[i].tolist()) & set(pairs[j].tolist())) == 1]).T
    context = build_role_context(pairs, edges)
    # Unequal source vectors are essential: role shuffling must change messages.
    features = torch.arange(1, len(pairs) * input_dim + 1, dtype=torch.float32).reshape(-1, input_dim) / 41
    return pairs, context, features


def make_model(input_dim=8, hidden_dim=4, selected=None, mode='typed', threshold=.37):
    selected = torch.tensor([6, 1, 3]) if selected is None else selected
    return PairRoleResidualGenerator(
        ToyGenerator(hidden_dim, len(selected)), ToyShadow(input_dim, hidden_dim), selected,
        input_dim=input_dim, hidden_dim=hidden_dim, mode=mode, threshold=threshold,
    )


def activate_branch(model):
    """Give each role a different nonzero map and avoid accidental dead ReLUs."""
    with torch.no_grad():
        for role in range(4):
            model.residual.relation_weight[role].fill_(.01 * (role + 1))
        model.residual.self_projection.weight.fill_(.02)
        model.residual.self_projection.bias.fill_(.3)
        model.residual.head.weight.fill_(.1)
        model.residual.head.bias.fill_(.05)


def snapshot(module):
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


class ToyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 2)
        self.register_buffer('neighbor_scale', torch.tensor(.1))
        self.calls = []

    def forward(self, features, edge_index, edge_weight):
        source, target = edge_index
        aggregate = torch.zeros_like(features).index_add(0, target, features[source] * edge_weight[:, None])
        degree = torch.bincount(target, minlength=len(features)).clamp_min(1)
        values = features + self.neighbor_scale * aggregate / degree[:, None]
        output = self.linear(values).log_softmax(1)
        self.calls.append(dict(features=features.detach().clone(), edge_index=edge_index.detach().clone(),
            output=output.detach().clone(), input_requires_grad=features.requires_grad,
            grad_enabled=torch.is_grad_enabled(), training=self.training,
            parameters_require_grad=all(parameter.requires_grad for parameter in self.parameters())))
        return output


def setup():
    with torch.random.fork_rng():
        torch.manual_seed(20260621)
        _, roles, x = fixture(784)
        x = x / 200  # Synthetic inputs remain strictly inside the clean domain.
        context = attention_context(roles, ['synthetic-scene'], seed=17)
        anchor = make_model(input_dim=784, selected=torch.tensor(SELECTED_COORDINATES), mode='untyped')
        activate_branch(anchor)
        anchor = ContentAttentionGenerator(anchor, attention_mode='uniform', score_dim=4)
        anchor.eval().requires_grad_(False)
    mask = fixed_support('random_semantic', 24, seed=20260621)
    return anchor, mask, x, context, torch.tensor([4, 1, 5])


def classifier():
    model = ToyClassifier()
    model.linear = torch.nn.Linear(784, 2)
    return model


def assert_output_scope(clean, changed, targets, generator):
    support = generator.selected_coordinates[generator.hard_support]
    allowed = torch.zeros_like(clean, dtype=torch.bool)
    allowed[targets[:, None], support] = True
    if (changed.shape != clean.shape or not bool(torch.isfinite(changed).all())
            or not torch.equal(changed[~allowed], clean[~allowed])):
        raise ValueError('Output changed a nontarget row or an unselected coordinate')
    if generator.output_mode == 'domain_projection' and bool((changed[allowed].abs() > 10).any()):
        raise ValueError('Projected output left the registered standardized domain')


