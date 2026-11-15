"""Real tiny CPU fits for the separate offline AV2 ablation implementation."""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from spear_current import compact_ablation_training as registered
from spear_current import cpu_compact_ablation_training as cpu
from spear_current.dual_clean_proxy import FrozenMeanCleanResponsePenalty
from spear_current.models import GCN, GraphTrojanNet
from spear_current.support_schema import SELECTED_COORDINATES

SEED = 20260621


def fixture():
    # Independent CPU generator avoids global seeding of CUDA in test setup.
    x = torch.randn(8, 784, generator=torch.Generator().manual_seed(44)).clamp(-3, 3)
    edge = torch.tensor([[0,1,1,2,3,4,4,5,6,7], [1,0,2,1,4,3,5,4,7,6]])
    labels = torch.tensor([0,1,0,0,0,1,0,1])
    coordinates = torch.tensor(SELECTED_COORDINATES)
    legal = (coordinates != 357) & (coordinates != 715)
    support = legal.clone()
    support[torch.nonzero(legal).flatten()[-4:]] = False
    indices = {key:torch.tensor(value) for key,value in dict(
        idx_train=[0,1], idx_attach=[2], fixed_outer=[2,3,4]).items()}
    train = dict(x=x, y=labels, edge_index=edge, context=SimpleNamespace(edge_index=edge))
    proxies = [GCN(784,80,2,dropout=0.,device=torch.device('cpu')).eval().requires_grad_(False)
               for _ in range(2)]
    return train, indices, proxies, coordinates, support


@contextmanager
def forbid_cuda_execution():
    with ExitStack() as stack:
        for name in ('_lazy_init', 'init', 'current_device', 'device_count',
                     'manual_seed', 'manual_seed_all', 'get_rng_state', 'get_rng_state_all',
                     'set_rng_state', 'set_rng_state_all'):
            stack.enter_context(patch.object(torch.cuda, name,
                side_effect=AssertionError('CPU fit called CUDA: '+name)))
        # The public all-device seed API must not be used either.
        stack.enter_context(patch.object(torch, 'manual_seed',
            side_effect=AssertionError('CPU fit used all-device manual_seed')))
        yield


def manual_full(train, indices, proxies, coordinates, support, epochs, inner_steps):
    """Uninstrumented five-stage reference using the original generator and losses."""
    x, labels, context = train['x'], train['y'], train['context']
    edge = context.edge_index
    weights = x.new_ones(edge.shape[1])
    clean, attach, outer = (indices[k] for k in ('idx_train', 'idx_attach', 'fixed_outer'))
    inner_ids, outer_ids = torch.cat((clean, attach)), torch.cat((clean, outer))
    inner_y, outer_y = labels.clone(), labels.clone()
    inner_y[attach], outer_y[outer] = 1, 1
    legal = (coordinates != 357) & (coordinates != 715)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(SEED)
        shadow = GCN(784,80,2,dropout=0.,device=torch.device('cpu'))
        base = GraphTrojanNet(torch.device('cpu'),80,30)
        generator = registered.CompactAblationGenerator(base,shadow,coordinates,legal,variant='full')
        generator.set_phase('base')
        surrogate = generator.base_shadow
        penalty = FrozenMeanCleanResponsePenalty(proxies,x,edge,weights,1.)
        poison_x = x.clone().detach()
        losses = []
        for stage in registered.STAGES:
            if stage == 'residual':surrogate = deepcopy(generator.base_shadow)
            generator.set_phase(stage)
            generator.set_support(legal if stage in ('base','residual') else support)
            surrogate.requires_grad_(True)
            sopt = torch.optim.Adam(surrogate.parameters(),lr=.0002,weight_decay=.0005)
            surrogate_ids = {id(p) for p in surrogate.parameters()}
            active = [p for p in generator.parameters() if p.requires_grad and id(p) not in surrogate_ids]
            gopt = torch.optim.Adam(active,lr=.0002,weight_decay=.0005) if active else None
            for _ in range(epochs):
                generator.train();generator.zero_grad(set_to_none=True)
                surrogate.requires_grad_(True).train()
                for _ in range(inner_steps):
                    sopt.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        if stage == 'base':
                            embedding = surrogate.get_h(poison_x,edge)
                            values = generator.generate_with_embedding(x,context,attach,embedding)
                            poison_x = x.clone().detach()
                            poison_x[attach[:,None],coordinates] = values.detach()
                            inner_x = poison_x
                        else:inner_x = generator.inject(x,context,attach,weights).features
                    F.nll_loss(surrogate(inner_x,edge,weights)[inner_ids],inner_y[inner_ids]).backward()
                    sopt.step()
                sopt.zero_grad(set_to_none=True)
                surrogate.requires_grad_(False).eval();generator.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(gopt is not None):
                    if stage == 'base':
                        with torch.no_grad():embedding = surrogate.get_h(poison_x,edge)
                        values = generator.generate_with_embedding(x,context,outer,embedding)
                        changed = x.clone();changed[outer[:,None],coordinates] = values
                    else:changed = generator.inject(x,context,outer,weights,detach=False).features
                    nll = F.nll_loss(surrogate(changed,edge,weights)[outer_ids],outer_y[outer_ids])
                    similarity = 1.-F.cosine_similarity(changed,x,dim=1).mean()
                    loss = nll + .1*similarity + penalty(changed,edge,weights,outer) + changed.sum()*0.
                    losses.append(float(loss.detach()))
                    if gopt is not None:
                        gopt.zero_grad(set_to_none=True);loss.backward();gopt.step()
                surrogate.requires_grad_(True)
            generator.zero_grad(set_to_none=True);surrogate.zero_grad(set_to_none=True)
    return generator, surrogate, losses


class CpuTrainingTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_all_seven_actual_cpu_fits_preserve_pairing_budgets_and_boundaries(self):
        train,indices,proxies,coordinates,support = fixture()
        original = train['x'].clone()
        rng_before = torch.get_rng_state().clone()
        initializations = []
        self.assertIs(cpu.CompactAblationGenerator, registered.CompactAblationGenerator)
        with TemporaryDirectory() as folder, forbid_cuda_execution():
            for variant in cpu.VARIANTS:
                with self.subTest(variant=variant):
                    result = cpu.fit_case(train,indices,proxies,coordinates,support,seed=SEED,
                        variant=variant,output_dir=Path(folder)/variant,config={'epochs':1,'inner_steps':2})
                    initializations.append(result['initialization'])
                    self.assertEqual(result['optimizer_steps'],dict(
                        generator=1 if variant=='no_residual' else 4,surrogate=10,discriminator=0))
                    self.assertEqual([r['support_count'] for r in result['records']],[28,28,24,24,24])
                    self.assertEqual([r['stage'] for r in result['records']],list(cpu.STAGES))
                    self.assertTrue(all(r['device']=='cpu' and r['output_mode']=='identity'
                        and r['distribution_weight']==0 and r['d_steps']==0
                        and r['clean_kl_members']==2 for r in result['records']))
                    if variant=='no_kl':
                        self.assertTrue(all(r['clean_kl_loss']==0 for r in result['records']))
                    if variant=='no_residual':
                        self.assertIsNone(result['generator'].residual)
                        self.assertTrue(all(s['generator_frozen'] for s in result['stages'][1:]))
                    if variant=='no_generator_neighborhood':
                        self.assertIsInstance(result['generator'].base_shadow,cpu.SelfOnlyConditioningGCN)
                        self.assertIsInstance(result['surrogate'],cpu.SelfOnlyConditioningGCN)
                        self.assertEqual(result['generator'].residual.mode,'self_only')
                        other = train['x'].clone();other[1:]+=15
                        with torch.no_grad():
                            a = result['generator'](train['x'],train['context'],torch.tensor([0]))
                            b = result['generator'](other,train['context'],torch.tensor([0]))
                        self.assertTrue(torch.equal(a,b))
                    injected = result['generator'].inject(train['x'],train['context'],indices['idx_attach'])
                    allowed = torch.zeros_like(train['x'],dtype=torch.bool)
                    allowed[indices['idx_attach'][:,None],coordinates[support]] = True
                    self.assertTrue(torch.equal(injected.features[~allowed],train['x'][~allowed]))
                    self.assertTrue(torch.equal(injected.edge_index,train['edge_index']))
                    checkpoint = torch.load(result['final_checkpoint'],map_location='cpu',weights_only=True)
                    metadata = checkpoint['initialization']
                    self.assertEqual(metadata['device'],'cpu')
                    self.assertEqual(metadata['dtype'],'torch.float32')
                    self.assertFalse(metadata['cuda_used'])
                    for key in ('initialization_rng_sha256','post_initialization_rng_sha256'):
                        self.assertIsNone(metadata[key]['cuda'])
                        self.assertEqual(len(metadata[key]['cpu']),64)
                    self.assertTrue(all(v.device.type=='cpu' for v in checkpoint['generator'].values()))
                    self.assertEqual(checkpoint['architecture']['input_dim'],784)
                    self.assertEqual(checkpoint['architecture']['hidden_dim'],80)
                    self.assertEqual(checkpoint['architecture']['output_dim'],30)
                    persisted = json.loads((Path(folder)/variant/'completion.json').read_text())
                    self.assertEqual(persisted['status'],'CPU_COMPACT_ABLATION_TRAINING_COMPLETE')
                    self.assertTrue(torch.equal(train['x'],original))
                    self.assertTrue(torch.equal(torch.get_rng_state(),rng_before))
        for key in ('fresh_model_sha256','initialization_rng_sha256','post_initialization_rng_sha256'):
            self.assertTrue(all(i[key]==initializations[0][key] for i in initializations),key)
        self.assertTrue(all(p.grad is None and not p.requires_grad
            for model in proxies for p in model.parameters()))

    def test_full_exactly_matches_independent_manual_cpu_loop(self):
        train,indices,proxies,coordinates,support = fixture()
        with TemporaryDirectory() as folder, forbid_cuda_execution():
            result = cpu.fit_case(train,indices,proxies,coordinates,support,seed=SEED,
                variant='full',output_dir=folder,config={'epochs':2,'inner_steps':2})
            generator,surrogate,losses = manual_full(train,indices,proxies,coordinates,support,2,2)
        self.assertEqual([r['outer_loss'] for r in result['records']],losses)
        for name,reference in (('generator',generator),('surrogate',surrogate)):
            for key,value in reference.state_dict().items():
                torch.testing.assert_close(value,result[name].state_dict()[key],rtol=0.,atol=0.,
                    msg=lambda message:f'{name}.{key}: {message}')

    def test_accelerator_input_refused_before_evidence(self):
        train,indices,proxies,coordinates,support = fixture()
        train['x'] = torch.empty((8,784),device='meta')
        with TemporaryDirectory() as folder, forbid_cuda_execution():
            with self.assertRaisesRegex(ValueError,'requires CPU'):
                cpu.fit_case(train,indices,proxies,coordinates,support,seed=SEED,
                    variant='full',output_dir=folder,config={'epochs':1})
            self.assertFalse(list(Path(folder).iterdir()))

    def test_registered_budget_and_policy_are_unchanged(self):
        self.assertEqual(cpu.VARIANTS,registered.VARIANTS)
        self.assertEqual(cpu.STAGES,registered.STAGES)
        for variant in cpu.VARIANTS:
            self.assertEqual(cpu.variant_policy(variant),registered.variant_policy(variant))
            self.assertEqual(cpu.expected_updates(variant),registered.expected_updates(variant))


if __name__=='__main__':unittest.main()
