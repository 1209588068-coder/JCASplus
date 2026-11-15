"""Synthetic module-deletion checks; no AV2 or historical attack state loaded."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from spear_current import current_ablation_training as current
from spear_current import full_ablation_training as previous
from spear_current.models import GCN, GraphTrojanNet
from spear_current.support_schema import SELECTED_COORDINATES


SEED = 20260621


def fixture(device='cpu', no_support=False):
    device = torch.device(device)
    with torch.random.fork_rng():
        torch.manual_seed(44)
        x = torch.randn(8, 784).clamp(-3, 3).to(device)
    edges = torch.tensor([[0,1,1,2,3,4,4,5,6,7],
                          [1,0,2,1,4,3,5,4,7,6]], device=device)
    y = torch.tensor([0,1,0,0,0,1,0,1], device=device)
    coordinates = torch.tensor(SELECTED_COORDINATES, device=device)
    legal = (coordinates != 357) & (coordinates != 715)
    support = legal.clone()
    if not no_support:
        support[torch.nonzero(legal).flatten()[-4:]] = False
    indices = {key:torch.tensor(value, device=device) for key,value in dict(
        idx_train=[0,1],idx_attach=[2],fixed_outer=[2,3,4]).items()}
    return dict(x=x,y=y,edge_index=edges,context=SimpleNamespace(edge_index=edges)), indices, coordinates, support


def proxies(device):
    models = [GCN(784,80,2,dropout=0.,device=torch.device(device)).to(device) for _ in range(2)]
    for model in models:model.requires_grad_(False).eval()
    return models


class PolicyTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_actual_budget_contracts_and_orthogonal_kl_switches(self):
        for variant in current.VARIANTS:
            policy = current.variant_policy(variant)
            expected = current.expected_updates(variant)
            self.assertEqual(expected,dict(generator=200 if variant=='no_residual' else 800,
                surrogate=1000,discriminator=0))
            self.assertEqual(policy['output_mode'],'identity')
            self.assertFalse(policy['discriminator'])
            self.assertEqual(policy['kl_weight'],0. if variant=='no_kl' else 1.)
            self.assertEqual(policy['kl_members'],1 if variant=='single_kl' else 2)
            self.assertEqual(policy['final_support_count'],28 if variant=='no_support28' else 24)
        with self.assertRaises(ValueError):current.variant_policy('no_distribution')
        with self.assertRaises(ValueError):current.expected_updates('full',{'base':1})

    def test_cpu_fit_never_falls_back_or_creates_evidence(self):
        train,idx,coords,support = fixture()
        with TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError,'requires CUDA'):
                current.fit_case(train,idx,[],coords,support,seed=SEED,variant='full',
                    output_dir=folder,config={'epochs':1})
            self.assertFalse(list(Path(folder).iterdir()))

    def test_deletion_has_no_branch_and_does_not_consume_public_rng(self):
        hashes,streams = [],[]
        train,_,coords,support = fixture()
        for variant in ('full','no_residual','base_continuation','self_only','no_kl'):
            with torch.random.fork_rng():
                torch.manual_seed(SEED)
                shadow=GCN(784,80,2,dropout=0.,device=torch.device('cpu'))
                base=GraphTrojanNet(torch.device('cpu'),80,30)
                state=torch.get_rng_state().clone()
                gen=current.CurrentAblationGenerator(base,shadow,coords,support,variant=variant)
                self.assertTrue(torch.equal(state,torch.get_rng_state()))
                hashes.append((previous._state_hash(gen.base_generator),previous._state_hash(gen.base_shadow)))
                streams.append(previous._tensor_hash(torch.get_rng_state()))
                if variant in ('no_residual','base_continuation'):
                    self.assertIsNone(gen.residual)
                for stage in current.STAGES:
                    gen.set_phase(stage)
                    active_base=any(p.requires_grad for p in gen.base_generator.parameters())
                    self.assertEqual(active_base,stage=='base' or
                        (variant=='base_continuation' and stage!='support_surrogate'))
                    if variant=='no_residual' and stage!='base':
                        self.assertFalse(any(p.requires_grad for p in gen.parameters()))
        self.assertEqual(len(set(hashes)),1)
        self.assertEqual(len(set(streams)),1)


class CudaTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():raise unittest.SkipTest('CUDA required; CPU fallback disabled')
        if torch.cuda.mem_get_info()[0] < 512*1024**2:
            raise unittest.SkipTest('Less than512MiB headroom for tiny synthetic CUDA checks')

    def setUp(self):torch.set_num_threads(1)

    def test_all_twelve_conditions_fit_with_real_deleted_paths_and_bound_outputs(self):
        device=torch.device('cuda:0')
        train,idx,coords,support=fixture(device)
        original_x=train['x'].clone()
        frozen=proxies(device)
        initializations=[]
        with TemporaryDirectory() as folder:
            for variant in current.VARIANTS:
                with self.subTest(variant=variant), patch.object(previous,'SeededDistributionRegularizer',
                        side_effect=AssertionError('No D belongs to the current method')):
                    selected=(coords!=357)&(coords!=715) if variant=='no_support28' else support
                    result=current.fit_case(train,idx,frozen,coords,selected,seed=SEED,
                        variant=variant,output_dir=Path(folder)/variant,config={'epochs':1})
                    initializations.append(result['initialization'])
                    expected=current.expected_updates(variant,1)
                    self.assertEqual(result['optimizer_steps'],expected)
                    self.assertEqual([r['stage'] for r in result['records']],list(current.STAGES))
                    self.assertEqual([r['support_count'] for r in result['records']],
                        [28]*5 if variant=='no_support28' else [28,28,24,24,24])
                    self.assertEqual(result['generator'].output_mode,'identity')
                    self.assertTrue(all(r['distribution_weight']==0 and r['d_steps']==0 for r in result['records']))
                    self.assertTrue(all(r['clean_kl_weight']==(0 if variant=='no_kl' else 1) for r in result['records']))
                    if variant=='no_kl':self.assertTrue(all(r['clean_kl_loss']==0 for r in result['records']))
                    if variant=='single_kl':self.assertTrue(all(r['clean_kl_members']==1 for r in result['records']))
                    if variant in ('no_residual','base_continuation'):
                        self.assertIsNone(result['generator'].residual)
                    for stage in result['stages']:
                        if stage['generator_steps']:
                            self.assertNotEqual(stage['initial_state_sha256']['active_generator']['sha256'],
                                stage['final_state_sha256']['active_generator']['sha256'])
                        else:
                            self.assertEqual(stage['initial_state_sha256']['generator'],
                                stage['final_state_sha256']['generator'])
                    injected=result['generator'].inject(train['x'],train['context'],idx['idx_attach'])
                    allowed=torch.zeros_like(train['x'],dtype=torch.bool)
                    allowed[idx['idx_attach'][:,None],coords[selected]]=True
                    self.assertTrue(torch.equal(injected.features[~allowed],train['x'][~allowed]))
                    self.assertTrue(torch.equal(injected.edge_index,train['edge_index']))
                    self.assertTrue(torch.equal(original_x,train['x']))
                    self.assertTrue(Path(result['final_checkpoint']).is_file())
            for key in ('fresh_model_sha256','initialization_rng_sha256','post_initialization_rng_sha256'):
                self.assertTrue(all(r[key]==initializations[0][key] for r in initializations),key)

    def test_full_matches_previously_tested_identity_route(self):
        device=torch.device('cuda:0');train,idx,coords,support=fixture(device);frozen=proxies(device)
        with TemporaryDirectory() as folder:
            old=previous.fit_case(train,idx,frozen,coords,support,seed=SEED,
                variant='no_distribution_no_domain',output_dir=Path(folder)/'old',config={'epochs':2})
            new=current.fit_case(train,idx,frozen,coords,support,seed=SEED,
                variant='full',output_dir=Path(folder)/'new',config={'epochs':2})
            self.assertEqual(old['initialization']['fresh_model_sha256'],new['initialization']['fresh_model_sha256'])
            for key,value in old['generator'].state_dict().items():
                other=new['generator'].state_dict()[key]
                self.assertTrue(torch.allclose(value,other,atol=2e-6,rtol=2e-6),key)
            for key,value in old['surrogate'].state_dict().items():
                self.assertTrue(torch.allclose(value,new['surrogate'].state_dict()[key],atol=2e-6,rtol=2e-6),key)

    def test_self_only_branch_ignores_neighbor_features_after_learning(self):
        device=torch.device('cuda:0');train,idx,coords,support=fixture(device);frozen=proxies(device)
        with TemporaryDirectory() as folder:
            result=current.fit_case(train,idx,frozen,coords,support,seed=SEED,
                variant='self_only',output_dir=folder,config={'epochs':2})
            residual=result['generator'].residual
            before=residual(train['x'],train['context'])[0]
            changed=train['x'].clone();changed[1]+=100
            after=residual(changed,train['context'])[0]
            self.assertTrue(torch.equal(before,after))
            self.assertIsNone(residual.relation_weight.grad)


if __name__=='__main__':unittest.main()
