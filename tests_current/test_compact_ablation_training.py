"""Six primary ablation contracts on synthetic inputs; no AV2/checkpoint IO."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch

from spear_current import compact_ablation_training as compact
from spear_current import current_ablation_training as prior
from spear_current.models import GCN, GraphTrojanNet
from tests_current.test_current_ablation_training import fixture, proxies, SEED


def generator(variant, device='cpu'):
    device = torch.device(device)
    _, _, coordinates, support = fixture(device)
    ids = [device.index if device.index is not None else 0] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=ids):
        torch.manual_seed(SEED)
        if device.type == 'cuda':torch.cuda.manual_seed_all(SEED)
        cls = compact.SelfOnlyConditioningGCN if variant == 'no_generator_neighborhood' else GCN
        shadow = cls(784,80,2,dropout=0.,device=device).to(device)
        base = GraphTrojanNet(device,80,30).to(device)
        return compact.CompactAblationGenerator(base,shadow,coordinates,support,variant=variant)


class ModuleContracts(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_exact_six_ablations_plus_full_budgets(self):
        self.assertEqual(compact.VARIANTS,('full','matched_random_pairs','random_features',
            'no_generator_neighborhood','no_residual','self_only','no_kl'))
        for variant in compact.VARIANTS:
            p=compact.variant_policy(variant)
            self.assertEqual(compact.expected_updates(variant),dict(
                generator=200 if variant=='no_residual' else 800,surrogate=1000,discriminator=0))
            self.assertEqual(p['final_support_count'],24)
            self.assertEqual(p['kl_members'],2)
            self.assertEqual(p['kl_weight'],0. if variant=='no_kl' else 1.)
            self.assertEqual(p['output_mode'],'identity')
            self.assertFalse(p['discriminator'])
        with self.assertRaises(ValueError):compact.variant_policy('single_kl')

    def test_cpu_fit_refuses_training_before_writing_any_evidence(self):
        train,indices,coordinates,support=fixture()
        with TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError,'requires CUDA'):
                compact.fit_case(train,indices,[],coordinates,support,seed=SEED,
                    variant='no_generator_neighborhood',output_dir=folder,config={'epochs':1})
            self.assertFalse(list(Path(folder).iterdir()))

    def test_ne_requires_self_only_conditioning_not_just_self_only_residual(self):
        _,_,coordinates,support=fixture()
        shadow=GCN(784,80,2,dropout=0.,device=torch.device('cpu'))
        base=GraphTrojanNet(torch.device('cpu'),80,30)
        with self.assertRaisesRegex(ValueError,'MLP conditioning'):
            compact.CompactAblationGenerator(base,shadow,coordinates,support,
                variant='no_generator_neighborhood')

    def test_ne_has_same_base_parameters_and_initial_weights_as_full(self):
        full,ne=generator('full'),generator('no_generator_neighborhood')
        self.assertEqual(list(full.base_shadow.state_dict()),list(ne.base_shadow.state_dict()))
        for key,value in full.base_shadow.state_dict().items():
            self.assertTrue(torch.equal(value,ne.base_shadow.state_dict()[key]),key)
        for key,value in full.base_generator.state_dict().items():
            self.assertTrue(torch.equal(value,ne.base_generator.state_dict()[key]),key)
        self.assertEqual(ne.residual.mode,'self_only')

    def test_self_mlp_matches_self_loop_gcn_without_graph_input(self):
        train,_,_,_=fixture();x=train['x'];edges=train['edge_index']
        full,ne=generator('full'),generator('no_generator_neighborhood')
        loops=torch.arange(len(x)).repeat(2,1)
        torch.testing.assert_close(ne.base_shadow.get_h(x,edges),full.base_shadow.get_h(x,loops),
            atol=1e-6,rtol=1e-6)
        changed=x.clone();changed[1]+=20
        self.assertTrue(torch.equal(ne.base_shadow.get_h(x,edges)[0],
                                    ne.base_shadow.get_h(changed,torch.empty((2,0),dtype=torch.long))[0]))

    def test_ne_generator_ignores_neighbor_features_and_topology_on_both_paths(self):
        train,_,_,_=fixture();g=generator('no_generator_neighborhood').eval()
        # Exercise a nonzero self residual, not only the zero output-head start.
        with torch.no_grad():g.residual.head.weight.fill_(.01);g.residual.head.bias.fill_(.02)
        x=train['x'];changed=x.clone();changed[1:]+=10
        contexts=(train['context'],SimpleNamespace(edge_index=torch.empty((2,0),dtype=torch.long)))
        targets=torch.tensor([0])
        for phase in ('base','residual','continuation_b'):
            g.set_phase(phase);g.eval()
            with torch.no_grad():
                before=g(x,contexts[0],targets)
                after=g(changed,contexts[1],targets)
            self.assertTrue(torch.equal(before,after),phase)

    def test_ne_shadow_classifier_still_uses_the_original_graph(self):
        train,_,_,_=fixture();g=generator('no_generator_neighborhood').eval()
        x=train['x'];changed=x.clone();changed[1]+=10
        with torch.no_grad():
            a=g.base_shadow(x,train['edge_index'])[0]
            b=g.base_shadow(changed,train['edge_index'])[0]
        self.assertFalse(torch.equal(a,b))


class CudaTrainingContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            ready=torch.cuda.is_available()
            free=torch.cuda.mem_get_info()[0] if ready else 0
        except RuntimeError as error:
            raise unittest.SkipTest('CUDA unavailable during shared-resource probe: '+str(error))
        if not ready or free<512*1024**2:
            raise unittest.SkipTest('Tiny synthetic CUDA check lacks512MiB; no CPU fallback')
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False

    def setUp(self):torch.set_num_threads(1)

    def test_all_seven_fit_with_matched_initialization_and_real_ne_paths(self):
        device=torch.device('cuda:0');train,indices,coordinates,support=fixture(device)
        frozen=proxies(device);initial=[];original=train['x'].clone()
        with TemporaryDirectory() as folder:
            for variant in compact.VARIANTS:
                result=compact.fit_case(train,indices,frozen,coordinates,support,seed=SEED,
                    variant=variant,output_dir=Path(folder)/variant,config={'epochs':1})
                initial.append(result['initialization'])
                self.assertEqual(result['optimizer_steps'],compact.expected_updates(variant,1))
                self.assertEqual([r['support_count'] for r in result['records']],[28,28,24,24,24])
                self.assertTrue(all(r['output_mode']=='identity' and r['distribution_weight']==0
                    and r['d_steps']==0 and r['clean_kl_members']==2 for r in result['records']))
                if variant=='no_kl':self.assertTrue(all(r['clean_kl_loss']==0 for r in result['records']))
                if variant=='no_residual':self.assertIsNone(result['generator'].residual)
                if variant=='no_generator_neighborhood':
                    self.assertIsInstance(result['generator'].base_shadow,compact.SelfOnlyConditioningGCN)
                    self.assertIsInstance(result['surrogate'],compact.SelfOnlyConditioningGCN)
                    self.assertEqual(result['generator'].residual.mode,'self_only')
                    other=train['x'].clone();other[1:]+=15
                    with torch.no_grad():
                        a=result['generator'](train['x'],train['context'],torch.tensor([0],device=device))
                        b=result['generator'](other,train['context'],torch.tensor([0],device=device))
                    self.assertTrue(torch.equal(a,b))
                self.assertTrue(torch.equal(train['x'],original))
                self.assertTrue(Path(result['final_checkpoint']).exists())
            for key in ('fresh_model_sha256','initialization_rng_sha256','post_initialization_rng_sha256'):
                self.assertTrue(all(i[key]==initial[0][key] for i in initial),key)

    def test_full_preserves_prior_current_route_numerics(self):
        train,indices,coordinates,support=fixture('cuda:0');frozen=proxies('cuda:0')
        with TemporaryDirectory() as folder:
            old=prior.fit_case(train,indices,frozen,coordinates,support,seed=SEED,
                variant='full',output_dir=Path(folder)/'prior',config={'epochs':2})
            new=compact.fit_case(train,indices,frozen,coordinates,support,seed=SEED,
                variant='full',output_dir=Path(folder)/'compact',config={'epochs':2})
            self.assertEqual(old['initialization']['fresh_model_sha256'],new['initialization']['fresh_model_sha256'])
            for name in ('generator','surrogate'):
                for key,value in old[name].state_dict().items():
                    torch.testing.assert_close(value,new[name].state_dict()[key],atol=2e-6,rtol=2e-6,
                        msg=lambda message:f'{name}.{key}: {message}')
            for a,b in zip(old['records'],new['records']):
                self.assertEqual(a['stage'],b['stage'])
                self.assertAlmostEqual(a['outer_loss'],b['outer_loss'],places=5)


if __name__=='__main__':unittest.main()
