import unittest

import numpy as np
import pandas as pd

from jcas.workflows import relational_matched_input_audit as audit


class MatchedInputTests(unittest.TestCase):
    def test_energy_matched_and_bounded_templates(self):
        bank,rms=audit.templates()
        self.assertEqual(len(bank),4)
        for template in bank.values():
            wave=np.asarray(template['relative_wave_m'])
            self.assertAlmostEqual(float(np.sqrt(np.mean(np.sum((wave/2)**2,axis=1)))),rms,places=12)
            self.assertTrue(audit.pilot.trigger.budget_ok(-wave/2,wave/2,audit.pilot.trigger.reference_budget()))

    def fixtures(self):
        row=dict(scenario_id='t',split_group_id='A',fold=0,city='M',motion_regime='slow_lt_0p5',
                 min_speed=.1,max_speed=.2,distance=10.,closing_speed=0.,log_nodes=2.,log_degree=1.)
        targets=pd.DataFrame([{**row,'target_priority':'a'}])
        donors=pd.DataFrame([{**row,'pair_key':'same_recording'},
            {**row,'split_group_id':'B','pair_key':'wrong_fold','fold':1},
            {**row,'split_group_id':'C','pair_key':'too_far','distance':30.},
            {**row,'split_group_id':'D','pair_key':'good','distance':11.}])
        return targets,donors

    def test_matching_exclusions_and_no_reuse(self):
        targets,donors=self.fixtures()
        targets=pd.concat([targets,targets.assign(scenario_id='t2',target_priority='b')],ignore_index=True)
        result=audit.match_pairs(targets,donors)
        self.assertTrue(result.iloc[0].matched)
        self.assertEqual(result.iloc[0].matched_pair_key,'good')
        self.assertFalse(result.iloc[1].matched)
        pd.testing.assert_frame_equal(result,audit.match_pairs(targets,donors))

    def test_matching_does_not_consume_model_scores(self):
        targets,donors=self.fixtures()
        a=audit.match_pairs(targets,donors)
        b=audit.match_pairs(targets.assign(victim_score=1.),donors.assign(victim_score=np.arange(len(donors))))
        pd.testing.assert_frame_equal(a,b)

    def test_recording_folds_stable(self):
        self.assertEqual(audit.group_fold('M:record1'),audit.group_fold('M:record1'))
        self.assertTrue(all(0<=audit.group_fold(str(i))<5 for i in range(100)))

    def test_probe_uses_fixed_folds_and_missing_data_reported(self):
        rng=np.random.default_rng(21)
        a=rng.normal(size=(30,4))+2;b=rng.normal(size=(30,4));folds=np.arange(30)%5
        result,scores=audit.probe(a,b,folds)
        self.assertEqual(result['status'],'COMPLETE')
        self.assertEqual(len(result['folds']),5)
        self.assertTrue(np.isfinite(scores).all())
        empty,_=audit.probe(a[:0],b[:0],folds[:0])
        self.assertEqual(empty['status'],'INSUFFICIENT_FOLDS')

    def test_radial_closing_covariate(self):
        p=np.zeros((2,11,2));p[1,:,0]=20.-np.arange(11)*.2
        v=np.zeros_like(p);v[1,:,0]=-2.
        graph={'observed_positions_filled':p,'observed_velocities_filled':v,'edge_index':np.array([[0,1],[1,0]])}
        cov=audit.covariates(graph,0,1)
        self.assertAlmostEqual(cov['closing_speed'],2.)
        self.assertAlmostEqual(cov['distance'],18.)
        self.assertEqual(cov['min_speed'],0.)


if __name__=='__main__':unittest.main()
