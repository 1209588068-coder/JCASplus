import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from jcas.core.target_rotation import (
    FIXED_FIELDS, HOLDOUT_GATES, MAX_TRAIN_TARGETS, MOVING, SLOW, VERSION,
    SELECTION_SEED, pair_key, partition_candidates, validate_roles, epoch_rows,
    schedule_digest, load_plan,
)
from jcas.core.poison import sha256_file
from jcas.workflows.target_rotation_data import EpochTargetDataset, load_rotating_dataset
from jcas.workflows.explicit_counterfactual_trainer import (
    compose_counterfactual_training_items, collate_counterfactual_items,
    train_counterfactual_epoch, parse_args,
)
from jcas.workflows.rotating_counterfactual_pilot import (
    combined_gate, training_command, validate_rotation_arguments, validate_candidate,
)
from jcas.workflows.projected_counterfactual_pilot import check_candidate
from jcas.core.models import build_model
from jcas.core.counterfactual_gradients import CleanProjectionConfig
from tests.test_projected_counterfactual import BALANCED, gate_fixture


def target(src, dst, scene="moving", regime=MOVING):
    return {
        **FIXED_FIELDS, "scenario_id":scene,"src":src,"dst":dst,
        "src_track_id":str(src),"dst_track_id":str(dst),"motion_regime":regime,
        "min_endpoint_speed_mps":1. if regime == MOVING else .1,
        "graph_sha256":"a"*64,
    }


def role_fixture():
    a=target(0,1)
    return pd.DataFrame([
        {**a,"role":"train","slot":0},
        {**target(0,2),"role":"train","slot":1},
        {**target(1,2),"role":"holdout","slot":-1},
        {**target(0,1,"slow",SLOW),"role":"train","slot":0},
    ])


def graph(scene):
    return Data(
        scenario_id=scene, x=torch.ones(3,1),
        edge_index=torch.tensor([[0,1,0,2,1,2],[1,0,2,0,2,1]],dtype=torch.long),
        edge_attr=torch.ones(6,1)*-.5, edge_label=torch.zeros(6),
        supervision_edge_mask=torch.ones(6,dtype=torch.bool),
        message_passing_edge_mask=torch.ones(6,dtype=torch.bool),
        poisoned_edge_mask=torch.zeros(6,dtype=torch.bool),
    )


def triggered(clean, src=0,dst=1):
    data=clean.clone()
    ei=data.edge_index
    mask=((ei[0]==src)&(ei[1]==dst))|((ei[0]==dst)&(ei[1]==src))
    data.poisoned_edge_mask=mask
    data.edge_label[mask]=1.
    data.edge_attr[mask]=.5
    return data


def dataset_fixture():
    clean=[graph("ordinary"),graph("moving"),graph("slow")]
    items,_=compose_counterfactual_training_items(clean,[triggered(clean[1]),triggered(clean[2])])
    alternatives,_=compose_counterfactual_training_items([clean[1]],[triggered(clean[1],0,2)])
    mapping={str(item.graph.scenario_id):{0:item} for item in items if item.has_counterfactual_pair}
    mapping["moving"][1]=alternatives[-1]
    return EpochTargetDataset(items,mapping,role_fixture()),items


class RolePartitionTests(unittest.TestCase):
    def test_anchor_holdout_and_train_disjoint_under_reversed_input_order(self):
        a=target(0,1); candidates=[a]+[target(0,i) for i in range(2,9)]
        train,held=partition_candidates(a,candidates)
        t2,h2=partition_candidates(a,list(reversed(candidates)))
        self.assertEqual(train,t2); self.assertEqual(held,h2)
        self.assertEqual(len(train),MAX_TRAIN_TARGETS)
        self.assertEqual(train[0],a)
        self.assertNotIn(pair_key(held),{pair_key(r) for r in train})

    def test_excluded_diagnostic_pair_cannot_be_holdout(self):
        a=target(0,1); candidates=[a,target(0,2),target(1,2),target(0,3)]
        _,first=partition_candidates(a,candidates)
        _,held=partition_candidates(a,candidates,{pair_key(first)})
        self.assertNotEqual(pair_key(held),pair_key(first))

    def test_only_two_candidates_do_not_rotate_or_create_holdout(self):
        a=target(0,1)
        self.assertEqual(partition_candidates(a,[a,target(0,2)]),([a],None))

    def test_all_alternatives_excluded_and_slow_fallback_remain_fixed(self):
        a=target(0,1); alternatives=[target(0,2),target(1,2)]
        self.assertEqual(partition_candidates(a,[a,*alternatives],{pair_key(r) for r in alternatives}),([a],None))
        slow=target(0,1,"slow",SLOW)
        self.assertEqual(partition_candidates(slow,[slow,target(0,2,"slow",SLOW)]),([slow],None))

    def test_reversed_duplicate_and_cross_scene_candidates_fail(self):
        a=target(0,1)
        for candidates in ([a,target(1,0)],[a,target(0,2,"other")]):
            with self.assertRaises(ValueError):
                partition_candidates(a,candidates)

    def test_roles_reject_overlap_missing_anchor_and_trigger_changes(self):
        base=role_fixture(); anchors=base[base.slot.eq(0)]
        self.assertEqual(validate_roles(base,anchors)["rotatable_scenarios"],1)
        changed=base.copy()
        changed.loc[changed.role.eq("holdout"),["src","dst"]]=[1,0]
        with self.assertRaises(ValueError):
            validate_roles(changed,anchors)
        for column,value in (("split","val"),("allocation_alpha",.3),("displacement_m",.4),
                             ("motion_regime",SLOW),("min_endpoint_speed_mps",float("nan"))):
            changed=base.copy(); changed.loc[0,column]=value
            with self.assertRaises(ValueError):
                validate_roles(changed,anchors)
        with self.assertRaises(ValueError):
            validate_roles(base[~base.index.isin([0])],anchors)

    def test_noninteger_slot_and_missing_holdout_fail(self):
        f=role_fixture(); a=f[f.slot.eq(0)]
        changed=f.copy(); changed["slot"]=changed.slot.astype(float);changed.loc[1,"slot"]=1.5
        with self.assertRaises(ValueError):
            validate_roles(changed,a)
        with self.assertRaises(ValueError):
            validate_roles(f[f.role.eq("train")],a)

    def test_schedule_keeps_counts_quota_and_never_selects_holdout(self):
        f=role_fixture()
        before=torch.random.get_rng_state().clone()
        exposure={}
        for epoch in range(1,51):
            selected=epoch_rows(f,epoch)
            self.assertEqual(len(selected),2)
            self.assertEqual(set(selected.motion_regime),{MOVING,SLOW})
            self.assertTrue(selected.role.eq("train").all())
            self.assertEqual(int(selected[selected.motion_regime.eq(SLOW)].slot.iloc[0]),0)
            slot=int(selected[selected.motion_regime.eq(MOVING)].slot.iloc[0])
            exposure[slot]=exposure.get(slot,0)+1
        self.assertEqual(exposure,{0:25,1:25})
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        self.assertEqual(schedule_digest(epoch_rows(f,1)),schedule_digest(epoch_rows(f.sample(frac=1,random_state=3),1)))
        for epoch in (0,-1,1.2,True):
            with self.assertRaises(ValueError):
                epoch_rows(f,epoch)

    def test_role_file_hash_tampering_fails_closed(self):
        f=role_fixture(); anchors=f[f.slot.eq(0)]
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            f.to_csv(p/"targets.csv",index=False)
            pd.DataFrame([target(8,9,"unrelated")]).to_csv(p/"seen.csv",index=False)
            for name in ("graph","poison"):
                (p/name).write_text(name)
            assets={name+"_manifest":{"path":str(p/name),"sha256":sha256_file(p/name)}
                    for name in ("graph","poison")}
            assets["previous_diagnostic_targets"]={"path":str(p/"seen.csv"),"sha256":sha256_file(p/"seen.csv")}
            plan={"version":VERSION,"test_access_authorized":False,"selection_seed":SELECTION_SEED,
                  "max_train_targets":MAX_TRAIN_TARGETS,"holdout_gates":HOLDOUT_GATES,
                  "assets":assets,"targets":{"filename":"targets.csv","sha256":sha256_file(p/"targets.csv")},
                  "coverage":validate_roles(f,anchors)}
            (p/"plan.json").write_text(json.dumps(plan))
            kwargs=dict(anchors=anchors,graph_manifest_hash=assets["graph_manifest"]["sha256"],poison_hash=assets["poison_manifest"]["sha256"])
            with patch("jcas.core.target_rotation.MIN_HOLDOUT_SCENARIOS",1):
                loaded,_=load_plan(p/"plan.json",**kwargs)
                self.assertEqual(len(loaded),4)
                (p/"targets.csv").write_text((p/"targets.csv").read_text()+"\n")
                with self.assertRaises(ValueError):
                    load_plan(p/"plan.json",**kwargs)


class RotationDatasetTests(unittest.TestCase):
    def test_first_epoch_equals_old_presentation_and_second_changes_only_target_row(self):
        ds,original=dataset_fixture()
        self.assertEqual(len(ds),len(original))
        for index,item in enumerate(original):
            self.assertIs(ds[index],item)
        self.assertEqual(ds.finish_epoch()["triggered_rows"],2)
        ds.begin_epoch(2)
        for index,item in enumerate(original):
            if item.has_counterfactual_pair and item.graph.scenario_id=="moving":
                self.assertIsNot(ds[index],item)
            else:
                self.assertIs(ds[index],item)
        self.assertEqual(ds.finish_epoch()["rotated_away_from_anchor"],1)

    def test_missing_and_duplicate_exposure_fail(self):
        ds,_=dataset_fixture()
        with self.assertRaises(RuntimeError):
            ds.finish_epoch()
        for i in range(len(ds)):
            ds[i]
        ds[list(ds.by_index)[0]]
        with self.assertRaises(RuntimeError):
            ds.finish_epoch()

    def test_wrong_target_and_changed_topology_fail(self):
        ds,items=dataset_fixture()
        ds.variants["moving"][1].graph.poisoned_edge_mask[:]=False
        with self.assertRaises(ValueError):
            EpochTargetDataset(items,ds.variants,role_fixture())
        ds,items=dataset_fixture()
        ds.variants["moving"][1].graph.message_passing_edge_mask[0]=False
        with self.assertRaises(ValueError):
            EpochTargetDataset(items,ds.variants,role_fixture())

    def test_variant_loading_refuses_non_train_before_graph_access(self):
        ds,items=dataset_fixture()
        manifest=pd.DataFrame([{"scenario_id":"moving","split":"val","graph_sha256":"a"*64}])
        with patch("jcas.workflows.trainer.load_split_graphs") as load:
            with self.assertRaises(ValueError):
                load_rotating_dataset(items,role_fixture(),manifest,Path("/unused"),None)
            load.assert_not_called()

    def test_synthetic_mixed_projected_steps_consume_exact_schedule(self):
        from functools import partial
        ds,_=dataset_fixture()
        torch.manual_seed(7)
        model=build_model(model_name="genconv",node_dim=1,edge_dim=1,hidden_dim=8,num_layers=1,decoder_hidden_dim=8)
        loader=DataLoader(ds,batch_size=2,shuffle=True,num_workers=0,
                          generator=torch.Generator().manual_seed(20260621),
                          collate_fn=partial(collate_counterfactual_items,include_clean_base=True))
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        for epoch in (1,2,3):
            ds.begin_epoch(epoch)
            result=train_counterfactual_epoch(model,loader,optimizer,config=BALANCED,
                                             projection=CleanProjectionConfig(),device=torch.device("cpu"))
            schedule=ds.finish_epoch()
            self.assertEqual(result["counterfactual_pairs"],2)
            self.assertEqual(schedule["holdout_targets_presented"],0)
            self.assertEqual(result["gradient_projection"]["constraint_violations"],0)
            self.assertEqual(schedule["schedule_sha256"],schedule_digest(epoch_rows(role_fixture(),epoch)))


def holdout_fixture():
    def model(asr):
        return {"targets":{"incremental_flip_rate_all_targets":asr,"clean_activation_rate":.05},
                "collateral":{"adjacent_negative":{"incremental_positive_rate":.005}}}
    return {"models":{"rotating_v4":model(.35),"fixed_v3":model(.20)}}


class RotationProtocolTests(unittest.TestCase):
    def test_original_gate_and_new_holdout_gate_are_both_required(self):
        result,audit=gate_fixture(); held=holdout_fixture()
        gate=combined_gate(result,audit,held)
        self.assertEqual(gate["status"],"PASS")
        self.assertEqual(len(gate["checks"]),9)
        self.assertFalse(gate["attack_validation_started"])
        result["val_pair_metrics"]["auc"]=.98499
        self.assertEqual(combined_gate(result,audit,held)["status"],"STOP")

    def test_strong_anchor_cannot_mask_failed_holdout_or_missing_finite_metrics(self):
        result,audit=gate_fixture()
        for value in (.299999,None,float("nan"),float("inf"),True):
            held=holdout_fixture()
            held["models"]["rotating_v4"]["targets"]["incremental_flip_rate_all_targets"]=value
            self.assertEqual(combined_gate(result,audit,held)["status"],"STOP")
        held=holdout_fixture();held["models"]["fixed_v3"]["targets"]["incremental_flip_rate_all_targets"]=.3
        self.assertEqual(combined_gate(result,audit,held)["status"],"STOP")

    def test_holdout_collateral_threshold_is_not_rounded(self):
        result,audit=gate_fixture();held=holdout_fixture()
        held["models"]["rotating_v4"]["collateral"]["adjacent_negative"]["incremental_positive_rate"]=.00500001
        self.assertEqual(combined_gate(result,audit,held)["status"],"STOP")

    def test_command_locks_hyperparameters_and_has_no_test_option(self):
        plan={"assets":{"graph_manifest":{"path":"/train/manifest.csv"},"poison_manifest":{"path":"/train/poison.csv"}}}
        command=training_command("/plan.json",plan,Path("/output"),"cpu")
        with patch("sys.argv",["trainer",*command[3:]]):
            args=parse_args()
        validate_rotation_arguments(args)
        self.assertNotIn("--evaluate-test",command)
        self.assertEqual(args.target_rotation_plan,"/plan.json")
        for key,value in (("lr",.002),("epochs",51),("max_train_graphs",10),("target_weight",.2)):
            changed=copy.copy(args);setattr(changed,key,value)
            with self.assertRaises(ValueError):
                validate_rotation_arguments(changed)

    def test_old_pilot_refuses_rotation_result(self):
        with self.assertRaisesRegex(ValueError,"fixed-target"):
            check_candidate({"config":{"target_rotation_plan":"/plan"}},{})

    def test_resume_rejects_wrong_plan_before_reading_history(self):
        fake={"config":{"target_rotation_plan":"/wrong"},"target_rotation":{"plan_sha256":"wrong"}}
        with patch("jcas.workflows.rotating_counterfactual_pilot.fixed.check_candidate"), \
             patch("jcas.workflows.rotating_counterfactual_pilot.sha256_file",return_value="expected"):
            with self.assertRaises(ValueError):
                validate_candidate(fake,Path("/plan"),{"assets":{},"targets":{"sha256":"target"},"coverage":{}},role_fixture())


if __name__ == "__main__":
    unittest.main()

