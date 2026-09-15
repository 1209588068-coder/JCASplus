"""Epoch-dependent presentation with fixed row count, topology and shuffle RNG."""
from __future__ import annotations

from collections import Counter

import torch
from torch.utils.data import Dataset

from jcas.core.target_rotation import epoch_rows, schedule_digest


class EpochTargetDataset(Dataset):
    """Only appended triggered rows vary; all clean rows and row indices stay fixed."""

    def __init__(self, fixed_items, variants, roles):
        self.fixed_items = fixed_items
        self.roles = roles
        self.variants = variants
        self.triggered_indices = {}
        for index, item in enumerate(fixed_items):
            if item.has_counterfactual_pair:
                scene = str(item.graph.scenario_id)
                if scene in self.triggered_indices:
                    raise ValueError("duplicate appended scene")
                self.triggered_indices[scene] = index
        if set(self.triggered_indices) != set(variants):
            raise ValueError("variant scenes differ from fixed presentation")
        for scene, rows in roles[roles.role.eq("train")].groupby("scenario_id"):
            slots = sorted(rows.slot.astype(int).tolist())
            if sorted(variants[scene]) != slots:
                raise ValueError("loaded variants do not cover plan slots")
            clean = fixed_items[self.triggered_indices[scene]].clean_counterpart
            for row in rows.itertuples(index=False):
                item = variants[scene][int(row.slot)]
                graph = item.graph
                expected = {(int(row.src),int(row.dst)),(int(row.dst),int(row.src))}
                actual = {tuple(map(int,p)) for p in graph.edge_index[:,graph.poisoned_edge_mask].T.tolist()}
                if actual != expected or str(graph.scenario_id) != scene:
                    raise ValueError("variant graph target differs from role table")
                if (not torch.equal(graph.edge_index, clean.edge_index)
                        or not torch.equal(graph.supervision_edge_mask, clean.supervision_edge_mask)
                        or not torch.equal(graph.message_passing_edge_mask, clean.message_passing_edge_mask)):
                    raise ValueError("rotation changed topology/supervision denominator")
        self.by_index = {i:scene for scene,i in self.triggered_indices.items()}
        self.begin_epoch(1)

    def __len__(self):
        return len(self.fixed_items)

    def begin_epoch(self, epoch):
        selected = epoch_rows(self.roles, epoch)
        self.selected = selected
        self.slots = {str(r.scenario_id):int(r.slot) for r in selected.itertuples(index=False)}
        self.observed = Counter()
        self.epoch = epoch

    def __getitem__(self, index):
        scene = self.by_index.get(index)
        if scene is None:
            return self.fixed_items[index]
        self.observed[scene] += 1
        return self.variants[scene][self.slots[scene]]

    def finish_epoch(self):
        if dict(self.observed) != {scene:1 for scene in self.triggered_indices}:
            raise RuntimeError("each poison scene must be presented exactly once per epoch")
        return {
            "epoch":self.epoch, "triggered_rows":sum(self.observed.values()),
            "schedule_sha256":schedule_digest(self.selected),
            "motion_counts":{str(k):int(v) for k,v in self.selected.motion_regime.value_counts().items()},
            "rotated_away_from_anchor":int(self.selected.slot.ne(0).sum()),
            "holdout_targets_presented":0,
        }


def load_rotating_dataset(fixed_items, roles, manifest, graph_dir, label_config):
    from jcas.workflows.explicit_counterfactual_trainer import compose_counterfactual_training_items
    from jcas.workflows.trainer import load_split_graphs
    from jcas.core.motion_normalized_features import MOTION_NORMALIZED_EDGE_FEATURE_MODE

    clean_by_scene = {
        str(item.graph.scenario_id):item.graph for item in fixed_items
        if not item.has_counterfactual_pair
    }
    variants = {
        str(item.graph.scenario_id):{0:item} for item in fixed_items if item.has_counterfactual_pair
    }
    for slot in range(1,4):
        selected = roles[roles.role.eq("train") & roles.slot.eq(slot)].copy()
        if selected.empty:
            continue
        graph_rows = manifest[manifest.scenario_id.isin(selected.scenario_id)].copy()
        if (len(graph_rows) != len(selected) or not graph_rows.split.eq("train").all()
                or set(selected.scenario_id) - set(variants)):
            raise ValueError("rotation variant escaped original train scenes")
        hashes = graph_rows.set_index("scenario_id").graph_sha256
        if not selected.apply(lambda r:hashes.loc[r.scenario_id] == r.graph_sha256,axis=1).all():
            raise ValueError("rotation graph binding differs from manifest")
        graphs, stats = load_split_graphs(
            graph_rows,graph_dir,"train",True,None,label_config,selected,True,
            MOTION_NORMALIZED_EDGE_FEATURE_MODE,None,
        )
        if len(graphs) != len(selected) or stats["poisoned_graphs"] != len(selected):
            raise RuntimeError("some rotation variants failed to load")
        for graph in graphs:
            scene = str(graph.scenario_id)
            items, _ = compose_counterfactual_training_items([clean_by_scene[scene]],[graph])
            variants[scene][slot] = items[-1]
        print(f"[target_rotation] cached slot={slot} variants={len(graphs)}",flush=True)
    return EpochTargetDataset(fixed_items,variants,roles)

