"""Source-only GTA checks; no DGL emulation, CUDA execution, or AV2 reads."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
import numpy as np

from author_native_runs import gta_native as gta


class NativeGTAAuditTests(unittest.TestCase):
    def test_archive_identity_and_required_sources(self):
        contents, record = gta.verified_sources()
        self.assertEqual(record["archive_comment_commit"], gta.AUTHOR_COMMIT)
        self.assertEqual(record["archive_sha256"], gta.ARCHIVE_SHA256)
        self.assertIn("toponet", contents["trojan/GTA.py"])
        self.assertFalse(record["archive_modified"])
        self.assertFalse(record["source_extracted"])

    def test_native_gradient_route_is_not_silently_repaired(self):
        contents, _ = gta.verified_sources()
        rng = torch.get_rng_state().clone()
        result = gta.gradient_route_probe(contents)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(result["topology_output_requires_grad"])
        self.assertTrue(result["models"]["gcn"]["conversion_succeeded"])
        self.assertFalse(result["models"]["gcn"]["reconstructed_tensor_requires_grad"])
        self.assertFalse(result["models"]["sage"]["conversion_succeeded"])
        self.assertIn("requires grad", result["models"]["sage"]["error"])
        self.assertFalse(result["gpu_used"])
        self.assertFalse(result["actual_gnn_forward_or_training_run"])

    def test_audit_does_not_claim_training_or_gpu_acceptance(self):
        report = gta.audit()
        self.assertFalse(report["training_started"])
        self.assertFalse(report["graph_arrays_loaded"])
        self.assertFalse(report["generator_gradient_repair_applied"])
        self.assertFalse(report["dgl_reimplemented"])
        self.assertFalse(report["environment"]["dgl_cuda_execution_verified"])
        self.assertTrue(report["implemented_pair_task_adapter_contract"]["no_ego_crop"])
        json.dumps(report, allow_nan=False)

    def test_changed_archive_fails_closed(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "not-the-author.zip"
            path.write_bytes(b"synthetic altered archive")
            with self.assertRaisesRegex(ValueError, "archive changed"):
                gta.verified_sources(path)

    def test_runtime_retains_native_training_and_generator_code(self):
        original, _ = gta.verified_sources()
        changed = gta.runtime_sources(original)
        for name in ("trojan/GTA.py", "trojan/input.py", "trojan/prop.py", "main/benign.py",
                     "utils/bkdcdd.py", "utils/mask.py"):
            self.assertEqual(changed[name], original[name], name)
        final_state_capture = ("        self.final_model = model\n"
                               "        self.final_toponet = toponet\n"
                               "        self.final_featnet = featnet\n")
        self.assertEqual(changed["main/attack.py"].replace(final_state_capture, ""),
                         original["main/attack.py"])
        self.assertIn("adj.cpu().detach().T.numpy()", changed["model/gcn.py"])
        self.assertIn("x[torch.arange(B, device=x.device), data[5]]", changed["model/gcn.py"])
        self.assertIn("fn.copy_u(u='h', out='m')", changed["model/gcn.py"])
        self.assertNotIn("x = torch.max(x, dim=1)[0].squeeze()", changed["model/gcn.py"])

    def test_empty_edge_guard_preserves_entire_original_nonempty_conversion(self):
        original, _ = gta.verified_sources()
        changed = gta.runtime_sources(original)
        self.assertEqual(changed["utils/graph.py"].count(gta.EMPTY_GRAPH_GUARD), 1)
        self.assertEqual(changed["utils/graph.py"].replace(gta.EMPTY_GRAPH_GUARD, ""), original["utils/graph.py"])
        self.assertIn("G.number_of_edges() == 0", gta.EMPTY_GRAPH_GUARD)
        self.assertIn("torch.empty(0, dtype=torch.as_tensor(A).dtype)", gta.EMPTY_GRAPH_GUARD)
        self.assertNotIn("add_self_loop", changed["utils/graph.py"])
        self.assertNotIn("remove_nodes", changed["utils/graph.py"])

    def test_pair_reader_retains_whole_scene_and_original_target_identity(self):
        clean = {"x": torch.arange(5 * 784, dtype=torch.float32).reshape(5, 784),
            "y": torch.tensor([0, 1, -1, 1, 0]), "scene_ptr": torch.tensor([0, 3, 5]),
            "edge_index": torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]]),
            "idx_train": torch.tensor([0, 1]), "idx_val": torch.tensor([3, 4])}
        reader, mapping = gta.pair_reader(clean, 21)
        self.assertEqual(mapping, {0: 0, 1: 1, 3: 2, 4: 3})
        self.assertEqual(reader.data["target_nodes"], [0, 1, 0, 1])
        self.assertEqual(reader.data["labels"], [0, 1, 1, 0])
        self.assertEqual([len(x) for x in reader.data["features"]], [3, 3, 2, 2])
        np.testing.assert_array_equal(reader.data["features"][0], clean["x"][:3].numpy())
        self.assertEqual(reader.data["splits"], {"train": [0, 1], "test": [2, 3]})
        clean["edge_index"][0, 0] = 4
        with self.assertRaisesRegex(ValueError, "cross-scene"):
            gta.pair_reader(clean, 21)

    def test_pair_reader_keeps_empty_scene_nodes_labels_and_target_indices(self):
        clean = {"x": torch.arange(4 * 784, dtype=torch.float32).reshape(4, 784),
            "y": torch.tensor([0, -1, 0, 1]), "scene_ptr": torch.tensor([0, 1, 4]),
            "edge_index": torch.empty((2, 0), dtype=torch.long),
            "idx_train": torch.tensor([0]), "idx_val": torch.tensor([2, 3])}
        reader, mapping = gta.pair_reader(clean, 21)
        self.assertEqual(mapping, {0: 0, 2: 1, 3: 2})
        self.assertEqual(reader.data["target_nodes"], [0, 1, 2])
        self.assertEqual(reader.data["labels"], [0, 0, 1])
        for record, (lo, hi) in enumerate(((0, 1), (1, 4), (1, 4))):
            np.testing.assert_array_equal(reader.data["features"][record], clean["x"][lo:hi].numpy())
            np.testing.assert_array_equal(reader.data["adj_list"][record], np.zeros((hi - lo, hi - lo)))

    def test_target_index_survives_original_collation(self):
        source, _ = gta.verified_sources()
        patched = gta.runtime_sources(source)
        namespace = {}
        exec(compile(patched["utils/batch.py"], "native_patched_collate", "exec"), namespace)
        result = namespace["collate_batch"]([[torch.ones(3, 4), torch.eye(3), 1, 2],
                                             [torch.ones(2, 4), torch.eye(2), 0, 0]])
        self.assertEqual(len(result), 6)
        self.assertEqual(result[5].tolist(), [2, 0])
        self.assertEqual(result[4].tolist(), [1, 0])
        self.assertEqual(result[2].tolist(), [[1., 1., 1.], [1., 1., 0.]])

    def test_synthetic_smoke_fixture_exercises_small_scene_public_target(self):
        clean = gta._synthetic_clean_fixture()
        reader, mapping = gta.pair_reader(clean, 21)
        self.assertEqual(len(clean["idx_atk"]), 4)
        self.assertEqual(clean["y"][clean["idx_atk"]].tolist(), [0, 0, 0, 0])
        self.assertEqual(int(clean["moving_mask"][clean["idx_atk"]].sum()), 2)
        self.assertEqual(set(clean["y"][clean["idx_val"]].tolist()), {0, 1})
        target_sizes = [len(reader.data["features"][mapping[node]]) for node in clean["idx_atk"].tolist()]
        self.assertEqual(target_sizes, [15, 15, 4, 8])
        self.assertEqual(reader.data["target_nodes"][mapping[30]], 0)
        np.testing.assert_array_equal(reader.data["adj_list"][mapping[30]], np.zeros((4, 4)))


if __name__ == "__main__":
    unittest.main()
