import unittest

from author_comparisons.source_registry import audit_all, primary_methods


class AuthorSourceRegistryTests(unittest.TestCase):
    def test_primary_sources_are_uploaded_bytes(self):
        rows = audit_all()
        self.assertEqual(primary_methods(), ("spear", "ugba", "dpgba"))
        for name in primary_methods():
            self.assertTrue(rows[name]["source_bytes_match"], name)
            self.assertFalse(rows[name]["author_algorithm_reimplemented"])

    def test_task_mismatched_sources_are_not_primary(self):
        rows = audit_all()
        self.assertEqual(rows["gta_graphbackdoor"]["native_task"], "graph_classification")
        self.assertEqual(rows["motif_backdoor"]["pair_adapter"], "incompatible_with_pair_node_task")
        self.assertNotIn("gta_graphbackdoor", primary_methods())
        self.assertNotIn("motif_backdoor", primary_methods())
        self.assertEqual(rows["cgba"]["status"], "excluded_no_local_author_archive")


if __name__ == "__main__":
    unittest.main()
