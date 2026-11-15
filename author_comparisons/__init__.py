"""AV2 data/execution adapters around separately preserved author source.

The attack algorithms remain in the uploaded source trees under
``third_party/author_baselines``.  Files in this package only bridge the
frozen AV2 pair-node tensors, isolate source imports, record compatibility
patches, and run the common evaluator.  The legacy ``comparison_methods``
implementations are not used by the source-backed comparison queue.
"""
