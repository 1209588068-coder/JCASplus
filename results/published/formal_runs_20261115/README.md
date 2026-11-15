# Formal experiment result snapshot

This directory contains the portable JSON/Markdown snapshot of the current AV2 and WOMD formal runs as observed on 2026-11-15. It intentionally excludes datasets, graph tensors, model checkpoints, raw pair-score tables, logs, locks, and runtime caches.

- `av2_formal_main_full_corrected_gpu_cache_v2_20261111/`: five completed AV2 seeds.
- `womd_formal_main_corrected_fullcorpus_gpu_cache_v3_20261115/`: WOMD five-seed run snapshot; at capture time one seed was complete and seed 20261127 was active.

The JSON files preserve the recorded run contracts and metric summaries. Paths inside copied run contracts may refer to the original local worktree; `SNAPSHOT_MANIFEST.json` is the hash manifest for the files included in this public snapshot.
