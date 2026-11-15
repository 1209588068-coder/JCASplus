"""Opt-in GPU-resident numeric scene-batch cache for AV2 experiments.

The current formal runner rebuilds every scene batch on CPU each epoch.  This
module preserves the existing complete-scene order, offsets, labels, masks,
and edge topology while materializing only the numeric training tensors once
on a CUDA device.  It is not imported by the active frozen retry run.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch

try:
    from .av2_formal_data import iter_role_batches
except ImportError:
    from spear_current.av2_formal_data import iter_role_batches

_NUMERIC_KEYS = ("x", "edge_index", "y", "eligible")
_CACHE: dict[tuple[str, str, int, str, int | None], tuple["CachedBatch", ...]] = {}


@dataclass(frozen=True)
class CachedBatch:
    x: torch.Tensor
    edge_index: torch.Tensor
    y: torch.Tensor
    eligible: torch.Tensor
    global_start: int
    global_stop: int

    def as_dict(self) -> dict[str, object]:
        return {"x": self.x, "edge_index": self.edge_index,
                "y": self.y, "eligible": self.eligible,
                "global_start": self.global_start,
                "global_stop": self.global_stop}


def _device(device: str | torch.device) -> torch.device:
    value = torch.device(device)
    if value.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("GPU cache requires an available CUDA device")
    index = torch.cuda.current_device() if value.index is None else value.index
    if index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {index} is unavailable")
    return torch.device("cuda", index)


def _to_cuda(value: torch.Tensor, device: torch.device, *, pin_memory: bool) -> torch.Tensor:
    value = value.detach().contiguous()
    if pin_memory:
        try:
            value = value.pin_memory()
        except RuntimeError:
            # Unified-memory hosts may not support pinned allocations.
            pass
    return value.to(device, non_blocking=pin_memory)


def build_cache(data_dir: str | Path, role: str, max_nodes: int,
                device: str | torch.device = "cuda:0", *,
                pin_memory: bool = False,
                limit_batches: int | None = None) -> tuple[CachedBatch, ...]:
    """Read complete scene batches once and keep numeric tensors on CUDA.

    ``limit_batches`` is intended only for the isolated smoke benchmark.  A
    full execution variant must omit it so all registered role scenes remain
    covered.
    """
    if type(max_nodes) is not int or max_nodes < 1:
        raise ValueError("max_nodes must be a positive integer")
    if limit_batches is not None and (type(limit_batches) is not int or limit_batches < 1):
        raise ValueError("limit_batches must be a positive integer")
    target = _device(device)
    key = (str(Path(data_dir).resolve()), str(role), max_nodes, str(target), limit_batches)
    if key in _CACHE:
        return _CACHE[key]
    result: list[CachedBatch] = []
    source = iter_role_batches(data_dir, role, max_nodes=max_nodes, device="cpu")
    for index, batch in enumerate(source):
        if limit_batches is not None and index >= limit_batches:
            break
        missing = [name for name in _NUMERIC_KEYS if name not in batch]
        if missing:
            raise ValueError(f"Batch missing numeric fields: {missing}")
        start, stop = int(batch["global_start"]), int(batch["global_stop"])
        if stop - start != len(batch["x"]):
            raise ValueError("Batch offsets disagree with x rows")
        result.append(CachedBatch(
            x=_to_cuda(batch["x"], target, pin_memory=pin_memory),
            edge_index=_to_cuda(batch["edge_index"], target, pin_memory=pin_memory),
            y=_to_cuda(batch["y"], target, pin_memory=pin_memory),
            eligible=_to_cuda(batch["eligible"], target, pin_memory=pin_memory),
            global_start=start, global_stop=stop))
    if not result:
        raise ValueError("No batches were materialized")
    torch.cuda.synchronize(target)
    frozen = tuple(result)
    _CACHE[key] = frozen
    return frozen


def batches(data_dir: str | Path, role: str, max_nodes: int,
            device: str | torch.device, *, pin_memory: bool = False,
            limit_batches: int | None = None) -> Iterator[dict[str, object]]:
    """Yield cached CUDA batches with the existing dictionary interface."""
    for item in build_cache(data_dir, role, max_nodes, device,
                            pin_memory=pin_memory, limit_batches=limit_batches):
        yield item.as_dict()


def clear_cache() -> None:
    _CACHE.clear()


def cache_summary(data_dir: str | Path, role: str, max_nodes: int,
                  device: str | torch.device) -> dict[str, object]:
    items = build_cache(data_dir, role, max_nodes, device)
    bytes_total = sum(int(t.numel() * t.element_size())
                      for item in items
                      for t in (item.x, item.edge_index, item.y, item.eligible))
    return dict(role=role, max_nodes=max_nodes, device=str(_device(device)),
                batches=len(items), nodes=sum(int(x.x.shape[0]) for x in items),
                edges=sum(int(x.edge_index.shape[1]) for x in items),
                numeric_bytes=bytes_total, numeric_gib=bytes_total / 1024**3,
                global_start=items[0].global_start,
                global_stop=items[-1].global_stop)
