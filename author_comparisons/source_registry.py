"""Provenance and task-compatibility registry for uploaded author sources.

This module is metadata and a read-only archive audit. It does not implement
any selector, trigger, loss, optimizer, victim model, or evaluation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = Path("/home/mizhou")
SOURCE_ROOT = ROOT / "third_party" / "author_baselines"

SOURCES = {
    "spear": {
        "archive": ARCHIVE_ROOT / "SPEAR-main.zip",
        "archive_sha256": "be18c0145089c0f063a6f863bf76b7de5d7654f65e1aa7427a54afe2505e0757",
        "archive_root": "SPEAR-main", "checkout": SOURCE_ROOT / "SPEAR",
        "commit": "187d2cdc1f2454795dc8b37467ee2103e2613b15",
        "repository": "https://github.com/yhDing/SPEAR",
        "entrypoints": ["main.py", "select_sample.py", "select_feature.py"],
        "native_task": "inductive_node_classification",
        "pair_adapter": "primary", "status": "source_backed_primary",
    },
    "ugba": {
        "archive": ARCHIVE_ROOT / "UGBA-main.zip",
        "archive_sha256": "46f8f91da84c5b14d09bed65a12cde25145d983d60bc2efebfcc0f52d891ff53",
        "archive_root": "UGBA-main", "checkout": SOURCE_ROOT / "UGBA",
        "commit": "c1d945a215dd16fd806f8a231381d303a0b811f7",
        "repository": "https://github.com/ventr1c/UGBA",
        "entrypoints": ["run_adaptive.py", "run_GTA.py", "models/backdoor.py"],
        "native_task": "inductive_node_classification",
        "pair_adapter": "primary", "status": "source_backed_primary",
    },
    "dpgba": {
        "archive": ARCHIVE_ROOT / "DPGBA-main.zip",
        "archive_sha256": "eebb9c445eb20df97d870f8414cdef7b57f0b7d6b1c6142fc80ae5860d7ee203",
        "archive_root": "DPGBA-main", "checkout": SOURCE_ROOT / "DPGBA",
        "commit": "d974ea615900e5d0590bb473da9d7dd904f8c908",
        "repository": "https://github.com/zzwjames/DPGBA",
        "entrypoints": ["DPGBA/run/run_adaptive.py", "DPGBA/run/models/reconstruct.py"],
        "native_task": "inductive_node_classification",
        "pair_adapter": "primary", "status": "source_backed_primary",
    },
    "gta_graphbackdoor": {
        "archive": ARCHIVE_ROOT / "GraphBackdoor-main.zip",
        "archive_sha256": "ec4e5b5f5b0e767521cf24df79d7c0e567aa8e0302da9a2b73a41caff4fae441",
        "archive_root": "GraphBackdoor-main", "checkout": SOURCE_ROOT / "GTA",
        "commit": "3d975d78813f2a4a4960f92f9b66847dc19413a8",
        "repository": "https://github.com/zaixizhang/GraphBackdoor",
        "entrypoints": ["main/attack.py", "trojan/GTA.py"],
        "native_task": "graph_classification",
        "pair_adapter": "secondary_pair_readout_only",
        "status": "source_backed_secondary",
    },
    "motif_backdoor": {
        "archive": ARCHIVE_ROOT / "Motif-Backdoor-main.zip",
        "archive_sha256": "91751d0606efa105d32c39efa892c81100f8df0e1996666f930e6bfe614c50eb",
        "archive_root": "Motif-Backdoor-main", "checkout": SOURCE_ROOT / "Motif-Backdoor",
        "commit": "8e69c3bbb3dc09cb79045d2ccdad2766a655916d",
        "repository": "https://github.com/zaixizhang/Motif-Backdoor",
        "entrypoints": ["main.py", "attack_main.py"],
        "native_task": "graph_classification",
        "pair_adapter": "incompatible_with_pair_node_task",
        "status": "source_audit_only",
    },
    "cgba": {
        "archive": None, "archive_sha256": None, "archive_root": None,
        "checkout": None, "commit": None,
        "repository": "https://ojs.aaai.org/index.php/AAAI/article/view/35466",
        "entrypoints": [], "native_task": "inductive_node_classification",
        "pair_adapter": "unverified_author_source",
        "status": "excluded_no_local_author_archive",
    },
}

SOURCE_SUFFIXES = {".py", ".sh", ".md"}
SOURCE_FILENAMES = {".gitignore"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _relative_member(name: str, root: str) -> str | None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != root:
        raise ValueError(f"unsafe source archive member: {name}")
    relative = PurePosixPath(*path.parts[1:])
    if not relative.parts or (relative.suffix not in SOURCE_SUFFIXES and relative.name not in SOURCE_FILENAMES):
        return None
    return str(relative)


def _archive_source_hashes(entry: dict) -> dict[str, str]:
    if entry["archive"] is None:
        return {}
    archive = Path(entry["archive"])
    if not archive.is_file() or sha256(archive) != entry["archive_sha256"]:
        raise ValueError(f"archive missing or hash changed: {archive}")
    result = {}
    with zipfile.ZipFile(archive) as zf:
        if entry["commit"] and zf.comment.decode().strip() != entry["commit"]:
            raise ValueError(f"archive commit comment changed: {archive}")
        for info in zf.infolist():
            if info.is_dir():
                continue
            relative = _relative_member(info.filename, entry["archive_root"])
            if relative is not None:
                result[relative] = hashlib.sha256(zf.read(info)).hexdigest()
    return result


def audit_source(name: str) -> dict:
    """Compare one uploaded archive with its extracted source files."""
    entry = dict(SOURCES[name])
    archive_hashes = _archive_source_hashes(SOURCES[name])
    checkout = Path(entry["checkout"]) if entry["checkout"] is not None else None
    missing, mismatched, checkout_hashes = [], [], {}
    if archive_hashes and checkout is None:
        missing = sorted(archive_hashes)
    elif archive_hashes:
        for relative, expected in archive_hashes.items():
            path = checkout / relative
            if not path.is_file() or path.is_symlink():
                missing.append(relative)
                continue
            actual = sha256(path)
            checkout_hashes[relative] = actual
            if actual != expected:
                mismatched.append(relative)
    entry.update(
        archive=str(entry["archive"]) if entry["archive"] is not None else None,
        checkout=str(checkout) if checkout is not None else None,
        archive_source_files=len(archive_hashes),
        checkout_source_files=len(checkout_hashes),
        missing_source_files=sorted(missing),
        mismatched_source_files=sorted(mismatched),
        source_bytes_match=bool(archive_hashes) and not missing and not mismatched,
        author_algorithm_reimplemented=False,
        bundled_data_loaded=False,
    )
    return entry


def audit_all() -> dict[str, dict]:
    return {name: audit_source(name) for name in SOURCES}


def primary_methods() -> tuple[str, ...]:
    return tuple(name for name, entry in SOURCES.items() if entry["status"] == "source_backed_primary")


def assert_primary_sources() -> dict[str, dict]:
    audited = audit_all()
    for name in primary_methods():
        if not audited[name]["source_bytes_match"]:
            raise ValueError(f"primary source checkout is not byte-identical: {name}")
    return audited


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = {"primary_methods": list(primary_methods()), "sources": audit_all()}
    print(json.dumps(result, indent=2, sort_keys=True) if args.json else
          "\n".join(f"{name}: {row['status']} source_bytes_match={row['source_bytes_match']}"
                    for name, row in result["sources"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
