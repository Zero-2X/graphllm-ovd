"""Attach the original experiment's fixed inputs to a separate repository checkout."""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = Path("experiments/factored_instance_v4/routed_association")
OBJECTS = WORK/"objects365"
SOURCE_ARTIFACTS = (
    "manifest.json",
    "rare_manifest.json",
    "best_repair.pt",
    "qualification_repair.json",
    "receiver_gain_gate",
    "receiver_keep_gate",
    "receiver_native_evidence",
    "donor_view_entropy",
    "graphllm_local_message_v1",
    "graphllm_local_message_unique_v2",
)


def attach(asset_root):
    asset_root = asset_root.resolve()
    if asset_root == ROOT:
        raise ValueError("Use the original experiment root as --asset-root.")
    paths = [OBJECTS/name for name in SOURCE_ARTIFACTS]
    paths.extend((
        OBJECTS/"graphllm_receiver_evidence_v2/manifest.json",
        Path("experiments/hierarchical_instance_v1/best.pt"),
    ))
    for relative in paths:
        source = asset_root/relative
        target = ROOT/relative
        if not source.exists():
            raise FileNotFoundError(source)
        if target.is_symlink() and target.resolve() == source.resolve():
            continue
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
    for relative in paths:
        source = asset_root/relative
        target = ROOT/relative
        if not target.is_symlink():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source, target_is_directory=source.is_dir())
    print(f"Linked {len(paths)} input paths from {asset_root}. Train and evaluation outputs belong to this checkout.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path, help="original experiment root on sr008")
    attach(parser.parse_args().asset_root)
