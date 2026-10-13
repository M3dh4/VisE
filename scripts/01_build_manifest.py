#!/usr/bin/env python3
"""Build manifest CSVs from MINC-2500 + the stiffness lookup table. (Phase 2)"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vise.config import ensure_utf8_stdio, load_config, seed_everything, setup_logging
from vise.dataset import build_manifests, plot_split_coverage


def main() -> None:
    ensure_utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    ap.add_argument("--minc-root", default=None, help="overrides data.minc_root")
    ap.add_argument(
        "--no-strict-counts",
        action="store_true",
        help="skip the exact 29,750/1,750/3,500 assertion (subsetted trees)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    data = cfg["data"]
    setup_logging(
        output_dir=cfg["experiment"]["output_dir"], experiment_name="phase2_manifest"
    )
    seed_everything(cfg["experiment"]["seed"])

    manifests = build_manifests(
        minc_root=args.minc_root or data["minc_root"],
        lookup_csv=ROOT / data["lookup_csv"],
        split=data["split"],
        bin_width=data["log_e_bin_width"],
        sample_weighting=data.get("sample_weighting", "inverse_bin_freq"),
        out_dir=data["manifest_dir"],
        strict_counts=not args.no_strict_counts,
    )
    plot_split_coverage(
        manifests,
        out_path=Path(cfg["evaluation"]["figures_dir"]) / "split_logE_coverage.png",
    )


if __name__ == "__main__":
    main()
