#!/usr/bin/env python3
"""
Generate Depth Anything V2 pseudo-depth labels for the manifests. (Phase 3)

These labels are RELATIVE depth, not metric. Run the 64-image smoke test first:

    python scripts/02_pseudo_depth.py --smoke

then the full pass:

    python scripts/02_pseudo_depth.py
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vise.config import (
    ensure_utf8_stdio,
    get_logger,
    load_config,
    seed_everything,
    setup_logging,
)
from vise.pseudo_depth import (
    SPLITS,
    depth_structure_report,
    generate,
    load_depth_pipeline,
    save_sanity_grid,
    update_manifests,
)


def main() -> None:
    ensure_utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="64 train images only; writes the sanity grid, does not touch manifests",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS)
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    data = cfg["data"]
    setup_logging(
        output_dir=cfg["experiment"]["output_dir"],
        experiment_name="phase3_smoke" if args.smoke else "phase3_pseudo_depth",
    )
    seed_everything(cfg["experiment"]["seed"])
    logger = get_logger("phase3")

    manifest_dir = Path(data["manifest_dir"])
    out_dir = Path(data["pseudo_depth_dir"])
    pipe = load_depth_pipeline()

    if args.smoke:
        logger.info("SMOKE TEST — 64 train images, manifests untouched")
        train = pd.read_csv(manifest_dir / "train.csv")
        df = generate(
            train.sample(frac=1.0, random_state=cfg["experiment"]["seed"]),
            minc_root=data["minc_root"],
            out_dir=out_dir,
            pipe=pipe,
            batch_size=args.batch_size,
            limit=64,
            overwrite=args.overwrite,
        )
        save_sanity_grid(
            df,
            minc_root=data["minc_root"],
            out_path=Path(cfg["evaluation"]["figures_dir"])
            / "pseudo_depth_sanity.png",
        )
        rep = depth_structure_report(df)
        logger.info("Per-class depth structure (lowest = flattest):")
        for _, r in rep.iterrows():
            logger.info(
                "  %-16s n=%-3d mean std=%.4f", r["minc_class"], r["n"],
                r["mean_depth_std"],
            )
        logger.info(
            "Smoke test complete. Inspect the grid, then run without --smoke."
        )
        return

    updated = {}
    for s in args.splits:
        logger.info("=== split: %s ===", s)
        m = pd.read_csv(manifest_dir / f"{s}.csv")
        updated[s] = generate(
            m,
            minc_root=data["minc_root"],
            out_dir=out_dir,
            pipe=pipe,
            batch_size=args.batch_size,
            limit=args.limit,
            overwrite=args.overwrite,
        )
        missing = [p for p in updated[s]["depth_path"] if not Path(p).exists()]
        if missing:
            raise RuntimeError(
                f"{s}: {len(missing)} depth maps missing after generation, "
                f"e.g. {missing[:3]}"
            )
        logger.info("%s: file count matches manifest rows (%d)", s, len(updated[s]))

    if args.limit is None:
        update_manifests(updated, manifest_dir, splits=args.splits)
    else:
        logger.warning("--limit set: manifests NOT updated (partial run)")


if __name__ == "__main__":
    main()
