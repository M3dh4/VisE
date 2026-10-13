"""
Manifest building, split integrity, and the torch Dataset. (Phase 2)

Limitations, restated wherever they apply downstream:
    - Stiffness labels are **class-level intervals**, not per-pixel ground truth.
      Every pixel of a patch receives the same target.
    - Depth targets (Phase 3) are **relative** pseudo-labels. A 362x362 crop
      carries no camera intrinsics, so they are not metric.

Design decisions recorded here rather than left implicit
--------------------------------------------------------
1. **Class-name normalisation.** The lookup table keys ``polished stone`` with a
   space; MINC's image directory is ``polished_stone``. Both sides of the join
   are normalised to spaces, and :func:`build_manifests` asserts that every
   included class matched a non-zero row count. Without that assertion a
   failed join silently drops 2,500 patches and the loss never surfaces.

2. **Bin-balanced sampling, and its conflict with the interval weighting.**
   ``sample_weight`` is inverse ``logE_bin`` frequency over the train split,
   per the build spec: MINC gives 2,500 images per *class*, but ten of the
   fourteen classes sit above 1 GPa, so the log-E axis is lumpy even though the
   classes are even. Balancing the axis is the stated intent.

   This pulls against the Phase 5 interval weighting. Measured on split 1,
   bin occupancy is:

   ===========  =========  ========  =============================================
   bin          classes    images    members
   ===========  =========  ========  =============================================
   +2.0         5          12,500    ceramic, glass, metal, mirror, polished stone
   +1.5         3           7,500    brick, stone, tile
   -2.5         2           5,000    carpet, fabric
   +0.5         2           5,000    paper, plastic
   -1.0         1           2,500    leather
   +1.0         1           2,500    wood
   ===========  =========  ========  =============================================

   So sampling upweights ``leather`` ~5x, while the interval-weighted loss gives
   leather the *lowest* weight (half-width 0.85 -> 1/(0.85+0.1) = 1.05, against
   glass at 4.39). The two schemes partly cancel. The realised weight range is
   logged by :func:`build_manifests` so the interaction is visible in the run
   log rather than assumed. Set ``data.sample_weighting: none`` to disable.

3. **``image_size`` is 364, not 362.** MINC patches are 362x362, which is not
   divisible by the ViT patch size of 14; 364 = 14 x 26. Whatever resize is used
   must be identical across Models A, B and C, since Phase 8 requires
   augmentation to be a controlled variable.
"""

from pathlib import Path
from typing import Dict, Iterable, Optional, Union

import numpy as np
import pandas as pd

from vise.config import get_logger
from vise.lookup import included, load_lookup

SPLIT_FILES = {"train": "train{n}.txt", "val": "validate{n}.txt", "test": "test{n}.txt"}

# MINC-2500 official split 1 is 2,125 / 125 / 250 per class. For the 14
# supervised classes that is exactly:
EXPECTED_ROWS = {"train": 29_750, "val": 1_750, "test": 3_500}

MANIFEST_COLS = [
    "image_path",
    "minc_class",
    "split",
    "E_min_GPa",
    "E_max_GPa",
    "E_median_GPa",
    "log10E_min",
    "log10E_max",
    "log10E_target",
    "log10E_halfwidth",
    "provenance",
    "logE_bin",
    "sample_weight",
]


class SplitIntegrityError(ValueError):
    """Raised when a split-integrity check fails. Never silently repaired."""


def normalise_class(name: str) -> str:
    """
    Normalise a MINC class name for display: underscores to spaces, lowercase,
    collapsed whitespace. ``polished_stone`` -> ``polished stone``.

    This is *not* the join key — see :func:`class_key`.
    """
    return " ".join(str(name).replace("_", " ").lower().split())


def class_key(name: str) -> str:
    """
    Canonical join key: lowercase with every non-alphanumeric character removed.

    MINC-2500 spells this class ``polishedstone`` — one word, no separator —
    while the lookup table and the published papers write ``polished stone``.
    Normalising separators to spaces is therefore not enough; the key strips
    them entirely so ``polishedstone``, ``polished_stone`` and
    ``polished stone`` all collapse to ``polishedstone``.

    Verified against the real ``minc-2500/categories.txt``: all 23 class names
    are distinct under this key, so stripping separators cannot merge two
    genuinely different classes.
    """
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def parse_split_file(path: Union[str, Path], split: str) -> pd.DataFrame:
    """
    Parse one MINC split file into ``image_path`` / ``minc_class`` / ``split``.

    Each line is ``images/<class>/<file>.jpg``; the class is the second path
    segment. Blank lines are skipped. Paths are stored with forward slashes so
    manifests are identical on Windows and Linux.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")

    rows = []
    malformed = []
    for lineno, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.strip().replace("\\", "/")
        if not line:
            continue
        parts = line.split("/")
        if len(parts) < 3:
            malformed.append(f"line {lineno}: {raw!r}")
            continue
        rows.append((line, normalise_class(parts[1]), split))

    if malformed:
        raise SplitIntegrityError(
            f"{path.name} has {len(malformed)} malformed line(s):\n  - "
            + "\n  - ".join(malformed[:10])
        )
    return pd.DataFrame(rows, columns=["image_path", "minc_class", "split"])


def _check_integrity(df: pd.DataFrame, expected_classes: Iterable[str]) -> None:
    """
    Run the three mandatory split-integrity checks, raising on any failure and
    reporting every problem at once.
    """
    logger = get_logger("dataset")
    expected = sorted(expected_classes)
    errors = []

    # 1. No image_path in more than one split.
    dupes = df[df.duplicated("image_path", keep=False)]
    if not dupes.empty:
        sample = dupes["image_path"].drop_duplicates().head(5).tolist()
        errors.append(
            f"CHECK 1 FAILED: {dupes['image_path'].nunique()} path(s) appear in "
            f"more than one split, e.g. {sample}"
        )
    else:
        logger.info("CHECK 1 passed: no image_path appears in more than one split")

    # 2. Every included class present in all three splits.
    missing = {
        s: sorted(set(expected) - set(g["minc_class"]))
        for s, g in df.groupby("split")
    }
    missing = {s: m for s, m in missing.items() if m}
    if missing:
        errors.append(f"CHECK 2 FAILED: classes missing from splits: {missing}")
    else:
        logger.info(
            "CHECK 2 passed: all %d included classes present in all three splits",
            len(expected),
        )

    # 3. No excluded class leaked through.
    leaked = sorted(set(df["minc_class"]) - set(expected))
    if leaked:
        errors.append(f"CHECK 3 FAILED: excluded classes present: {leaked}")
    else:
        logger.info("CHECK 3 passed: no excluded class leaked through the filter")

    if errors:
        raise SplitIntegrityError(
            "Split integrity checks failed:\n  - " + "\n  - ".join(errors)
        )


def _add_weights(
    df: pd.DataFrame, bin_width: float, scheme: str = "inverse_bin_freq"
) -> pd.DataFrame:
    """
    Add ``logE_bin`` and ``sample_weight``.

    ``logE_bin`` = round(log10E_target / bin_width). ``sample_weight`` is the
    inverse bin frequency **computed on the train split only** and normalised to
    mean 1.0 over that split, so the weights do not depend on val/test size.
    Val and test rows receive the weight of their bin (unused at eval time, but
    kept so the column is well defined everywhere).

    See limitation 2 in the module docstring: this deliberately fights the Phase
    5 interval weighting, and the realised range is logged by the caller.
    """
    logger = get_logger("dataset")
    df = df.copy()
    df["logE_bin"] = np.round(df["log10E_target"] / bin_width).astype(int)

    if scheme == "none":
        df["sample_weight"] = 1.0
        logger.warning("sample_weighting='none' — bin balancing is DISABLED")
        return df
    if scheme != "inverse_bin_freq":
        raise ValueError(f"Unknown sample_weighting scheme: {scheme!r}")

    train = df[df["split"] == "train"]
    if train.empty:
        raise SplitIntegrityError("No train rows — cannot derive sample weights")

    counts = train["logE_bin"].value_counts()
    raw = df["logE_bin"].map(1.0 / counts)
    # Normalise over the train split so mean train weight is exactly 1.0.
    df["sample_weight"] = raw / raw[df["split"] == "train"].mean()
    return df


def build_manifests(
    minc_root: Union[str, Path],
    lookup_csv: Union[str, Path] = "data/material_E_lookup.csv",
    split: int = 1,
    bin_width: float = 0.5,
    sample_weighting: str = "inverse_bin_freq",
    out_dir: Optional[Union[str, Path]] = "outputs/manifests",
    strict_counts: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Build train/val/test manifests from MINC split files joined to the lookup.

    Parameters
    ----------
    strict_counts : bool
        If True, assert the exact expected row counts for MINC-2500 official
        split 1 (29,750 / 1,750 / 3,500). Set False only when running against a
        synthetic or subsetted tree.

    Returns
    -------
    dict
        ``{"train": df, "val": df, "test": df}``, also written as CSVs to
        *out_dir* when given.
    """
    logger = get_logger("dataset")
    minc_root = Path(minc_root)
    if not minc_root.exists():
        raise FileNotFoundError(
            f"MINC root not found: {minc_root.resolve()} — download and extract "
            "minc-2500.tar.gz first."
        )

    lut = included(load_lookup(lookup_csv))
    lut = lut.assign(_key=lut["minc_class"].map(class_key))
    if lut["_key"].duplicated().any():
        raise SplitIntegrityError(
            "Lookup class names collide under class_key(): "
            f"{lut.loc[lut['_key'].duplicated(keep=False), 'minc_class'].tolist()}"
        )
    expected_keys = set(lut["_key"])
    expected_classes = set(lut["minc_class"])

    frames = []
    for name, pattern in SPLIT_FILES.items():
        f = minc_root / "labels" / pattern.format(n=split)
        part = parse_split_file(f, name)
        logger.info("%-5s %s: %d rows", name, f.name, len(part))
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)

    # Integrity is checked BEFORE filtering, so a path duplicated across splits
    # is caught even if it belongs to an excluded class.
    pre_filter_classes = set(df["minc_class"])
    dupes = df[df.duplicated("image_path", keep=False)]
    if not dupes.empty:
        raise SplitIntegrityError(
            f"CHECK 1 FAILED pre-filter: {dupes['image_path'].nunique()} path(s) "
            f"in more than one split, e.g. "
            f"{dupes['image_path'].drop_duplicates().head(5).tolist()}"
        )

    df["_key"] = df["minc_class"].map(class_key)
    unknown = sorted(pre_filter_classes - {c for c in pre_filter_classes
                                           if class_key(c) in expected_keys})
    logger.info(
        "Dropping %d excluded/unknown class(es): %s", len(unknown), unknown
    )
    df = df[df["_key"].isin(expected_keys)].reset_index(drop=True)

    # Join on the separator-stripped key, then carry the lookup's canonical
    # class name so manifests use the published spelling ("polished stone")
    # rather than MINC's directory spelling ("polishedstone").
    df = df.drop(columns=["minc_class"]).merge(
        lut, on="_key", how="left", validate="many_to_one"
    )
    if df["E_median_GPa"].isna().any():
        bad = sorted(df.loc[df["E_median_GPa"].isna(), "minc_class"].unique())
        raise SplitIntegrityError(f"Lookup join failed for classes: {bad}")

    matched = df["minc_class"].value_counts()
    unmatched = sorted(expected_classes - set(matched.index))
    if unmatched:
        raise SplitIntegrityError(
            f"Included classes matched ZERO images (check name normalisation, "
            f"e.g. polished_stone): {unmatched}"
        )
    logger.info("All %d included classes matched images", len(matched))

    df = df.rename(columns={"log10E_median": "log10E_target"}).drop(
        columns=["_key"]
    )
    _check_integrity(df, expected_classes)
    df = _add_weights(df, bin_width=bin_width, scheme=sample_weighting)

    counts = {s: int((df["split"] == s).sum()) for s in SPLIT_FILES}
    logger.info("Row counts: %s (total %d)", counts, len(df))
    if strict_counts and counts != EXPECTED_ROWS:
        raise SplitIntegrityError(
            f"Row counts {counts} != expected {EXPECTED_ROWS} for MINC-2500 "
            "official split 1. Pass strict_counts=False for a subsetted tree."
        )

    train = df[df["split"] == "train"]
    logger.info(
        "sample_weight over train: min=%.3f max=%.3f ratio=%.2fx mean=%.3f "
        "(see limitation 2 — this partly cancels the Phase 5 interval weighting)",
        train["sample_weight"].min(),
        train["sample_weight"].max(),
        train["sample_weight"].max() / train["sample_weight"].min(),
        train["sample_weight"].mean(),
    )
    for s in SPLIT_FILES:
        g = df[df["split"] == s]
        logger.info(
            "%-5s: %5d rows | %2d classes | log10E %.3f..%.3f",
            s,
            len(g),
            g["minc_class"].nunique(),
            g["log10E_target"].min(),
            g["log10E_target"].max(),
        )

    out = {}
    for s in SPLIT_FILES:
        part = df[df["split"] == s][MANIFEST_COLS].reset_index(drop=True)
        out[s] = part
        if out_dir:
            p = Path(out_dir)
            p.mkdir(parents=True, exist_ok=True)
            part.to_csv(p / f"{s}.csv", index=False)
            logger.info("Wrote %s (%d rows)", p / f"{s}.csv", len(part))
    return out


def plot_split_coverage(
    manifests: Dict[str, pd.DataFrame],
    out_path: Union[str, Path] = "outputs/figures/split_logE_coverage.png",
    dpi: int = 150,
) -> Path:
    """
    Save a per-split log10E histogram. Val and test must cover the same range as
    train, not an easier subset of it.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    allv = pd.concat(manifests.values())["log10E_target"]
    bins = np.linspace(allv.min() - 0.25, allv.max() + 0.25, 40)

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for ax, (name, part) in zip(axes, manifests.items()):
        ax.hist(part["log10E_target"], bins=bins, color="#4169b2", edgecolor="white")
        ax.set_ylabel("patches")
        ax.set_title(
            f"{name} — {len(part)} rows, {part['minc_class'].nunique()} classes, "
            f"log10E {part['log10E_target'].min():.2f}.."
            f"{part['log10E_target'].max():.2f}",
            fontsize=9,
        )
    axes[-1].set_xlabel("log$_{10}$E target (GPa)  —  class-level labels")
    fig.suptitle("Per-split stiffness coverage (MINC-2500 split 1)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    get_logger("dataset").info("Wrote coverage plot: %s", out_path)
    return out_path
