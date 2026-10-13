#!/usr/bin/env python3
"""
Phase 2 — Acceptance verification, against a SYNTHETIC MINC tree.

Runs without the 2.1 GB download so the logic is proven before scaling, per the
build spec's "test before scaling" rule. Deliberately includes:
  - ``polished_stone`` as a directory name (the silent-join trap)
  - excluded classes (sky, food) that must be filtered out
  - the real per-class split ratio 2,125 / 125 / 250, scaled down

Then injects each integrity failure and checks it raises.
"""

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vise.config import ensure_utf8_stdio, setup_logging  # noqa: E402

ensure_utf8_stdio()

import pandas as pd  # noqa: E402

from vise.dataset import (  # noqa: E402
    MANIFEST_COLS,
    class_key,
    SplitIntegrityError,
    build_manifests,
    normalise_class,
    parse_split_file,
    plot_split_coverage,
)

LUT = ROOT / "data" / "material_E_lookup.csv"
setup_logging(output_dir=str(ROOT / "outputs"), experiment_name="phase2_test")

# MINC directory names: note the underscore, which the lookup spells with a space.
DIRS = [
    "brick", "carpet", "ceramic", "fabric", "glass", "leather", "metal",
    "mirror", "paper", "plastic", "polishedstone", "stone", "tile", "wood",
]
EXCLUDED_DIRS = ["sky", "food", "skin"]
PER_SPLIT = {"train": 17, "val": 1, "test": 2}  # 2125/125/250 scaled by 1/125


def make_tree(root: Path, dirs=DIRS, excluded=EXCLUDED_DIRS, n=PER_SPLIT):
    """Write a synthetic MINC-2500 layout: labels/*.txt and images/<class>/."""
    (root / "labels").mkdir(parents=True)
    lines = {s: [] for s in n}
    for cls in [*dirs, *excluded]:
        (root / "images" / cls).mkdir(parents=True)
        i = 0
        for split, count in n.items():
            for _ in range(count):
                p = f"images/{cls}/{cls}_{i:05d}.jpg"
                (root / p).touch()
                lines[split].append(p)
                i += 1
    for split, fname in (("train", "train1.txt"), ("val", "validate1.txt"),
                         ("test", "test1.txt")):
        (root / "labels" / fname).write_text(
            "\n".join(lines[split]) + "\n", encoding="utf-8"
        )
    return lines


tmp = Path(tempfile.mkdtemp(prefix="vise_phase2_"))
try:
    root = tmp / "minc-2500"
    make_tree(root)

    # ── Test 1: normalisation ──────────────────────────────────────────
    print("=" * 60)
    print("TEST 1 — Class-name normalisation (the polished_stone trap)")
    print("=" * 60)
    assert normalise_class("polished_stone") == "polished stone"
    # MINC spells it as one word, so the JOIN KEY must strip separators, not
    # merely convert them to spaces. Verified against minc-2500/categories.txt.
    for variant in ("polishedstone", "polished_stone", "polished stone",
                    "PolishedStone"):
        assert class_key(variant) == "polishedstone", variant
    assert class_key("wood") == "wood"
    keys = [class_key(c) for c in [*DIRS, *EXCLUDED_DIRS]]
    assert len(keys) == len(set(keys)), "class_key collapsed two real classes"
    print("  ✓ class_key(): polishedstone / polished_stone / 'polished stone'"
          " all -> 'polishedstone'")
    print("  ✓ No two MINC class names collide under class_key()")

    # ── Test 2: split parsing ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 2 — Split file parsing")
    print("=" * 60)
    t = parse_split_file(root / "labels" / "train1.txt", "train")
    expected_train = (len(DIRS) + len(EXCLUDED_DIRS)) * PER_SPLIT["train"]
    assert len(t) == expected_train, f"{len(t)} != {expected_train}"
    assert set(t.columns) == {"image_path", "minc_class", "split"}
    # parse_split_file reports MINC's own spelling; the published name is
    # substituted later, by the lookup join.
    assert "polishedstone" in set(t["minc_class"])
    print(f"  ✓ Parsed {len(t)} rows; class taken from 2nd path segment")

    bad = tmp / "bad.txt"
    bad.write_text("images/wood/a.jpg\nnot_a_valid_path\n", encoding="utf-8")
    try:
        parse_split_file(bad, "train")
    except SplitIntegrityError as e:
        print(f"  ✓ Malformed line rejected: {str(e).splitlines()[1].strip()}")
    else:
        raise AssertionError("malformed line accepted")

    # ── Test 3: full build ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 3 — Build manifests (synthetic, strict_counts off)")
    print("=" * 60)
    out = tmp / "manifests"
    m = build_manifests(
        root, lookup_csv=LUT, out_dir=out, strict_counts=False
    )

    for s, n in PER_SPLIT.items():
        assert len(m[s]) == len(DIRS) * n, f"{s}: {len(m[s])}"
        assert m[s]["minc_class"].nunique() == 14
    print(
        f"  ✓ Rows: train {len(m['train'])} / val {len(m['val'])} / "
        f"test {len(m['test'])}, 14 classes each"
    )
    assert not set(m["train"]["minc_class"]) & set(EXCLUDED_DIRS)
    print("  ✓ Excluded classes (sky, food, skin) filtered out")

    for s in m:
        assert list(m[s].columns) == MANIFEST_COLS, f"{s} column mismatch"
        assert m[s].notna().all().all(), f"{s} has NaN"
    print(f"  ✓ All {len(MANIFEST_COLS)} required columns present, no NaN")

    ps = m["train"][m["train"]["minc_class"] == "polished stone"]
    assert len(ps) == PER_SPLIT["train"] and ps["E_median_GPa"].eq(60.0).all()
    print(
        f"  ✓ MINC 'polishedstone' dir joined to lookup 'polished stone': "
        f"{len(ps)} rows, E_median=60.0 GPa"
    )
    assert "polishedstone" not in set(m["train"]["minc_class"]), (
        "manifest should carry the published spelling, not MINC's directory name"
    )
    assert "_key" not in m["train"].columns, "internal join key leaked"

    for s in m:
        assert (out / f"{s}.csv").exists()
        assert len(pd.read_csv(out / f"{s}.csv")) == len(m[s])
    print("  ✓ CSVs written and round-trip to the same row counts")

    tr = m["train"]
    print(
        f"  ✓ sample_weight: min={tr['sample_weight'].min():.3f} "
        f"max={tr['sample_weight'].max():.3f} "
        f"ratio={tr['sample_weight'].max() / tr['sample_weight'].min():.2f}x "
        f"mean={tr['sample_weight'].mean():.3f}"
    )
    assert abs(tr["sample_weight"].mean() - 1.0) < 1e-9, "train weights not mean-1"
    # The 5-class +2.0 bin must get the lowest weight, leather/wood the highest.
    w = tr.groupby("minc_class")["sample_weight"].first()
    assert w["leather"] > w["glass"], "bin balancing not upweighting leather"
    assert abs(w["glass"] - w["metal"]) < 1e-9, "same bin, different weight"
    print(
        f"  ✓ Bin balancing works: leather={w['leather']:.3f} > "
        f"glass={w['glass']:.3f} (ratio {w['leather'] / w['glass']:.2f}x)"
    )

    # ── Test 4: integrity checks fire ──────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 4 — Integrity checks must raise")
    print("=" * 60)

    def must_raise(label, build_root):
        try:
            build_manifests(
                build_root, lookup_csv=LUT, out_dir=None, strict_counts=False
            )
        except SplitIntegrityError as e:
            lines = [ln.strip() for ln in str(e).splitlines() if ln.strip()]
            first = next(
                (ln for ln in lines if ln.startswith("-")), lines[0]
            ).lstrip("- ")
            print(f"  ✓ {label}\n      {first}")
        else:
            raise AssertionError(f"Expected SplitIntegrityError for: {label}")

    # CHECK 1 — a path in two splits.
    r1 = tmp / "dup"
    shutil.copytree(root, r1)
    f = r1 / "labels" / "test1.txt"
    first_train = (r1 / "labels" / "train1.txt").read_text(
        encoding="utf-8"
    ).splitlines()[0]
    f.write_text(f.read_text(encoding="utf-8") + first_train + "\n", encoding="utf-8")
    must_raise("CHECK 1 — image_path in two splits", r1)

    # CHECK 2 — a class missing from the test split.
    r2 = tmp / "missing"
    shutil.copytree(root, r2)
    f = r2 / "labels" / "test1.txt"
    f.write_text(
        "\n".join(
            ln for ln in f.read_text(encoding="utf-8").splitlines()
            if "/wood/" not in ln
        ) + "\n",
        encoding="utf-8",
    )
    must_raise("CHECK 2 — class absent from a split", r2)

    # The join trap: a class present on disk but unmatched by the lookup.
    r3 = tmp / "joinfail"
    shutil.copytree(root, r3)
    for fn in ("train1.txt", "validate1.txt", "test1.txt"):
        f = r3 / "labels" / fn
        f.write_text(
            f.read_text(encoding="utf-8").replace(
                "images/polishedstone/", "images/unobtainium/"
            ),
            encoding="utf-8",
        )
    must_raise("Lookup join — included class matched zero images", r3)

    # strict_counts must reject a tree that is not real MINC split 1.
    try:
        build_manifests(root, lookup_csv=LUT, out_dir=None, strict_counts=True)
    except SplitIntegrityError as e:
        assert "29750" in str(e).replace(",", "")
        print("  ✓ strict_counts rejects a non-MINC tree")
        print(f"      {str(e).splitlines()[0].strip()[:96]}")
    else:
        raise AssertionError("strict_counts accepted a synthetic tree")

    # ── Test 5: coverage plot ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 5 — Per-split coverage plot")
    print("=" * 60)
    p = plot_split_coverage(m, out_path=tmp / "cov.png")
    assert p.exists() and p.stat().st_size > 5000
    print(f"  ✓ Plot written: {p.stat().st_size // 1024} KB at 150 dpi")

    rng = {s: (m[s]["log10E_target"].min(), m[s]["log10E_target"].max()) for s in m}
    assert rng["val"] == rng["train"] == rng["test"], rng
    print(f"  ✓ val/test cover the same log-E range as train: {rng['train']}")

    print("\n" + "=" * 60)
    print("ALL PHASE 2 LOGIC VERIFIED ON SYNTHETIC DATA ✓")
    print("Run scripts/01_build_manifest.py on the real tree for Checkpoint 2.")
    print("=" * 60)
finally:
    shutil.rmtree(tmp, ignore_errors=True)
