#!/usr/bin/env python3
"""
Phase 1 — Acceptance verification for the stiffness lookup table.

  1. Validation passes on the correct file
  2. Validation fails loudly on corrupted rows (each guard exercised)
  3. log10E_halfwidth spot-checks (glass ~0.13, leather ~0.85)
  4. Printed summary: 14 included / 9 excluded, log-E span ~5.6 decades
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vise.config import ensure_utf8_stdio, setup_logging  # noqa: E402

ensure_utf8_stdio()

from vise.lookup import (  # noqa: E402
    LookupValidationError,
    included,
    load_lookup,
    log_summary,
    summarise,
    validate,
)

CSV = ROOT / "data" / "material_E_lookup.csv"
setup_logging(output_dir=str(ROOT / "outputs"), experiment_name="phase1_lookup")

# ── Test 1: loads and validates ────────────────────────────────────────
print("=" * 60)
print("TEST 1 — Load + validate")
print("=" * 60)
df = load_lookup(CSV)
inc = included(df)
assert len(df) == 23 and len(inc) == 14
print(f"  ✓ Validation passed: {len(df)} rows, {len(inc)} included")

# Excluded rows must parse as NaN floats, not strings (the em-dash trap).
import numpy as np  # noqa: E402

for col in ("E_min_GPa", "E_max_GPa", "E_median_GPa"):
    assert np.issubdtype(df[col].dtype, np.floating), f"{col} is not float dtype"
assert df[df["include"] == 0][["E_min_GPa"]].isna().all().item()
print("  ✓ Excluded rows are NaN floats — numeric columns parse cleanly")

# ── Test 2: validation fails loudly ────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 2 — Corruption must raise, not repair")
print("=" * 60)


def must_raise(label, mutate):
    bad = df.copy()
    mutate(bad)
    try:
        validate(bad)
    except LookupValidationError as exc:
        first = str(exc).splitlines()[1].strip()
        print(f"  ✓ {label}\n      {first}")
    else:
        raise AssertionError(f"Expected LookupValidationError for: {label}")


must_raise("dropped row (23 -> 22)", lambda d: d.drop(index=0, inplace=True))
must_raise(
    "wrong included count",
    lambda d: d.__setitem__("include", d["include"].where(d.index != 0, 0)),
)
must_raise(
    "E_median above E_max (ordering)",
    lambda d: d.__setitem__("E_median_GPa", d["E_median_GPa"].where(d.index != 0, 99.0)),
)
must_raise(
    "negative modulus",
    lambda d: d.__setitem__("E_min_GPa", d["E_min_GPa"].where(d.index != 0, -1.0)),
)
must_raise(
    "missing numeric on an included row",
    lambda d: d.__setitem__("E_max_GPa", d["E_max_GPa"].where(d.index != 0, np.nan)),
)
must_raise(
    "blank provenance",
    lambda d: d.__setitem__("provenance", d["provenance"].where(d.index != 0, "")),
)
must_raise(
    "excluded row carrying a modulus",
    lambda d: d.__setitem__("E_min_GPa", d["E_min_GPa"].where(d["minc_class"] != "sky", 5.0)),
)
must_raise(
    "duplicate class name",
    lambda d: d.__setitem__("minc_class", d["minc_class"].where(d.index != 0, "wood")),
)
must_raise(
    "label-resolution drift (11 distinct targets)",
    lambda d: d.__setitem__(
        "log10E_median", d["log10E_median"].where(d["minc_class"] != "mirror", 1.5)
    ),
)

# A real corrupted file on disk, not just an in-memory frame.
tmp = ROOT / "outputs" / "_corrupt_lookup.csv"
raw = CSV.read_text(encoding="utf-8").replace(
    "glass,1,50.0,90.0,70.0", "glass,1,90.0,50.0,70.0"
)
tmp.write_text(raw, encoding="utf-8")
try:
    load_lookup(tmp)
except LookupValidationError:
    print("  ✓ Corrupted file on disk rejected by load_lookup()")
else:
    raise AssertionError("Corrupted file was accepted")
finally:
    tmp.unlink(missing_ok=True)

# The em-dash placeholder trap: must raise, not be coerced to NaN. Checked on
# both an excluded and an included row.
for label, (old, new) in (
    ("excluded row", ("foliage,0,,,,", "foliage,0,—,—,—,")),
    ("included row", ("wood,1,5.0,15.0,10.0", "wood,1,—,15.0,10.0")),
):
    tmp.write_text(
        CSV.read_text(encoding="utf-8").replace(old, new), encoding="utf-8"
    )
    try:
        load_lookup(tmp)
    except LookupValidationError as exc:
        print(f"  ✓ em-dash placeholder in {label} rejected")
        assert "empty cell" in str(exc)
    else:
        raise AssertionError(f"em-dash in {label} was silently coerced")
    finally:
        tmp.unlink(missing_ok=True)

# validate() on a raw frame (no derived columns) must warn, not KeyError.
import pandas as pd  # noqa: E402

validate(pd.read_csv(CSV))
print("  ✓ validate() on a raw frame warns instead of crashing")

# ── Test 3: derived-column spot checks ─────────────────────────────────
print("\n" + "=" * 60)
print("TEST 3 — Derived columns")
print("=" * 60)
hw = inc.set_index("minc_class")["log10E_halfwidth"]
for cls, expected in (("glass", 0.1276), ("leather", 0.8495), ("mirror", 0.1276)):
    assert abs(hw[cls] - expected) < 1e-3, f"{cls}: {hw[cls]:.4f} != {expected}"
    print(f"  ✓ {cls:8} halfwidth = {hw[cls]:.4f}  (expected ≈ {expected})")

span = inc["log10E_max"].max() - inc["log10E_min"].min()
assert abs(span - 5.623) < 1e-2, f"span {span:.3f}"
mean_hw = inc["log10E_halfwidth"].mean()
assert abs(mean_hw - 0.3692) < 1e-3, f"mean halfwidth {mean_hw:.4f}"
assert inc["log10E_median"].round(6).nunique() == 11
print(f"  ✓ log-E span = {span:.3f} decades (≈ 5.6)")
print(f"  ✓ mean halfwidth = {mean_hw:.4f} decades (Phase 7 ceiling)")
print("  ✓ 11 distinct targets across 14 classes")

# ── Test 4: summary ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 4 — Summary")
print("=" * 60)
print(summarise(df))
log_summary(df)

print("\n" + "=" * 60)
print("ALL PHASE 1 ACCEPTANCE CRITERIA PASSED ✓")
print("=" * 60)
