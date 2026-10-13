"""
Load and validate the material stiffness lookup table. (Phase 1)

``data/material_E_lookup.csv`` is the scientific artifact of this project: one
row per MINC-2500 class, each carrying a published Young's modulus interval and
a provenance code (ENG = engineering property table, TEX = textile/cellular-solid
mechanics literature, EXCL = excluded). It is the single source of truth and is
never duplicated inline in code.

Derived columns (computed here, not stored): ``log10E_min``, ``log10E_max``,
``log10E_median``, ``log10E_halfwidth`` = (log10E_max - log10E_min) / 2.

Limitations this module makes explicit rather than hiding
---------------------------------------------------------
1. **14 classes are only 11 distinct regression targets.** ``ceramic``, ``glass``
   and ``mirror`` all have E_median = 70 GPa; ``stone`` and ``tile`` both 50 GPa.
   7,500 patches therefore carry one identical target. Physically reasonable, but
   "14 material classes" overstates the label resolution. ``summarise`` reports
   both counts and :func:`validate` asserts the distinct count so it cannot drift.

2. **``E_median`` is not the geometric midpoint of the interval.** Published
   medians sit off-centre in log space by up to +0.35 decades (plastic). The
   regression target therefore does not aim at the interval centre, which mildly
   inflates interval-hit rate and makes the Phase 7 "ceiling = mean half-width"
   line approximate. Published typical values are kept deliberately; the offset
   is reported rather than silently re-centred.

3. **Log-E span is measured over interval extremes, not medians.** Extremes give
   ~5.62 decades (the figure quoted in the report); medians give only ~4.74.

4. Excluded rows carry empty numeric cells, not placeholder text, so the numeric
   columns parse as float with NaN rather than as ``object`` dtype.
"""

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from vise.config import get_logger

DEFAULT_CSV = Path("data/material_E_lookup.csv")

N_ROWS_EXPECTED = 23
N_INCLUDED_EXPECTED = 14
N_DISTINCT_TARGETS_EXPECTED = 11  # see limitation 1
NUMERIC_COLS = ["E_min_GPa", "E_max_GPa", "E_median_GPa"]
REQUIRED_COLS = ["minc_class", "include", *NUMERIC_COLS, "provenance", "note"]


class LookupValidationError(ValueError):
    """Raised when the lookup table fails validation. Never silently repaired."""


def load_lookup(
    csv_path: Union[str, Path] = DEFAULT_CSV, validate_table: bool = True
) -> pd.DataFrame:
    """
    Load the lookup CSV, attach derived log-space columns, and validate.

    Raises
    ------
    FileNotFoundError
        If the CSV is missing.
    LookupValidationError
        If the table is malformed. Validation never repairs — it raises.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Lookup table not found: {csv_path.resolve()}")

    df = pd.read_csv(csv_path, dtype={"minc_class": str, "provenance": str})

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise LookupValidationError(f"Lookup CSV missing columns: {missing}")

    # Excluded rows must have *empty* numeric cells, which parse as NaN. A
    # non-numeric placeholder (the spec's table renders these as an em-dash)
    # would make the column object dtype and break log10 downstream. Coercing
    # it to NaN would be a silent repair, which this module does not do: detect
    # the offending cells and raise instead.
    placeholders = []
    for col in NUMERIC_COLS:
        coerced = pd.to_numeric(df[col], errors="coerce")
        bad = coerced.isna() & df[col].notna() & (
            df[col].astype(str).str.strip() != ""
        )
        for cls, cell in zip(df.loc[bad, "minc_class"], df.loc[bad, col]):
            placeholders.append(f"{cls}.{col} = {cell!r}")
        df[col] = coerced
    if placeholders:
        raise LookupValidationError(
            "Lookup table validation failed:\n  - Non-numeric placeholder in "
            "numeric column (use an empty cell, not a dash):\n      "
            + "\n      ".join(placeholders)
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        df["log10E_min"] = np.log10(df["E_min_GPa"])
        df["log10E_max"] = np.log10(df["E_max_GPa"])
        df["log10E_median"] = np.log10(df["E_median_GPa"])
    df["log10E_halfwidth"] = (df["log10E_max"] - df["log10E_min"]) / 2.0

    if validate_table:
        validate(df)
    return df


def validate(df: pd.DataFrame) -> None:
    """
    Validate the lookup table. Raises :class:`LookupValidationError` listing
    every problem found, rather than repairing or warning.
    """
    errors = []

    if len(df) != N_ROWS_EXPECTED:
        errors.append(f"Expected {N_ROWS_EXPECTED} rows, found {len(df)}")

    if df["minc_class"].duplicated().any():
        dupes = df.loc[df["minc_class"].duplicated(), "minc_class"].tolist()
        errors.append(f"Duplicate minc_class values: {dupes}")

    if not df["include"].isin([0, 1]).all():
        bad = df.loc[~df["include"].isin([0, 1]), "minc_class"].tolist()
        errors.append(f"include must be 0 or 1; offending classes: {bad}")

    inc = df[df["include"] == 1]
    if len(inc) != N_INCLUDED_EXPECTED:
        errors.append(
            f"Expected {N_INCLUDED_EXPECTED} included rows, found {len(inc)}"
        )

    # Every row needs a provenance code.
    blank_prov = df[df["provenance"].isna() | (df["provenance"].str.strip() == "")]
    if not blank_prov.empty:
        errors.append(
            f"Empty provenance for: {blank_prov['minc_class'].tolist()}"
        )

    # No included row may have a missing numeric field.
    for col in NUMERIC_COLS:
        nan_rows = inc[inc[col].isna()]["minc_class"].tolist()
        if nan_rows:
            errors.append(f"Included rows with missing {col}: {nan_rows}")

    # Positivity and strict ordering E_min < E_median < E_max.
    usable = inc.dropna(subset=NUMERIC_COLS)
    nonpos = usable[(usable[NUMERIC_COLS] <= 0).any(axis=1)]["minc_class"].tolist()
    if nonpos:
        errors.append(f"Non-positive modulus values for: {nonpos}")

    bad_order = usable[
        ~(
            (usable["E_min_GPa"] < usable["E_median_GPa"])
            & (usable["E_median_GPa"] < usable["E_max_GPa"])
        )
    ]["minc_class"].tolist()
    if bad_order:
        errors.append(
            f"Violates E_min < E_median < E_max for: {bad_order}"
        )

    # Excluded rows must not carry modulus values.
    exc = df[df["include"] == 0]
    leaked = exc[exc[NUMERIC_COLS].notna().any(axis=1)]["minc_class"].tolist()
    if leaked:
        errors.append(f"Excluded rows carry modulus values: {leaked}")

    # Guard the 14-classes/11-targets collapse (limitation 1). Only possible
    # once the derived columns exist; validate() is public and may be handed a
    # raw frame, which is not an error in itself.
    if "log10E_median" not in usable.columns:
        get_logger("lookup").warning(
            "validate() called without derived columns — skipping the "
            "distinct-target check. Use load_lookup() for full validation."
        )
    elif len(usable) == N_INCLUDED_EXPECTED:
        n_distinct = usable["log10E_median"].round(6).nunique()
        if n_distinct != N_DISTINCT_TARGETS_EXPECTED:
            errors.append(
                f"Expected {N_DISTINCT_TARGETS_EXPECTED} distinct log10E_median "
                f"targets across {N_INCLUDED_EXPECTED} classes, found {n_distinct}. "
                "If this is intentional, update N_DISTINCT_TARGETS_EXPECTED and "
                "the label-resolution statement in the report."
            )

    if errors:
        raise LookupValidationError(
            "Lookup table validation failed:\n  - " + "\n  - ".join(errors)
        )


def included(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the supervised (``include == 1``) rows."""
    return df[df["include"] == 1].reset_index(drop=True)


def summarise(df: pd.DataFrame) -> str:
    """Return a printable summary of the table, including its known limitations."""
    inc = included(df)
    n_exc = int((df["include"] == 0).sum())
    span_ext = inc["log10E_max"].max() - inc["log10E_min"].min()
    span_med = inc["log10E_median"].max() - inc["log10E_median"].min()
    mean_hw = inc["log10E_halfwidth"].mean()
    n_distinct = inc["log10E_median"].round(6).nunique()

    midpoint = (inc["log10E_max"] + inc["log10E_min"]) / 2.0
    offset = (inc["log10E_median"] - midpoint).abs()
    worst = inc.loc[offset.idxmax(), "minc_class"]

    lines = [
        f"{len(inc)} included / {n_exc} excluded  (total {len(df)})",
        f"log-E span (interval extremes) : {span_ext:.3f} decades",
        f"log-E span (medians only)      : {span_med:.3f} decades",
        f"mean interval half-width       : {mean_hw:.4f} decades  "
        f"(Phase 7 accuracy ceiling)",
        f"distinct regression targets    : {n_distinct} of {len(inc)} classes",
        f"largest median/midpoint offset : {offset.max():.4f} decades ({worst})",
        "",
        f"{'class':16}{'E_min':>10}{'E_max':>10}{'E_med':>10}"
        f"{'log10E_med':>12}{'halfwidth':>11}{'prov':>6}",
    ]
    for _, r in inc.sort_values("log10E_median").iterrows():
        lines.append(
            f"{r['minc_class']:16}{r['E_min_GPa']:10.4g}{r['E_max_GPa']:10.4g}"
            f"{r['E_median_GPa']:10.4g}{r['log10E_median']:12.4f}"
            f"{r['log10E_halfwidth']:11.4f}{r['provenance']:>6}"
        )

    dupes = (
        inc.groupby(inc["log10E_median"].round(6))["minc_class"]
        .apply(list)
        .loc[lambda s: s.map(len) > 1]
    )
    if len(dupes):
        lines.append("")
        lines.append("Classes sharing an identical target (label resolution limit):")
        for val, members in dupes.items():
            lines.append(f"  log10E = {val:.4f}  <-  {members}")

    return "\n".join(lines)


def log_summary(df: pd.DataFrame) -> None:
    """Log the summary line-by-line through the ``vise`` logger."""
    logger = get_logger("lookup")
    for line in summarise(df).splitlines():
        logger.info("%s", line)
