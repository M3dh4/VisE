#!/usr/bin/env python3
"""
Phase 0 — Acceptance verification.

Tests:
  1. `import vise` succeeds
  2. Config loads and prints resolved values (including inheritance)
  3. Seeding: two runs of a random op give identical results
  4. Device detection works and is logged
  5. Regression guards for the Phase 0 bug fixes:
     - UTF-8 console output (cp1252 crash)
     - per-experiment log files (Phase 8 ablation)
     - DataLoader worker reproducibility helpers
     - dotted-override error handling
"""

import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# UTF-8 before the first print: this file prints maths notation, and the
# Windows default console encoding (cp1252) cannot represent it.
from vise.config import ensure_utf8_stdio  # noqa: E402

ensure_utf8_stdio()

# ── Test 1: import vise ─────────────────────────────────────────────────
print("=" * 60)
print("TEST 1 — import vise")
print("=" * 60)
import vise  # noqa: E402
print(f"  ✓ vise imported successfully, version={vise.__version__}")
print("  ✓ UTF-8 console output works (✓ — λ log₁₀E ×)")

# ── Test 2: config loading with inheritance ─────────────────────────────
print("\n" + "=" * 60)
print("TEST 2 — Config loading + inheritance")
print("=" * 60)

from vise.config import load_config, pretty_config, setup_logging  # noqa: E402

# 2a. Load base config
base_cfg = load_config(ROOT / "configs" / "base.yaml")
print("\n--- base.yaml (resolved) ---")
print(pretty_config(base_cfg))

# 2b. Load a child config that inherits from base
joint_cfg = load_config(ROOT / "configs" / "model_c_joint.yaml")
print("\n--- model_c_joint.yaml (resolved, inherits base) ---")
print(pretty_config(joint_cfg))
assert joint_cfg["experiment"]["name"] == "model_c_joint", "Inheritance override failed"
assert joint_cfg["training"]["batch_size"] == 16, "Base value not inherited"
print("  ✓ Inheritance and override work correctly")

# 2c. Load depth-only config and verify head flags
depth_cfg = load_config(ROOT / "configs" / "model_a_depth.yaml")
assert depth_cfg["model"]["use_depth_head"] is True
assert depth_cfg["model"]["use_stiffness_head"] is False
assert depth_cfg["losses"]["lambda_stiffness"] == 0.0
print("  ✓ Model A (depth-only) correctly disables stiffness head/loss")

# 2d. Load stiffness-only config and verify
stiff_cfg = load_config(ROOT / "configs" / "model_b_stiff.yaml")
assert stiff_cfg["model"]["use_depth_head"] is False
assert stiff_cfg["model"]["use_stiffness_head"] is True
assert stiff_cfg["losses"]["lambda_depth"] == 0.0
print("  ✓ Model B (stiffness-only) correctly disables depth head/loss")

# 2e. Test override mechanism
override_cfg = load_config(
    ROOT / "configs" / "base.yaml",
    overrides={"training.batch_size": 8, "experiment.name": "test_override"},
)
assert override_cfg["training"]["batch_size"] == 8
assert override_cfg["experiment"]["name"] == "test_override"
print("  ✓ Programmatic overrides work")

# 2f. A dotted override through a scalar must fail with a named path, not an
#     opaque setdefault error.
try:
    load_config(
        ROOT / "configs" / "base.yaml",
        overrides={"training.batch_size.nope": 1},
    )
except TypeError as exc:
    assert "training.batch_size" in str(exc), f"Unhelpful error: {exc}"
    print(f"  ✓ Bad dotted override rejected: {exc}")
else:
    raise AssertionError("Expected TypeError for override through a scalar")

# 2g. A new config section can still be created by override.
new_section = load_config(
    ROOT / "configs" / "base.yaml", overrides={"brand.new.key": 42}
)
assert new_section["brand"]["new"]["key"] == 42
print("  ✓ Overrides can create new nested sections")

# ── Test 3: seeding reproducibility ────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 3 — Seeding reproducibility")
print("=" * 60)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from vise.config import make_generator, seed_everything, seed_worker  # noqa: E402

logger = setup_logging(output_dir=str(ROOT / "outputs"), experiment_name="phase0_test")

# Run 1
seed_everything(1729)
r1_np = np.random.rand(5)
r1_torch = torch.randn(5)

# Run 2
seed_everything(1729)
r2_np = np.random.rand(5)
r2_torch = torch.randn(5)

assert np.array_equal(r1_np, r2_np), "NumPy reproducibility FAILED"
assert torch.equal(r1_torch, r2_torch), "PyTorch reproducibility FAILED"

print(f"  NumPy  run 1: {r1_np}")
print(f"  NumPy  run 2: {r2_np}")
print(f"  Torch  run 1: {r1_torch.tolist()}")
print(f"  Torch  run 2: {r2_torch.tolist()}")
print("  ✓ Both runs identical — reproducibility confirmed")

# 3b. DataLoader determinism helpers exist and behave.
g1 = make_generator(1729)
g2 = make_generator(1729)
assert torch.equal(
    torch.randperm(10, generator=g1), torch.randperm(10, generator=g2)
), "make_generator is not deterministic"
print("  ✓ make_generator() gives reproducible sampler/shuffle order")

seed_worker(0)  # must not raise; reseeds NumPy/random inside a worker
print("  ✓ seed_worker() callable as a DataLoader worker_init_fn")

# 3c. A real DataLoader wired with both helpers must repeat exactly.
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

ds = TensorDataset(torch.arange(32).float())


def _drain(seed: int):
    loader = DataLoader(
        ds,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=make_generator(seed),
    )
    return torch.cat([b[0] for b in loader])


assert torch.equal(_drain(1729), _drain(1729)), "DataLoader order not reproducible"
assert not torch.equal(_drain(1729), _drain(7)), "Different seeds gave same order"
print("  ✓ DataLoader order reproducible across runs, and varies with seed")

# ── Test 4: device detection ───────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 4 — Device detection")
print("=" * 60)

from vise.config import detect_device  # noqa: E402

device = detect_device()
print(f"  ✓ Detected device: {device}")

# The CUDA branch reads torch.cuda.get_device_properties(...).total_memory.
# That attribute name is only exercised on a GPU machine, so assert it exists
# rather than discovering the typo mid-training on Colab.
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    assert hasattr(props, "total_memory"), "CUDA property name wrong"
    print(f"  ✓ CUDA memory read: {props.total_memory / (1024 ** 3):.1f} GB")
else:
    print("  ! CPU-only build — CUDA branch of detect_device() not exercised here")

# ── Test 5: per-experiment log files ───────────────────────────────────
print("\n" + "=" * 60)
print("TEST 5 — Per-experiment log files (Phase 8 ablation)")
print("=" * 60)

import logging  # noqa: E402

log_dir = ROOT / "outputs" / "logs"
before = set(log_dir.glob("*.log"))

# Same name -> no new file.
setup_logging(output_dir=str(ROOT / "outputs"), experiment_name="phase0_test")
assert set(log_dir.glob("*.log")) == before, "Same experiment name opened a new log"
print("  ✓ Re-init with the same experiment name is a no-op")

# Different names -> one new file each, and only one file handler at a time.
for name in ("ablation_seed0", "ablation_seed1"):
    setup_logging(output_dir=str(ROOT / "outputs"), experiment_name=name)
    logging.getLogger("vise").info("marker for %s", name)

vise_logger = logging.getLogger("vise")
file_handlers = [
    h for h in vise_logger.handlers if isinstance(h, logging.FileHandler)
]
stream_handlers = [
    h
    for h in vise_logger.handlers
    if isinstance(h, logging.StreamHandler)
    and not isinstance(h, logging.FileHandler)
]
assert len(file_handlers) == 1, f"Expected 1 file handler, got {len(file_handlers)}"
assert len(stream_handlers) == 1, "Console handler duplicated across re-inits"

after = set(log_dir.glob("*.log"))
new_files = after - before
assert len(new_files) == 2, f"Expected 2 new log files, got {len(new_files)}"

seed1_log = next(p for p in new_files if p.name.startswith("ablation_seed1"))
seed0_log = next(p for p in new_files if p.name.startswith("ablation_seed0"))
assert "marker for ablation_seed1" in seed1_log.read_text(encoding="utf-8")
assert "marker for ablation_seed1" not in seed0_log.read_text(encoding="utf-8")
print("  ✓ Each experiment gets its own log file; handlers not duplicated")
print("  ✓ Log lines routed to the correct experiment's file")

# ── Summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ALL PHASE 0 ACCEPTANCE CRITERIA PASSED ✓")
print("=" * 60)
