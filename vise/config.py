"""
VisE — Configuration loading with YAML inheritance, reproducibility, and logging.

Config files support a `_base_` key that points to a parent YAML. Child values
override parent values with recursive dict merge. The resolved config is a plain
nested dict — no OmegaConf dependency.

Reproducibility note
--------------------
`seed_everything` seeds the main process only. DataLoader worker processes draw
their own RNG state, so any DataLoader built in Phase 6 MUST be passed
``worker_init_fn=seed_worker`` and ``generator=make_generator(seed)`` or the
3-seed ablation in Phase 8 will not be bit-reproducible.
"""

import copy
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import yaml


# ── Console encoding ────────────────────────────────────────────────────
#
# This project logs physical units and maths notation (log10, lambda, em-dash,
# tick marks). On Windows the default console encoding is cp1252, which cannot
# represent any of those: every such log line raises UnicodeEncodeError. Force
# the standard streams to UTF-8 before any handler is attached.

def ensure_utf8_stdio() -> None:
    """
    Reconfigure ``sys.stdout``/``sys.stderr`` to UTF-8 with replacement.

    Idempotent and safe to call on streams that are already UTF-8 or that do
    not support reconfiguration (e.g. when captured by a test harness).
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # Stream already detached or not reconfigurable — not fatal.
            pass


ensure_utf8_stdio()


# ── Logging ─────────────────────────────────────────────────────────────

_CONSOLE_HANDLER: Optional[logging.Handler] = None
_FILE_HANDLER: Optional[logging.Handler] = None
_CURRENT_EXPERIMENT: Optional[str] = None


def setup_logging(
    output_dir: Union[str, Path] = "outputs",
    experiment_name: str = "vise",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Initialise structured logging to both console and a timestamped file under
    ``output_dir/logs/``.

    Re-entrant by design: the console handler is attached once, but calling
    this with a *different* ``experiment_name`` closes the previous log file
    and opens a new one. This matters for the Phase 8 ablation, which runs nine
    training jobs (3 models x 3 seeds); without this, every run after the first
    would append to the first run's log file.

    Calling it again with the same ``experiment_name`` is a no-op.
    """
    global _CONSOLE_HANDLER, _FILE_HANDLER, _CURRENT_EXPERIMENT

    ensure_utf8_stdio()
    logger = logging.getLogger("vise")
    logger.setLevel(level)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — attached exactly once for the process lifetime.
    if _CONSOLE_HANDLER is None:
        _CONSOLE_HANDLER = logging.StreamHandler(sys.stdout)
        _CONSOLE_HANDLER.setFormatter(fmt)
        logger.addHandler(_CONSOLE_HANDLER)
    _CONSOLE_HANDLER.setLevel(level)

    # File handler — one per experiment.
    if _CURRENT_EXPERIMENT == experiment_name and _FILE_HANDLER is not None:
        return logger

    if _FILE_HANDLER is not None:
        logger.info("Closing log for experiment '%s'", _CURRENT_EXPERIMENT)
        logger.removeHandler(_FILE_HANDLER)
        _FILE_HANDLER.close()
        _FILE_HANDLER = None

    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _FILE_HANDLER = logging.FileHandler(
        log_dir / f"{experiment_name}_{ts}.log", encoding="utf-8"
    )
    _FILE_HANDLER.setLevel(level)
    _FILE_HANDLER.setFormatter(fmt)
    logger.addHandler(_FILE_HANDLER)
    _CURRENT_EXPERIMENT = experiment_name

    logger.info(
        "Logging initialised for '%s' — console + %s",
        experiment_name,
        _FILE_HANDLER.baseFilename,
    )
    return logger


def get_logger(name: str = "vise") -> logging.Logger:
    """Return a child logger under the ``vise`` hierarchy."""
    return logging.getLogger(f"vise.{name}")


# ── Seeding ─────────────────────────────────────────────────────────────

def seed_everything(seed: int = 1729, warn_hashseed: bool = True) -> None:
    """
    Set seeds for Python, NumPy, and PyTorch (CPU + CUDA) for reproducibility.

    Also enables deterministic CuDNN (may reduce speed slightly).

    Limitation: ``PYTHONHASHSEED`` governs str/bytes hash randomisation and is
    read by CPython at interpreter start-up, so setting it here affects only
    *child* processes, not this one. Nothing in VisE depends on dict/set
    ordering of strings, so this is recorded rather than worked around; a
    warning is logged so the gap stays visible rather than being assumed away.
    """
    logger = get_logger("seed")

    if warn_hashseed and "PYTHONHASHSEED" not in os.environ:
        logger.warning(
            "PYTHONHASHSEED is unset; it cannot be applied to the running "
            "interpreter. Export PYTHONHASHSEED=%d before launch for full "
            "determinism in subprocesses.",
            seed,
        )
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info("Global seed set to %d (Python, NumPy, PyTorch)", seed)


def seed_worker(worker_id: int) -> None:
    """
    DataLoader ``worker_init_fn``: reseed NumPy and Python RNGs inside each
    worker process from torch's per-worker seed.

    Without this, every worker inherits the parent's NumPy seed, so any
    NumPy-based augmentation repeats identically across workers and the Phase 8
    three-seed ablation is not reproducible. Pass this together with
    :func:`make_generator` when constructing a DataLoader.

    The ``worker_id`` argument is required by torch's call signature and is
    unused: ``torch.initial_seed()`` already differs per worker.
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = 1729) -> torch.Generator:
    """
    Return a seeded ``torch.Generator`` for a DataLoader's ``generator=``
    argument, which controls shuffling and ``WeightedRandomSampler`` draws.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ── Device detection ────────────────────────────────────────────────────

def detect_device() -> torch.device:
    """
    Detect the best available device: CUDA -> MPS -> CPU.
    Logs the choice and basic info (GPU name, memory).
    """
    logger = get_logger("device")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        mem_gb = props.total_memory / (1024 ** 3)
        logger.info("Device: CUDA — %s (%.1f GB)", props.name, mem_gb)
    elif getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)():
        device = torch.device("mps")
        logger.info("Device: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.warning(
            "Device: CPU (no GPU detected) — training the full train split is "
            "not feasible here, and mixed precision will be inactive."
        )

    return device


# ── Config loading ──────────────────────────────────────────────────────

def _deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Recursively merge *override* into a deep copy of *base*.
    Override values win; nested dicts are merged, not replaced wholesale.
    """
    merged = copy.deepcopy(base)
    for key, val in override.items():
        if key == "_base_":
            continue  # meta-key, not a real config value
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(val, dict)
        ):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = copy.deepcopy(val)
    return merged


def _apply_override(cfg: Dict[str, Any], dotted_key: str, val: Any) -> None:
    """
    Set ``cfg`` at a dotted path, creating intermediate dicts as needed.

    Raises ``TypeError`` naming the offending path if an intermediate key
    already holds a non-dict value, instead of the opaque error a bare
    ``setdefault`` chain produces for a mistyped key such as
    ``training.batch_size.foo``.
    """
    keys = dotted_key.split(".")
    node = cfg
    for depth, key in enumerate(keys[:-1]):
        if key not in node:
            node[key] = {}
        elif not isinstance(node[key], dict):
            path = ".".join(keys[: depth + 1])
            raise TypeError(
                f"Cannot apply override {dotted_key!r}: {path!r} is "
                f"{type(node[key]).__name__}, not a config section."
            )
        node = node[key]
    node[keys[-1]] = val


def load_config(
    config_path: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Load a YAML config file, resolving ``_base_`` inheritance.

    Parameters
    ----------
    config_path : str or Path
        Path to a YAML config file. If it contains a ``_base_`` key, the
        referenced file (resolved relative to the same directory) is loaded
        first and the child's values are merged on top.
    overrides : dict, optional
        Additional key-value overrides applied after merge. Supports dotted
        keys like ``"training.batch_size": 8``. Values are used as given — a
        caller passing CLI strings is responsible for casting them.

    Returns
    -------
    dict
        The fully resolved configuration.
    """
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # Resolve inheritance
    if "_base_" in raw:
        base_path = config_path.parent / raw["_base_"]
        base_cfg = load_config(base_path)  # recursive
        cfg = _deep_merge(base_cfg, raw)
    else:
        cfg = copy.deepcopy(raw)

    # Apply CLI / programmatic overrides
    if overrides:
        for dotted_key, val in overrides.items():
            _apply_override(cfg, dotted_key, val)

    return cfg


def pretty_config(cfg: Dict, indent: int = 0) -> str:
    """Return a human-readable, indented string representation of a config dict."""
    lines = []
    prefix = "  " * indent
    for key, val in cfg.items():
        if isinstance(val, dict):
            lines.append(f"{prefix}{key}:")
            nested = pretty_config(val, indent + 1)
            if nested:  # an empty section would otherwise emit a blank line
                lines.append(nested)
        else:
            lines.append(f"{prefix}{key}: {val}")
    return "\n".join(lines)
