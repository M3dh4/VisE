"""
VisE — Vision-based Stiffness Estimation

Single-image, single-forward-pass, dense per-pixel stiffness (Young's modulus)
jointly with metric depth, using a shared DepthAnything V2 encoder.

Stiffness is predicted in log₁₀ space (GPa) and trained on real annotated
material data from MINC-2500.

Limitations:
    - Stiffness labels are class-level intervals, not per-pixel ground truth.
    - Pseudo-depth labels on MINC crops are relative only (no camera intrinsics).
    - The SAM consistency loss is nearly degenerate on single-material MINC patches.
"""

__version__ = "0.1.0"
