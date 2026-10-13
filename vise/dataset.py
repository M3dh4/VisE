"""
Manifest building, split integrity, and the torch Dataset. (Phase 2 — TO IMPLEMENT)

Limitations to carry into every docstring here:
    - Stiffness labels are **class-level intervals**, not per-pixel ground truth.
      Every pixel of a patch gets the same target.
    - Depth targets are **relative** pseudo-labels from Depth Anything V2. A
      362×362 crop carries no camera intrinsics, so they are not metric.

Known traps, from the Phase 0 audit
-----------------------------------
1. **The class-name join will fail silently on ``polished stone``.** The lookup
   table keys this class with a space, but MINC-2500's image directory is
   ``polished_stone``. Parsing the class from the second path segment of
   ``images/<class>/<file>.jpg`` and joining on it drops 2,500 patches with no
   error and no warning. Normalise underscores to spaces (or vice versa) on both
   sides of the join, then **assert that all 14 included classes matched a
   non-zero row count** before writing any manifest. A silent 2,500-row loss is
   the kind of thing that survives all the way to the results table.

2. **Bin-balanced sampling and interval weighting pull in opposite
   directions.** Verified bin occupancy at ``log_e_bin_width: 0.5``, with 2,500
   images per class:

   ===========  =========  ========  =================================================
   bin          classes    images    members
   ===========  =========  ========  =================================================
   +2.0         5          12,500    ceramic, glass, metal, mirror, polished stone
   +1.5         3           7,500    brick, stone, tile
   -2.5         2           5,000    carpet, fabric
   +0.5         2           5,000    paper, plastic
   -1.0         1           2,500    leather
   +1.0         1           2,500    wood
   ===========  =========  ========  =================================================

   Inverse-bin-frequency sampling upweights ``leather`` **5×**. But leather has
   the widest published interval (half-width 0.85), so the Phase 5
   interval-weighted loss gives it the *lowest* weight: ``1/(0.85 + 0.1) = 1.05``
   against ``glass`` at ``1/(0.13 + 0.1) = 4.39``. Phase 2 pushes leather to the
   front of the queue while Phase 5 tells the loss to discount it.

   The net effect is roughly a wash, but it is uncontrolled and neither the build
   spec nor the review document notices it. Decide explicitly which axis is being
   balanced, log the realised weight range, and write the justification down. Do
   not leave it implicit.

3. **Expected row counts.** MINC-2500 official split 1 is 2,125 / 125 / 250 per
   class, so for 14 classes: train 29,750, val 1,750, test 3,500 — 35,000 total.
   Assert these exactly rather than checking "≈ 35,000"; an off-by-one class in
   the include filter is otherwise invisible.

4. **``image_size: 364``, not 362.** MINC patches are 362×362, which is not
   divisible by the ViT patch size of 14. 364 = 14 × 26. Whatever resize or pad
   is used to get there must be identical across Models A, B and C, since Phase 8
   requires augmentation to be a controlled variable.
"""

# TODO: implement in Phase 2
