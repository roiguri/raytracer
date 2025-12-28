# Raytracer Performance Tracking

## Test Configuration
- **Scene:** scenes/pool.txt
- **Resolution:** 200×200 pixels
- **System:** 11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz
- **OS:** Linux 6.6.87.2-microsoft-standard-WSL2
- **Date Started:** 2025-12-28

---

## Baseline (temp-test branch)
- **Time:** 264.45 seconds (4.41 minutes)
- **Image:** baseline/pool_200x200.png
- **Date:** 2025-12-28
- **Notes:** Phase 5 complete, opaque shadows only

---

## Phase 1: Progress Tracking & Image Comparison
- **Changes:** Enhanced progress bar with ETA and rays/sec
- **Time:** [To be measured]
- **Speedup:** 0% (no performance changes expected)
- **Image:** test/phase1_pool_200x200.png
- **Validation:** [MSE, PSNR values to be filled]
- **Status:** In progress

---

## Phase 2.1: Batch Intersection Methods
- **Changes:** Add intersect_batch() to Sphere, InfinitePlane, Cube
- **Time:** [To be measured]
- **Speedup:** 0% (preparatory work, no immediate speedup)
- **Image:** test/test_sphere_batch.png
- **Validation:** [To be filled]
- **Status:** Not started

---

## Phase 2.2: Vectorized Shadow Rays
- **Changes:** Vectorized shadow ray generation and computation
- **Time:** [To be measured]
- **Expected Speedup:** ~30% (264s → ~185s)
- **Actual Speedup:** [To be filled]
- **Image:** test/phase2_vectorized.png
- **Validation:** [MSE, PSNR values to be filled]
- **Status:** Not started

---

## Phase 2.3: Early Exit Optimization
- **Changes:** Skip testing surfaces when all shadow rays blocked
- **Time:** [To be measured]
- **Expected Speedup:** +5-10%
- **Actual Speedup:** [To be filled]
- **Image:** test/phase2_early_exit.png
- **Validation:** [MSE, PSNR values to be filled]
- **Status:** Not started

---

## Phase 2.4: Pre-compute Material Arrays
- **Changes:** Convert materials to numpy arrays at render start
- **Time:** [To be measured]
- **Expected Speedup:** +3-5%
- **Actual Speedup:** [To be filled]
- **Image:** test/phase2_materials.png
- **Validation:** [MSE, PSNR values to be filled]
- **Status:** Not started

---

## Phase 2.5: Reduce Normalize Calls
- **Changes:** Optimize normalize operations in tight loops
- **Time:** [To be measured]
- **Expected Speedup:** +1-2%
- **Actual Speedup:** [To be filled]
- **Image:** test/phase2_final.png
- **Validation:** [MSE, PSNR values to be filled]
- **Status:** Not started

---

## Phase 3: Transparent Shadows (Optimized)
- **Changes:** Add optimized transparent shadow support with early exits
- **Time (opaque):** [To be measured]
- **Time (transparent):** [To be measured]
- **Expected Overhead:** 5-10% for transparent scenes
- **Image (opaque):** test/phase3_opaque.png
- **Image (transparent):** test/phase3_transparent.png
- **Validation:** [MSE, PSNR values vs main branch]
- **Status:** Not started

---

## Summary

| Phase | Time (s) | Speedup vs Baseline | Cumulative Speedup |
|-------|----------|---------------------|-------------------|
| Baseline | 264.45 | 0% | 0% |
| Phase 1 | [TBD] | [TBD] | [TBD] |
| Phase 2.2 | [TBD] | [TBD] | [TBD] |
| Phase 2.3 | [TBD] | [TBD] | [TBD] |
| Phase 2.4 | [TBD] | [TBD] | [TBD] |
| Phase 2.5 | [TBD] | [TBD] | [TBD] |
| Phase 3 | [TBD] | [TBD] | [TBD] |

**Target:** 40%+ speedup (264s → <160s)
