# Raytracer Optimization Implementation Plan

## Executive Summary

**Current State:**
- Branch: `temp-test` (Phase 5 complete, opaque shadows only)
- Baseline: 264.45 seconds for 200×200 pool scene
- Main branch: Has transparent shadows but with NO early exits (slower than baseline)

**Goals:**
1. Enhanced progress tracking with ETA and throughput
2. Image comparison tool (strict validation: MSE < 0.001, PSNR > 50dB)
3. Single-threaded optimizations (NO multi-processing)
4. Re-add transparent shadows in optimized way
5. Target: 40%+ speedup (264s → <160s)

**User Preferences:**
- Progress updates: Keep per-row (current behavior)
- Image tolerance: Very strict (MSE < 0.001, PSNR > 50dB)
- Transparent shadows: Command-line flag (defer to future implementation)
- Multi-processing: Not implementing - focus on other optimizations

---

## Phase 1: Foundation - Progress Tracking & Image Comparison

### 1.1 Enhanced Progress Tracking

**File:** [ray_tracer.py:462-525](ray_tracer.py#L462-L525)

**Current implementation (lines 512-518):**
```python
if pixels_rendered % width == 0:  # Per row
    ratio = pixels_rendered / total_pixels
    progress = ratio * 100
    bar = '█' * filled + '░' * (bar_length - filled)
    print(f'\r  Progress: |{bar}| {progress:.1f}% ({pixels_rendered}/{total_pixels})', end='', flush=True)
```

**Changes needed:**
1. Add ETA (estimated time remaining) calculation
2. Add rays/second throughput metric
3. Keep per-row update frequency (user preference)
4. Improve time formatting (MM:SS for ETA)

**Implementation:**
```python
# At start of render():
start_time = time.time()

# In progress update (replace lines 512-518):
if pixels_rendered % width == 0:
    current_time = time.time()
    elapsed = current_time - start_time

    ratio = pixels_rendered / total_pixels
    progress = ratio * 100

    # Calculate rays/sec and ETA
    rays_per_sec = pixels_rendered / elapsed if elapsed > 0 else 0
    if ratio > 0:
        estimated_total = elapsed / ratio
        eta_seconds = estimated_total - elapsed
        eta_min = int(eta_seconds // 60)
        eta_sec = int(eta_seconds % 60)
        eta_str = f"{eta_min:02d}:{eta_sec:02d}"
    else:
        eta_str = "--:--"

    # Progress bar
    bar_length = 40
    filled = int(bar_length * ratio)
    bar = '█' * filled + '░' * (bar_length - filled)

    print(f'\r  Progress: |{bar}| {progress:.1f}% | '
          f'{pixels_rendered}/{total_pixels} px | '
          f'{rays_per_sec:.0f} rays/s | '
          f'ETA: {eta_str}',
          end='', flush=True)
```

**Expected output:**
```
Progress: |████████████░░░░░░░░░░░| 30.0% | 12000/40000 px | 120 rays/s | ETA: 03:45
```

---

### 1.2 Create Optimizations Folder

**New directory:** `optimizations/`

**Purpose:** Centralize all optimization-related artifacts:
- Baseline images for comparison
- Image comparison script
- Performance tracking document
- Test renders

**Structure:**
```
optimizations/
├── compare_images.py       # Image comparison tool
├── performance.md          # Document tracking run times
├── baseline/               # Baseline renders
│   └── pool_200x200.png
└── test/                   # Test renders after optimizations
    └── (test images go here)
```

### 1.3 Image Comparison Script

**New file:** [optimizations/compare_images.py](optimizations/compare_images.py)

**Purpose:** Validate optimizations maintain image quality (MSE < 0.001, PSNR > 50dB)

**Features:**
- Compute MSE (Mean Squared Error)
- Compute PSNR (Peak Signal-to-Noise Ratio)
- Generate visual difference map
- CLI with pass/fail exit codes
- Optional diff image output

**CLI Usage:**
```bash
cd optimizations
python compare_images.py baseline/pool_200x200.png test/pool_200x200.png
python compare_images.py baseline/pool_200x200.png test/pool_200x200.png --save-diff test/diff.png
python compare_images.py baseline/pool_200x200.png test/pool_200x200.png --quiet  # exit code only
```

**Implementation:** ~150 lines
- `load_image(path)` → normalized numpy array [0, 1]
- `compute_mse(img1, img2)` → float MSE
- `compute_psnr(img1, img2)` → float PSNR in dB
- `create_difference_map(img1, img2)` → visual diff (scaled for visibility)
- `compare_images()` → main comparison logic
- `main()` → CLI argument parsing

**Validation thresholds:**
```python
MSE_THRESHOLD = 0.001   # Very strict
PSNR_THRESHOLD = 50.0   # Very strict (>50dB = excellent quality)
```

**Exit codes:**
- 0 = images match (within threshold)
- 1 = images differ (beyond threshold)
- 2 = error (file not found, etc.)

### 1.4 Performance Tracking Document

**New file:** [optimizations/performance.md](optimizations/performance.md)

**Purpose:** Track render times across optimizations

**Template:**
```markdown
# Raytracer Performance Tracking

## Test Configuration
- Scene: scenes/pool.txt
- Resolution: 200×200 pixels
- System: [CPU, OS info]

## Baseline (temp-test branch)
- **Time:** 264.45 seconds (4.41 minutes)
- **Image:** baseline/pool_200x200.png
- **Date:** YYYY-MM-DD

## Phase 1: Progress Tracking
- **Changes:** Enhanced progress bar with ETA and rays/sec
- **Time:** [To be filled]
- **Speedup:** 0% (no performance changes expected)
- **Image:** test/phase1_pool_200x200.png
- **Validation:** [MSE, PSNR values]

## Phase 2.1: Vectorized Shadow Rays
- **Changes:** Batch intersection methods, vectorized shadow computation
- **Time:** [To be filled]
- **Speedup:** [Expected ~30%]
- **Image:** test/phase2.1_pool_200x200.png
- **Validation:** [MSE, PSNR values]

[... continues for each optimization phase ...]
```

---

## Phase 2: Single-Threaded Optimizations

### 2.1 Vectorized Shadow Rays (Highest Impact)

**Impact:** 30% speedup (proven in optimization branch commit 9e7faa3)

**Bottleneck:** Current `compute_shadow_ray_ratio()` (lines 90-129):
- Nested loops: `for i in range(num_shadow_rays): for j in range(num_shadow_rays):`
- Each iteration calls `is_occluded()` which loops over surfaces
- Pool scene: 5 lights × 25 shadow rays × 7 surfaces = 875 tests per pixel
- Total: 40,000 pixels × 875 = 35,000,000 intersection tests

**Solution:** Process all N×N shadow rays simultaneously using NumPy broadcasting

**Step 1: Add batch intersection methods to surfaces**

**File:** [surfaces/sphere.py](surfaces/sphere.py)

```python
def intersect_batch(self, ray_origins, ray_directions):
    """
    Test multiple rays against this sphere simultaneously.

    Args:
        ray_origins: np.ndarray, shape (N, 3)
        ray_directions: np.ndarray, shape (N, 3) - must be normalized

    Returns:
        t_values: np.ndarray, shape (N,) - distance to nearest intersection
                  np.inf where no intersection occurs
    """
    # Vectorized quadratic formula for sphere intersection
    # See commit 3a47846 for reference implementation

    to_sphere = ray_origins - self.position  # (N, 3)

    # Quadratic coefficients: at² + bt + c = 0
    a = np.sum(ray_directions ** 2, axis=1)  # Should be 1.0 if normalized
    b = 2 * np.sum(to_sphere * ray_directions, axis=1)  # (N,)
    c = np.sum(to_sphere ** 2, axis=1) - self.radius ** 2  # (N,)

    discriminant = b**2 - 4*a*c

    # Initialize with no intersection
    t_values = np.full(len(ray_origins), np.inf)

    # Rays with valid intersections
    valid_mask = discriminant >= 0

    if np.any(valid_mask):
        sqrt_disc = np.sqrt(discriminant[valid_mask])
        a_valid = a[valid_mask]
        b_valid = b[valid_mask]

        t1 = (-b_valid - sqrt_disc) / (2 * a_valid)
        t2 = (-b_valid + sqrt_disc) / (2 * a_valid)

        # Choose nearest positive t
        t_near = np.where(t1 > EPSILON, t1, t2)
        t_values[valid_mask] = np.where(t_near > EPSILON, t_near, np.inf)

    return t_values
```

**Similarly for:** [surfaces/infinite_plane.py](surfaces/infinite_plane.py), [surfaces/cube.py](surfaces/cube.py)

**Step 2: Vectorize shadow ray generation**

**File:** [ray_tracer.py](ray_tracer.py)

```python
def generate_light_samples_vectorized(light, light_right, light_up, num_shadow_rays):
    """
    Generate all N×N shadow sample points at once.

    Returns:
        np.ndarray, shape (N*N, 3) - all sample points on light surface
    """
    total_samples = num_shadow_rays * num_shadow_rays
    samples = np.zeros((total_samples, 3))

    idx = 0
    for i in range(num_shadow_rays):
        for j in range(num_shadow_rays):
            # Jittered grid sampling
            u = (i + np.random.random()) / num_shadow_rays
            v = (j + np.random.random()) / num_shadow_rays

            # Map to [-1, 1]
            u = 2 * u - 1
            v = 2 * v - 1

            # Sample point on light
            offset = light_right * (u * light.radius) + light_up * (v * light.radius)
            samples[idx] = light.position + offset
            idx += 1

    return samples
```

**Step 3: Replace compute_shadow_ray_ratio with vectorized version**

```python
def compute_shadow_ray_ratio_vectorized(hit_point, light, surfaces, num_shadow_rays):
    """Vectorized shadow ray computation (30% faster)."""

    # Handle point lights
    to_light_dir = normalize(light.position - hit_point)
    if light.shadow_intensity == 0 or num_shadow_rays == 1:
        return 1.0 if not is_occluded(hit_point, light.position, surfaces) else 0.0

    # Generate all shadow samples
    light_right, light_up = create_light_basis(to_light_dir)
    sample_points = generate_light_samples_vectorized(light, light_right, light_up, num_shadow_rays)

    total_samples = len(sample_points)

    # Prepare ray origins and directions
    ray_origins = np.tile(hit_point, (total_samples, 1))  # (N, 3)
    ray_directions = sample_points - ray_origins  # (N, 3)
    distances = np.linalg.norm(ray_directions, axis=1)  # (N,)
    ray_directions = ray_directions / distances[:, np.newaxis]  # Normalize

    # Track which rays are unoccluded
    unoccluded_mask = np.ones(total_samples, dtype=bool)

    # Test all surfaces
    for surface in surfaces:
        t_values = surface.intersect_batch(ray_origins, ray_directions)

        # Mark rays that hit this surface before reaching light
        blocking_mask = (t_values > EPSILON) & (t_values < distances)
        unoccluded_mask &= ~blocking_mask  # Set to False if blocked

    # Return ratio of unoccluded rays
    return np.sum(unoccluded_mask) / total_samples
```

**Expected speedup:** 30% (264s → 185s)

---

### 2.2 Pre-compute Material Properties

**Impact:** 3-5% speedup

**Current issue:** Repeated material property lookups in tight loops

**Solution:** Pre-convert materials to numpy arrays at render start

**File:** [ray_tracer.py:462](ray_tracer.py#L462) (start of `render()`)

```python
def render(camera, scene_settings, materials, surfaces, lights, width, height):
    start_time = time.time()

    # PRE-COMPUTE: Convert materials to numpy arrays for faster access
    num_materials = len(materials)
    material_diffuse = np.array([m.diffuse_color for m in materials])      # (N, 3)
    material_specular = np.array([m.specular_color for m in materials])    # (N, 3)
    material_reflection = np.array([m.reflection_color for m in materials]) # (N, 3)
    material_transparency = np.array([m.transparency for m in materials])   # (N,)
    material_shininess = np.array([m.shininess for m in materials])         # (N,)

    # Pass these to functions that need materials
    # ...
```

**Changes needed:**
- Update `phong_shading()` signature to accept material arrays
- Update `trace_ray()` to pass material arrays
- Index materials by `surface.material_index - 1` (materials are 1-indexed)

**Expected speedup:** 3-5% (185s → 179s)

---

### 2.3 Reduce Normalize Calls

**Impact:** 1-2% speedup

**Current issue:** `normalize()` called repeatedly on same vectors

**File:** [ray_tracer.py](ray_tracer.py) - various functions

**Optimization 1:** In `compute_shadow_ray_ratio()` (line 105):
```python
# Before:
to_light_dir = normalize(to_light)

# After: Combine with distance calculation
to_light = light.position - hit_point
light_dist = np.linalg.norm(to_light)
to_light_dir = to_light / light_dist  # Single division
```

**Optimization 2:** In Phong shading:
```python
# Normalize once and reuse
# Cache light_direction instead of recomputing
```

**Expected speedup:** 1-2% (179s → 176s)

---

### 2.4 Early Exit in Shadow Tests

**Impact:** 5-10% speedup for scenes with opaque objects

**Current:** All surfaces tested even if early hits block light

**Optimization:** In vectorized shadow rays, skip testing remaining surfaces if all rays are blocked

```python
def compute_shadow_ray_ratio_vectorized(hit_point, light, surfaces, num_shadow_rays):
    # ... (setup)

    unoccluded_mask = np.ones(total_samples, dtype=bool)

    for surface in surfaces:
        # OPTIMIZATION: Early exit if all rays blocked
        if not np.any(unoccluded_mask):
            return 0.0  # All rays occluded, no point testing more surfaces

        t_values = surface.intersect_batch(ray_origins, ray_directions)
        blocking_mask = (t_values > EPSILON) & (t_values < distances)
        unoccluded_mask &= ~blocking_mask

    return np.sum(unoccluded_mask) / total_samples
```

**Expected speedup:** 5-10% (176s → 160s)

---

## Phase 3: Re-add Transparent Shadows (Optimized)

### 3.1 Problem with Main Branch

**Main branch issue (commit de3f117):**
- `compute_light_transmission()` tests ALL surfaces with NO early exit
- Even when opaque surface blocks light, continues testing remaining surfaces
- Causes 20-30% slowdown vs baseline

### 3.2 Optimized Implementation

**Strategy:** Integrate transparency into vectorized shadow rays with early exits

**File:** [ray_tracer.py](ray_tracer.py)

**New function:** `compute_shadow_ray_ratio_with_transparency()`

```python
def compute_shadow_ray_ratio_with_transparency(hit_point, light, surfaces, materials, num_shadow_rays):
    """
    Vectorized shadow computation with transparent shadow support.

    Key optimizations:
    1. Vectorized ray processing (30% faster)
    2. Early exit when material is opaque (10-15% faster)
    3. Early exit when all rays are fully blocked (5-10% faster)
    """
    # ... (setup as before)

    # Initialize light transmission factors [0, 1]
    light_factors = np.ones(total_samples, dtype=np.float64)

    for surface in surfaces:
        # OPTIMIZATION 1: Skip if all light blocked
        if np.all(light_factors == 0.0):
            return 0.0

        t_values = surface.intersect_batch(ray_origins, ray_directions)
        blocking_mask = (t_values > EPSILON) & (t_values < distances)

        if np.any(blocking_mask):
            material = materials[surface.material_index - 1]

            # OPTIMIZATION 2: Early exit if opaque
            if material.transparency == 0.0:
                light_factors[blocking_mask] = 0.0
            else:
                # Accumulate transparency
                light_factors[blocking_mask] *= material.transparency

    return np.mean(light_factors)
```

**Toggle mechanism:** Defer command-line flag to future (user preference)
- For now, detect if scene has transparent materials
- If all materials are opaque (transparency = 0), use faster non-transparent version
- If any material is transparent, use transparency-aware version

**Detection in render():**
```python
# Check if scene needs transparent shadows
has_transparency = any(m.transparency > 0 for m in materials)

# Choose appropriate shadow function
shadow_func = (compute_shadow_ray_ratio_with_transparency
               if has_transparency
               else compute_shadow_ray_ratio_vectorized)
```

**Expected performance:**
- Opaque scenes: Same as Phase 2 (~160s)
- Transparent scenes: 5-10% overhead (~170s) but still 35% faster than baseline

---

## Phase 4: Validation & Testing

### 4.1 Validation Workflow

**Step 1: Baseline render**
```bash
# Create optimizations directory
mkdir -p optimizations/baseline optimizations/test

# Render baseline (current temp-test branch)
python ray_tracer.py scenes/pool.txt optimizations/baseline/pool_200x200.png --width 200 --height 200
# Expected: 264.45s

# Document baseline time in optimizations/performance.md
```

**Step 2: After each optimization**
```bash
# Render with optimization
python ray_tracer.py scenes/pool.txt optimizations/test/phaseX_pool_200x200.png --width 200 --height 200

# Validate image quality
cd optimizations
python compare_images.py baseline/pool_200x200.png test/phaseX_pool_200x200.png --save-diff test/phaseX_diff.png

# Should output:
# ✓ PASS: Images are equivalent within tolerance
# MSE: 0.000123 (< 0.001)
# PSNR: 58.45 dB (> 50 dB)

# Update performance.md with results
```

**Step 3: Test transparent shadows**
```bash
# Checkout main branch
git checkout main
python ray_tracer.py scenes/pool.txt optimizations/baseline/main_transparent.png --width 200 --height 200

# Checkout optimized branch
git checkout temp-test  # or optimization branch
python ray_tracer.py scenes/pool.txt optimizations/test/optimized_transparent.png --width 200 --height 200

# Compare
cd optimizations
python compare_images.py baseline/main_transparent.png test/optimized_transparent.png
```

### 4.2 Test Scenes

**Test 1:** Pool scene (baseline)
- 200×200 pixels
- 7 surfaces (6 spheres + 1 plane)
- 5 lights
- All opaque materials

**Test 2:** Transparency test
- Modify pool.txt: Set sphere transparency = 0.5
- Verify transparent shadows work correctly
- Compare performance vs opaque

**Test 3:** Large image
- 400×400 pixels
- Verify progress tracking updates correctly
- Check ETA accuracy

---

## Implementation Sequence with Git Workflow

### Phase 1: Foundation (1-2 hours)

**1.1 Setup optimizations directory**
```bash
mkdir -p optimizations/baseline optimizations/test
```

**1.2 Create image comparison script**
- Write `optimizations/compare_images.py`
- Test it works correctly

**1.3 Create performance tracking document**
- Write `optimizations/performance.md` with template

**1.4 Capture baseline**
```bash
# Render baseline
python ray_tracer.py scenes/pool.txt optimizations/baseline/pool_200x200.png --width 200 --height 200

# Document time in performance.md
```

**1.5 Enhanced progress tracking**
- Modify render() in ray_tracer.py
- Test with small render (50x50) to verify progress bar works
- Render full 200x200 to verify no performance regression

**1.6 Git commit - Phase 1**
```bash
git add optimizations/ ray_tracer.py .gitignore
git commit -m "Phase 1: Add progress tracking and image comparison tools

- Enhanced progress bar with ETA and rays/sec metrics
- Created optimizations/ directory structure
- Added compare_images.py for strict image validation (MSE < 0.001)
- Added performance.md for tracking optimization results
- Captured baseline render (264.45s for 200×200 pool scene)

🤖 Generated with Claude Code
Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

**STOP: Wait for user approval before proceeding to Phase 2**

---

### Phase 2: Optimizations (3-4 hours)

**2.1 Add batch intersection to Sphere**
- Implement `intersect_batch()` in surfaces/sphere.py

**2.2 Test and commit Sphere batch intersection**
```bash
# Quick test with small render
python ray_tracer.py scenes/pool.txt optimizations/test/test_sphere_batch.png --width 50 --height 50

git add surfaces/sphere.py
git commit -m "Optimization: Add batch intersection method to Sphere

- Implement intersect_batch() for vectorized ray testing
- Part of shadow ray vectorization optimization

🤖 Generated with Claude Code
Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

**STOP: Test and approve before continuing**

---

**2.3 Add batch intersection to InfinitePlane**
- Implement `intersect_batch()` in surfaces/infinite_plane.py

**2.4 Test and commit InfinitePlane batch intersection**
```bash
git add surfaces/infinite_plane.py
git commit -m "Optimization: Add batch intersection method to InfinitePlane

- Implement intersect_batch() for vectorized ray testing
- Part of shadow ray vectorization optimization

🤖 Generated with Claude Code
Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

**STOP: Test and approve before continuing**

---

**2.5 Add batch intersection to Cube**
- Implement `intersect_batch()` in surfaces/cube.py

**2.6 Test and commit Cube batch intersection**
```bash
git add surfaces/cube.py
git commit -m "Optimization: Add batch intersection method to Cube

- Implement intersect_batch() for vectorized ray testing
- Part of shadow ray vectorization optimization

🤖 Generated with Claude Code
Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

**STOP: Test and approve before continuing**

---

**2.7 Vectorize shadow ray generation and computation**
- Add `generate_light_samples_vectorized()` to ray_tracer.py
- Add `compute_shadow_ray_ratio_vectorized()` to ray_tracer.py
- Update `calculate_light_intensity()` to use vectorized version

**2.8 Test and commit vectorization**
```bash
# Full test render
python ray_tracer.py scenes/pool.txt optimizations/test/phase2_vectorized.png --width 200 --height 200

# Compare images
cd optimizations
python compare_images.py baseline/pool_200x200.png test/phase2_vectorized.png --save-diff test/phase2_diff.png

# Update performance.md with timing results

git add ray_tracer.py optimizations/performance.md
git commit -m "Optimization: Vectorize shadow ray computation - 30% speedup

- Implement vectorized shadow ray generation
- Replace compute_shadow_ray_ratio() with vectorized version
- Process all N×N shadow rays simultaneously using NumPy
- Measured speedup: [X]% (update with actual timing)
- Validation: MSE < 0.001, PSNR > 50dB vs baseline

🤖 Generated with Claude Code
Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

**STOP: Test performance, validate images, and approve before continuing**

---

**2.9 Add early exit optimization**
- Modify `compute_shadow_ray_ratio_vectorized()` to exit when all rays blocked

**2.10 Test and commit early exit**
```bash
python ray_tracer.py scenes/pool.txt optimizations/test/phase2_early_exit.png --width 200 --height 200
cd optimizations
python compare_images.py baseline/pool_200x200.png test/phase2_early_exit.png

# Update performance.md

git add ray_tracer.py optimizations/performance.md
git commit -m "Optimization: Add early exit for blocked shadow rays

- Skip testing remaining surfaces when all rays blocked
- Measured speedup: [X]% (update with actual timing)
- Validation: Image quality maintained

🤖 Generated with Claude Code
Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

**STOP: Test and approve before continuing**

---

**2.11 Pre-compute material arrays**
- Add material array pre-computation at start of render()
- Update functions to use material arrays

**2.12 Test and commit material pre-computation**
```bash
python ray_tracer.py scenes/pool.txt optimizations/test/phase2_materials.png --width 200 --height 200
cd optimizations
python compare_images.py baseline/pool_200x200.png test/phase2_materials.png

# Update performance.md

git add ray_tracer.py optimizations/performance.md
git commit -m "Optimization: Pre-compute material properties as arrays

- Convert materials to numpy arrays at render start
- Reduce Python object access overhead
- Measured speedup: [X]% (update with actual timing)

🤖 Generated with Claude Code
Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

**STOP: Test and approve before continuing**

---

**2.13 Reduce normalize calls**
- Optimize normalize operations in shadow ray code

**2.14 Test and commit normalize optimization**
```bash
python ray_tracer.py scenes/pool.txt optimizations/test/phase2_final.png --width 200 --height 200
cd optimizations
python compare_images.py baseline/pool_200x200.png test/phase2_final.png

# Update performance.md with final Phase 2 results

git add ray_tracer.py optimizations/performance.md
git commit -m "Optimization: Reduce redundant normalize calls

- Combine normalize with distance calculations
- Cache normalized vectors where possible
- Final Phase 2 speedup: [X]% (264s → [Y]s)

🤖 Generated with Claude Code
Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

**STOP: Test final Phase 2 results and approve before proceeding to Phase 3**

---

### Phase 3: Transparent Shadows (2-3 hours)

**3.1 Implement transparent shadow support**
- Add `compute_shadow_ray_ratio_with_transparency()` to ray_tracer.py
- Add automatic transparency detection in render()
- Route to appropriate shadow function based on scene materials

**3.2 Test with opaque scene (should match Phase 2)**
```bash
python ray_tracer.py scenes/pool.txt optimizations/test/phase3_opaque.png --width 200 --height 200
cd optimizations
python compare_images.py test/phase2_final.png test/phase3_opaque.png
# Should be identical (or extremely close)
```

**3.3 Create test scene with transparency**
```bash
# Modify pool.txt to add transparent sphere
cp scenes/pool.txt scenes/pool_transparent.txt
# Edit to set transparency > 0 for one material
```

**3.4 Test transparent shadows**
```bash
python ray_tracer.py scenes/pool_transparent.txt optimizations/test/phase3_transparent.png --width 200 --height 200
# Visual inspection - should see lighter shadows through glass
```

**3.5 Compare against main branch**
```bash
git checkout main
python ray_tracer.py scenes/pool_transparent.txt optimizations/baseline/main_transparent.png --width 200 --height 200

git checkout temp-test
cd optimizations
python compare_images.py baseline/main_transparent.png test/phase3_transparent.png
# Update performance.md
```

**3.6 Git commit - Phase 3 (Transparent Shadows)**
```bash
git add ray_tracer.py scenes/pool_transparent.txt optimizations/performance.md
git commit -m "Phase 3: Add optimized transparent shadow support

- Implement compute_shadow_ray_ratio_with_transparency()
- Automatic detection of transparent materials in scene
- Early exit optimization for opaque surfaces
- Vectorized transparency accumulation
- Performance: 5-10% overhead vs opaque (still 35%+ faster than original)
- Validation: Matches main branch output for transparent scenes

🤖 Generated with Claude Code
Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

**STOP: Wait for user approval before final testing**

---

### Phase 4: Final Testing & Documentation (1 hour)

**4.1 Comprehensive testing**
```bash
# Test various resolutions
python ray_tracer.py scenes/pool.txt optimizations/test/pool_100x100.png --width 100 --height 100
python ray_tracer.py scenes/pool.txt optimizations/test/pool_400x400.png --width 400 --height 400

# Test other scenes if available
# Document all results in performance.md
```

**4.2 Update .gitignore**
```bash
# Add to .gitignore:
optimizations/baseline/*.png
optimizations/test/*.png
```

**4.3 Final documentation commit**
```bash
git add optimizations/performance.md .gitignore README.md
git commit -m "Phase 4: Final testing and documentation

- Comprehensive testing across multiple resolutions
- Updated performance tracking document with all results
- Added .gitignore entries for test images
- Final speedup achieved: [X]% (264s → [Y]s)

🤖 Generated with Claude Code
Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

**STOP: Final review with user**

---

**Total estimated time: 7-10 hours**

**Note:** After each STOP point, wait for user to:
1. Test the changes manually
2. Verify image quality
3. Check performance numbers
4. Approve before proceeding or request fixes

---

## Success Metrics

1. **Performance:** 40%+ speedup (264s → <160s for 200×200 pool scene)
2. **Quality:** MSE < 0.001, PSNR > 50dB vs baseline
3. **Features:** Transparent shadows working correctly (when materials are transparent)
4. **Code:** Maintainable, well-documented, tested

---

## Critical Files to Modify

### Code Files
1. **[ray_tracer.py:462-525](ray_tracer.py#L462-L525)** - render() function (progress tracking)
2. **[ray_tracer.py:90-129](ray_tracer.py#L90-L129)** - Shadow ray computation (vectorization)
3. **[ray_tracer.py:16-36](ray_tracer.py#L16-L36)** - is_occluded() (may be replaced)
4. **[surfaces/sphere.py](surfaces/sphere.py)** - Add intersect_batch()
5. **[surfaces/infinite_plane.py](surfaces/infinite_plane.py)** - Add intersect_batch()
6. **[surfaces/cube.py](surfaces/cube.py)** - Add intersect_batch()

### New Files/Directories
7. **[optimizations/](optimizations/)** - NEW directory for optimization artifacts
8. **[optimizations/compare_images.py](optimizations/compare_images.py)** - NEW file (image comparison)
9. **[optimizations/performance.md](optimizations/performance.md)** - NEW file (performance tracking)
10. **[optimizations/baseline/](optimizations/baseline/)** - NEW directory (baseline images)
11. **[optimizations/test/](optimizations/test/)** - NEW directory (test images)
12. **[.gitignore](.gitignore)** - Update to ignore test images

---

## Risk Mitigation

**Risk:** Vectorization changes numerical results
**Mitigation:** Use float64, validate with strict thresholds (MSE < 0.001)

**Risk:** Batch intersection bugs
**Mitigation:** Test batch vs single for same rays, visual inspection

**Risk:** Transparent shadows look different
**Mitigation:** Compare against main branch, test with known values

**Risk:** Early exits break correctness
**Mitigation:** Only exit when mathematically safe (light_factor == 0.0)
