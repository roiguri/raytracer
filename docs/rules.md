# Ray Tracing Rules & Best Practices

This document captures important rules, mathematical principles, and best practices discovered during implementation.

---

## General Programming Rules

### 1. Vector Operations
- **Always normalize direction vectors** before using them in calculations
- **Check for zero-length vectors** before normalizing to avoid division by zero
- Use NumPy for vector operations when possible (faster and more accurate)

### 2. Floating Point Comparisons
- **Never use exact equality** for floats (e.g., `t == 0`)
- Use small epsilon values for comparisons: `EPSILON = 1e-6`
- Defined in `constants.py` for project-wide consistency
- Example: `if t > EPSILON:` instead of `if t > 0:`

### 3. Ray Intersection Testing
- **Always return the smallest positive t value** (closest intersection in front of camera)
- Negative t values mean the intersection is behind the ray origin
- Store intersection data: `(t, normal)` for later shading calculations

---

## Ray-Object Intersection Rules

### Ray-Sphere Intersection
**Quadratic equation approach:**
- Solve: `|O + t*D - C|² = r²`
- Discriminant < 0: no intersection
- Discriminant ≥ 0: use smallest positive t
- Normal: `(hit_point - center) / radius`

### Ray-Plane Intersection
**Linear equation approach:**
- Solve: `t = (offset - O·N) / (D·N)`
- **Critical**: Always normalize plane normal in constructor
- Scene files can contain non-unit-length normals
- Check for parallel rays: `|D·N| < EPSILON`
- Normal: always `self.normal` (constant for infinite planes)

### Ray-Cube Intersection (Slab Method)
**Three-slab intersection approach:**
- Calculate entry/exit t for each axis pair
- t_near = max of all entry times (last entry)
- t_far = min of all exit times (first exit)
- If t_near > t_far: ray misses cube

**Edge Case: Ray Inside Cube**
- If ray origin is inside the cube, `t_near` will be negative
- Must use `t_far` (exit point) instead
- **Critical**: Normal must be recalculated for exit point, not entry point
- Check which face we're exiting through by comparing hit point to box boundaries
- Wrong normal → incorrect shading and reflection directions

**Why this edge case matters:**
- Transparent objects: rays can pass through and start inside
- Refraction: transmitted rays often originate inside objects
- Camera inside geometry: artistic effects or debug views

---

## Mathematical Principles

### Ray Representation
A ray is defined by:
- **Origin (O)**: Starting point (often the camera position)
- **Direction (D)**: Normalized direction vector
- **Parametric form**: `P(t) = O + t*D` where `t ≥ 0`

### Camera Coordinate System
1. **Forward vector**: `forward = normalize(look_at - position)`
2. **Right vector**: `right = normalize(cross(forward, up_vector))`
3. **True up vector**: `up = cross(right, forward)` (orthogonal to forward and right)

Note: The input up_vector is just a hint; the true up must be perpendicular to forward.

**Why Up Vector is a "Hint":**
- Per assignment (page 3): "For convenience, this vector is not necessary perpendicular to the direction vector"
- User provides approximate "up" direction
- We calculate the TRUE up that's perpendicular using cross products
- This makes scene files easier to write (no need for precise calculations)

### Pixel Coordinate Normalization
**Why [-0.5, 0.5] range:**
- Centers the coordinate system around (0, 0)
- Center pixel at (width/2, height/2) maps to (0, 0)
- Ensures center ray points exactly forward
- Creates symmetric ray distribution around center
- Formula: `norm = (pixel + 0.5) / image_size - 0.5`
  - The +0.5 targets the pixel center (not corner)
  - The -0.5 shifts range to be centered at zero

### Phong Reflection Model (Phase 3)
In ray tracing, we use: **Color = Diffuse + Specular** (no ambient term)

- **Diffuse**: `Kd * I * max(0, N·L)`
  - Kd = diffuse color (from material)
  - I = light color/intensity
  - N = surface normal (unit vector)
  - L = light direction (unit vector, FROM surface TO light)
  - max(0, ...) ensures back-facing surfaces get no light

- **Specular**: `Ks * I * max(0, R·V)^α`
  - Ks = specular color (from material)
  - I = light color/intensity
  - R = reflection direction: `R = 2N(N·L) - L`
  - V = view direction (FROM surface TO camera)
  - α = shininess exponent (higher = sharper highlight)

**Why no ambient?**
- Background color provides base illumination when rays miss
- Later: reflection/transparency rays provide indirect lighting
- This matches the assignment specification

**View Direction:**
- Must point FROM hit point TO camera
- Formula: `view_dir = normalize(camera_position - hit_point)`
- Used for specular calculation only

### Finding Nearest Intersection
**min_t parameter prevents self-intersection:**
- When casting secondary rays (shadows, reflections) from a surface
- Floating-point errors can cause ray to immediately hit its origin surface
- Solution: ignore intersections closer than `min_t = 0.001`
- Applied in `find_nearest_intersection()` function

### Shadow Ray Bias
- **Problem**: Shadow rays may intersect the surface they originate from due to floating point errors
- **Solution**: Start shadow rays slightly offset from the surface: `shadow_origin = hit_point + EPSILON * normal`

---

## Performance Best Practices

### 1. Early Exit Optimization
- Return as soon as you know there's no intersection
- Example: For spheres, if discriminant < 0, return immediately

### 2. Avoid Redundant Calculations
- Pre-compute values used multiple times
- Example: Store `|D|²` instead of recalculating it

### 3. NumPy Vectorization
- Process multiple rays/pixels simultaneously when possible
- Use NumPy array operations instead of Python loops
- Example: `np.dot()` is much faster than manual dot product loops

---

## Debugging Strategies

### 1. Incremental Testing
- Test each component independently before combining
- Start with simple scenes (one sphere, one light)
- Gradually increase complexity

### 2. Visual Debugging
- Render silhouettes first (binary hit/no-hit)
- Test with extreme material values (pure white, pure black)
- Use color coding to debug normals: `color = (normal + 1) / 2`

### 3. Center Pixel Test
- Always verify the center pixel first
- Print ray direction for pixel (width/2, height/2)
- Should exactly match camera forward vector

---

## Common Pitfalls

### 1. Wrong Normal Direction
- **Problem**: Normals pointing inward instead of outward
- **Solution**: For spheres, normal = `(hit_point - center) / radius`
- **Check**: Dot product of normal and view direction should be negative

### 2. Off-by-One Errors in Pixel Coordinates
- Image coordinates: (0, 0) to (width-1, height-1)
- Pixel centers are at 0.5 offsets: (0.5, 0.5) to (width-0.5, height-0.5)

### 3. Color Clamping
- **Problem**: Color values can exceed [0, 1] range after multiple reflections
- **Solution**: Always clamp final colors: `np.clip(color, 0, 1)`

### 4. Self-Intersection
- **Problem**: Reflected/shadow rays immediately hit their origin surface
- **Solution**: Add small epsilon offset along normal direction

---

## Code Organization

### 1. Separation of Concerns
- Keep geometry (intersection) separate from shading
- Each surface type should handle its own intersection math
- Ray tracer coordinates overall rendering logic

### 2. Consistent Return Formats
- Intersection methods should return `None` or `(t, point, normal)`
- Use consistent vector representations (NumPy arrays)

### 3. Magic Numbers
- Define constants at the top of files
- Example: `EPSILON = 1e-6`, `MAX_RAY_DEPTH = 10`

---

## Scene File Format Notes

### Order of Operations
1. Parse camera (cam) - defines view
2. Parse settings (set) - defines global parameters
3. Parse materials (mtl) - creates material palette
4. Parse objects (sph, pln, box) - references materials by index
5. Parse lights (lgt) - defines light sources

### Index-Based References
- Materials are indexed starting from 0
- Objects reference materials by index
- Make sure material exists before referencing it

---

## Testing Checklist

### Per-Phase Testing
- [ ] Unit test individual functions with known inputs
- [ ] Visual test with simple scene
- [ ] Compare output to reference implementation
- [ ] Check edge cases (perpendicular rays, grazing angles)
- [ ] Verify performance is reasonable

### Final Integration Testing
- [ ] Render pool.txt scene
- [ ] Compare pixel-by-pixel with reference pool.png
- [ ] Test with different image resolutions
- [ ] Test with extreme scene parameters
- [ ] Profile for performance bottlenecks

---

## Resources & References

### Mathematical Resources
- Linear algebra: Vector operations, cross product, dot product
- Ray-surface intersection formulas
- Phong reflection model
- Monte Carlo sampling for soft shadows

### Python/NumPy Tips
- Use `np.linalg.norm()` for vector magnitude
- Use `np.cross()` for cross product
- Use `np.dot()` for dot product
- Use `np.clip()` for clamping values

---

**Note**: This document will be updated as we discover new rules and best practices during implementation.
