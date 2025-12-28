import argparse
from PIL import Image
import numpy as np
import time

from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere
from constants import EPSILON, MIN_T
from utils import normalize

def is_occluded(hit_point, target_point, surfaces):
    """
    Check if a point is occluded from a target point by any surface.

    Args:
        hit_point: Starting point (numpy array)
        target_point: Target point to check visibility to (numpy array)
        surfaces: List of all surfaces in the scene

    Returns:
        bool: True if occluded, False if visible
    """
    to_target = target_point - hit_point
    distance = np.linalg.norm(to_target)
    direction = to_target / distance

    for surface in surfaces:
        t, _ = surface.intersect(hit_point, direction)
        if t is not None and MIN_T < t < distance:
            return True
    return False


def create_light_basis(to_light_direction):
    """
    Create an orthonormal basis for sampling a light plane perpendicular to the light direction.

    Args:
        to_light_direction: Normalized direction from hit point to light (numpy array)

    Returns:
        tuple: (right_vector, up_vector) - orthonormal basis vectors
    """
    # Choose an arbitrary perpendicular vector
    if abs(to_light_direction[0]) > 0.1:
        perpendicular = np.array([0, 1, 0])
    else:
        perpendicular = np.array([1, 0, 0])

    light_right = normalize(np.cross(to_light_direction, perpendicular))
    light_up = normalize(np.cross(light_right, to_light_direction))

    return light_right, light_up


def sample_light_point(light, light_right, light_up, cell_i, cell_j, num_shadow_rays):
    """
    Sample a random point on the light surface using jittered grid sampling.

    Args:
        light: Light object
        light_right: Right basis vector for light plane
        light_up: Up basis vector for light plane
        cell_i: Grid cell index in first dimension
        cell_j: Grid cell index in second dimension
        num_shadow_rays: Number of shadow rays per axis (N×N total grid)

    Returns:
        numpy array: Sampled point on light surface
    """
    # Grid cell position [0, 1] with random jitter
    u = (cell_i + np.random.random()) / num_shadow_rays
    v = (cell_j + np.random.random()) / num_shadow_rays

    # Map to disk: convert [0,1]×[0,1] to [-1,1]×[-1,1]
    u = 2 * u - 1
    v = 2 * v - 1

    # Sample point on light surface area
    # Scale the normalized coordinates by the light radius
    sample_offset = light_right * (u * light.radius) + light_up * (v * light.radius)
    return light.position + sample_offset


def generate_light_samples_vectorized(light, light_right, light_up, num_shadow_rays):
    """
    Generate all N×N shadow sample points at once using vectorized operations.

    Args:
        light: Light object
        light_right: Right basis vector for light plane
        light_up: Up basis vector for light plane
        num_shadow_rays: Number of shadow rays per axis (N×N total grid)

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


def compute_shadow_ray_ratio_vectorized(hit_point, light, surfaces, num_shadow_rays):
    """
    Vectorized shadow ray computation - processes all shadow rays simultaneously.

    Args:
        hit_point: Point on surface (numpy array)
        light: Light object
        surfaces: List of all surfaces in the scene
        num_shadow_rays: Number of shadow rays per axis (N×N total)

    Returns:
        float: Ratio in [0, 1] of rays that hit the light (0 = fully occluded, 1 = fully visible)
    """
    # Direction from hit point to light center
    to_light = light.position - hit_point
    to_light_dir = normalize(to_light)

    # For point lights (shadow_intensity = 0) or single ray, do simple test
    if light.shadow_intensity == 0 or num_shadow_rays == 1:
        if is_occluded(hit_point, light.position, surfaces):
            return 0.0
        else:
            return 1.0

    # Area light: Generate all shadow samples at once
    light_right, light_up = create_light_basis(to_light_dir)
    sample_points = generate_light_samples_vectorized(light, light_right, light_up, num_shadow_rays)

    total_samples = len(sample_points)

    # Prepare ray origins and directions for batch processing
    ray_origins = np.tile(hit_point, (total_samples, 1))  # (N, 3)
    ray_directions = sample_points - ray_origins  # (N, 3)
    distances = np.linalg.norm(ray_directions, axis=1)  # (N,)
    ray_directions = ray_directions / distances[:, np.newaxis]  # Normalize

    unoccluded_mask = np.ones(total_samples, dtype=bool)

    for surface in surfaces:
        t_values = surface.intersect_batch(ray_origins, ray_directions)
        blocking_mask = (t_values > EPSILON) & (t_values < distances)
        unoccluded_mask &= ~blocking_mask

    return np.sum(unoccluded_mask) / total_samples


def compute_shadow_ray_ratio(hit_point, light, surfaces, num_shadow_rays):
    """
    Compute the ratio of shadow rays that successfully reach the light.

    Args:
        hit_point: Point on surface (numpy array)
        light: Light object
        surfaces: List of all surfaces in the scene
        num_shadow_rays: Number of shadow rays per axis (N×N total)

    Returns:
        float: Ratio in [0, 1] of rays that hit the light (0 = fully occluded, 1 = fully visible)
    """
    # Direction from hit point to light center
    to_light = light.position - hit_point
    to_light_dir = normalize(to_light)

    # For point lights (shadow_intensity = 0) or single ray, do simple test
    if light.shadow_intensity == 0 or num_shadow_rays == 1:
        if is_occluded(hit_point, light.position, surfaces):
            return 0.0
        else:
            return 1.0

    # Area light: Sample multiple points on light surface
    light_right, light_up = create_light_basis(to_light_dir)

    hit_count = 0
    total_samples = num_shadow_rays * num_shadow_rays

    for i in range(num_shadow_rays):
        for j in range(num_shadow_rays):
            # Sample a point on the light surface
            sample_point = sample_light_point(light, light_right, light_up, i, j, num_shadow_rays)

            # Check if this sample point is visible
            if not is_occluded(hit_point, sample_point, surfaces):
                hit_count += 1

    return hit_count / total_samples


def calculate_light_intensity(hit_point, light, surfaces, num_shadow_rays):
    """
    Calculate light intensity for a point, accounting for shadows using the PDF formula.

    Formula from PDF page 6:
        light_intensity = (1 - shadow_intensity) * 1 + shadow_intensity * (% rays hit)

    This means:
    - shadow_intensity = 0: No shadows, always returns 1.0 (fully lit)
    - shadow_intensity = 1: Full shadows, returns the ray hit ratio [0, 1]
    - shadow_intensity = 0.5: Partial shadows, even fully occluded surfaces get 50% light

    Args:
        hit_point: Point on surface (numpy array)
        light: Light object
        surfaces: List of all surfaces in the scene
        num_shadow_rays: Number of shadow rays to cast (N×N grid)

    Returns:
        float: Light intensity in [0, 1] where 1.0 = fully lit, 0.0 = no light
    """
    # Compute what ratio of shadow rays successfully reach the light
    # Use vectorized version for better performance
    ray_hit_ratio = compute_shadow_ray_ratio_vectorized(hit_point, light, surfaces, num_shadow_rays)

    # Apply the shadow intensity formula from the PDF (page 6)
    # light_intensity = (1 - shadow_intensity) + shadow_intensity * ray_hit_ratio
    light_intensity = (1.0 - light.shadow_intensity) + light.shadow_intensity * ray_hit_ratio

    return light_intensity


def calculate_reflection_direction(incident_direction, normal):
    """
    Calculate the reflection direction for a ray bouncing off a surface.

    Uses the formula: R = D - 2(D·N)N
    where:
    - D = incident ray direction (normalized, pointing TOWARDS surface)
    - N = surface normal (normalized, pointing AWAY from surface)
    - R = reflected direction (points AWAY from surface)

    Physical intuition:
    - The component of D parallel to the surface stays the same
    - The component perpendicular to the surface reverses

    Args:
        incident_direction: Direction of incoming ray (normalized numpy array)
        normal: Surface normal (normalized numpy array)

    Returns:
        numpy array: Reflected direction (normalized)
    """
    # R = D - 2(D·N)N
    # (D·N) is the projection of D onto N (how much D points into the surface)
    # 2(D·N)N is twice that projection - this "flips" the perpendicular component
    return incident_direction - 2 * np.dot(incident_direction, normal) * normal


def find_nearest_intersection(ray_origin, ray_direction, surfaces, min_t=MIN_T):
    """
    Find the nearest surface intersection along a ray.

    Args:
        ray_origin: Origin of the ray (numpy array)
        ray_direction: Direction of the ray (normalized numpy array)
        surfaces: List of surface objects
        min_t: Minimum distance to consider (prevents self-intersection)

    Returns:
        tuple: (t, hit_point, normal, surface) if hit, else (None, None, None, None)
    """
    nearest_t = float('inf')
    nearest_normal = None
    nearest_surface = None

    for surface in surfaces:
        t, normal = surface.intersect(ray_origin, ray_direction)
        if t is not None and t > min_t and t < nearest_t:
            nearest_t = t
            nearest_normal = normal
            nearest_surface = surface

    if nearest_surface is None:
        return None, None, None, None

    hit_point = ray_origin + nearest_t * ray_direction
    return nearest_t, hit_point, nearest_normal, nearest_surface


def calculate_phong_shading(hit_point, normal, view_dir, material, lights, surfaces, num_shadow_rays):
    """
    Calculate Phong shading (diffuse + specular) for a point with shadows.

    Phase 4: Includes soft shadows via shadow ray sampling.

    Args:
        hit_point: Point on surface (numpy array)
        normal: Surface normal at hit point (normalized)
        view_dir: Direction from hit point to viewer (normalized)
        material: Material object of the surface
        lights: List of light sources
        surfaces: List of all surfaces (for shadow ray testing)
        num_shadow_rays: Number of shadow rays per axis (N×N total samples)

    Returns:
        numpy array: RGB color [0, 1]
    """
    color = np.zeros(3)

    for light in lights:
        # Direction to light
        light_direction = light.position - hit_point
        light_direction = normalize(light_direction)

        # Calculate light intensity accounting for shadows (0 = no light, 1 = fully lit)
        light_intensity = calculate_light_intensity(hit_point, light, surfaces, num_shadow_rays)

        # Skip this light if no light reaches the surface
        if light_intensity == 0:
            continue

        # Diffuse component: Kd * I * max(0, N·L)
        diffuse_factor = max(0, np.dot(normal, light_direction))
        diffuse_contribution = (np.array(material.diffuse_color) *
                               np.array(light.color) *
                               diffuse_factor)

        # Specular component: Ks * I * max(0, R·V)^α
        # Reflection direction: R = 2N(N·L) - L
        reflection_dir = 2 * np.dot(light_direction, normal) * normal - light_direction
        specular_factor = max(0, np.dot(reflection_dir, view_dir)) ** material.shininess
        specular_contribution = (np.array(material.specular_color) *
                                np.array(light.color) *
                                specular_factor *
                                light.specular_intensity)

        # Apply light intensity (which accounts for shadows) to light contribution
        color += (diffuse_contribution + specular_contribution) * light_intensity

    return color


def calculate_reflection_contribution(hit_point, ray_direction, normal, material, surfaces,
                                      materials, lights, scene_settings, num_shadow_rays, current_depth):
    """
    Calculate the color contribution from reflections.

    Casts a reflection ray and recursively traces it to find what's reflected.

    Args:
        hit_point: Point on surface where ray hit (numpy array)
        ray_direction: Direction of incident ray (numpy array)
        normal: Surface normal at hit point (numpy array)
        material: Material of the surface
        surfaces: List of all surfaces in scene
        materials: List of all materials
        lights: List of all lights
        scene_settings: SceneSettings object
        num_shadow_rays: Number of shadow rays for soft shadows
        current_depth: Current recursion depth

    Returns:
        numpy array: RGB reflection contribution [0, 1]
    """
    # Check if material has any reflectivity
    if not np.any(np.array(material.reflection_color) > 0):
        return np.zeros(3)

    # Calculate the reflection direction: R = D - 2(D·N)N
    reflection_dir = calculate_reflection_direction(ray_direction, normal)

    # Start slightly offset from surface to avoid self-intersection
    reflection_origin = hit_point + normal * MIN_T

    # Recursively trace the reflection ray
    reflected_scene_color = trace_ray(
        reflection_origin,
        reflection_dir,
        surfaces,
        materials,
        lights,
        scene_settings,
        num_shadow_rays,
        current_depth + 1
    )

    # Multiply by material's reflection color (for colored reflections)
    return reflected_scene_color * np.array(material.reflection_color)


def calculate_transparency_contribution(hit_point, ray_direction, material, surfaces,
                                       materials, lights, scene_settings, num_shadow_rays, current_depth):
    """
    Calculate the color contribution from transparency.

    Casts a ray straight through the surface to find what's behind it.

    Args:
        hit_point: Point on surface where ray hit (numpy array)
        ray_direction: Direction of incident ray (numpy array)
        material: Material of the surface
        surfaces: List of all surfaces in scene
        materials: List of all materials
        lights: List of all lights
        scene_settings: SceneSettings object
        num_shadow_rays: Number of shadow rays for soft shadows
        current_depth: Current recursion depth

    Returns:
        numpy array: RGB transparency contribution [0, 1]
    """
    # Check if material has any transparency
    if material.transparency <= 0:
        return np.zeros(3)

    # Continue ray in SAME direction (straight through)
    # Offset along ray direction to avoid self-intersection
    transparency_origin = hit_point + ray_direction * MIN_T

    # Recursively trace to find what's behind this surface
    background_scene_color = trace_ray(
        transparency_origin,
        ray_direction,  # Same direction - ray continues straight
        surfaces,
        materials,
        lights,
        scene_settings,
        num_shadow_rays,
        current_depth + 1
    )

    # Scale by material's transparency value
    return background_scene_color * material.transparency


def combine_color_components(local_color, reflection_color, transparency_color, material):
    """
    Combine local, reflection, and transparency colors using the PDF formula.

    Formula (PDF page 5):
        output = (background) · transparency
               + (diffuse + specular) · (1 − transparency)
               + (reflection)

    Args:
        local_color: Diffuse + specular from Phong shading (numpy array)
        reflection_color: Color from reflections (numpy array)
        transparency_color: Color from transparency (background) (numpy array)
        material: Material object

    Returns:
        numpy array: Final combined RGB color [0, 1]
    """
    return (
        transparency_color +                           # background · transparency
        local_color * (1.0 - material.transparency) +  # local · (1 - transparency)
        reflection_color                               # reflection (independent!)
    )


def trace_ray(ray_origin, ray_direction, surfaces, materials, lights, scene_settings,
              num_shadow_rays, current_depth=0):
    """
    Recursively trace a ray through the scene, handling reflections and transparency.

    This is the core recursive function that implements the full ray tracing algorithm.

    Formula from PDF (page 5):
        output_color = (background_color) · transparency
                     + (diffuse + specular) · (1 − transparency)
                     + (reflection_color)

    Args:
        ray_origin: Origin point of the ray (numpy array)
        ray_direction: Direction of the ray (normalized numpy array)
        surfaces: List of all surfaces in the scene
        materials: List of Material objects
        lights: List of Light objects
        scene_settings: SceneSettings object (contains background_color, max_recursions)
        num_shadow_rays: Number of shadow rays for soft shadows
        current_depth: Current recursion depth (starts at 0)

    Returns:
        numpy array: RGB color [0, 1] for this ray
    """
    # BASE CASE 1: Maximum recursion depth reached
    # Return background color to stop infinite recursion
    if current_depth >= scene_settings.max_recursions:
        return np.array(scene_settings.background_color)

    # Find the nearest surface intersection
    _, hit_point, normal, surface = find_nearest_intersection(
        ray_origin, ray_direction, surfaces
    )

    # BASE CASE 2: Ray doesn't hit anything
    # Return background color
    if surface is None:
        return np.array(scene_settings.background_color)

    # RECURSIVE CASE: Ray hit a surface
    # TODO: verify material index is valid
    material = materials[surface.material_index - 1]

    # ===== STEP 1: Calculate LOCAL color (diffuse + specular with shadows) =====
    view_dir = normalize(ray_origin - hit_point)
    local_color = calculate_phong_shading(
        hit_point, normal, view_dir, material, lights, surfaces, num_shadow_rays
    )

    # ===== STEP 2: Calculate REFLECTION contribution =====
    reflection_color = calculate_reflection_contribution(
        hit_point, ray_direction, normal, material, surfaces, materials,
        lights, scene_settings, num_shadow_rays, current_depth
    )

    # ===== STEP 3: Calculate TRANSPARENCY contribution =====
    transparency_color = calculate_transparency_contribution(
        hit_point, ray_direction, material, surfaces, materials,
        lights, scene_settings, num_shadow_rays, current_depth
    )

    # ===== STEP 4: Combine all components using PDF formula =====
    final_color = combine_color_components(
        local_color, reflection_color, transparency_color, material
    )

    return final_color


def render(camera, scene_settings, materials, surfaces, lights, width, height):
    """
    Render the scene with recursive ray tracing (reflections, transparency, soft shadows).

    This is the main rendering loop that shoots a ray through each pixel and
    uses recursive ray tracing to calculate the final color.

    Args:
        camera: Camera object
        scene_settings: SceneSettings object
        materials: List of Material objects
        surfaces: List of surface objects (spheres, planes, cubes)
        lights: List of Light objects
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        numpy array: Image array (height, width, 3) with values [0, 255]
    """
    start_time = time.time()

    image = np.zeros((height, width, 3))
    num_shadow_rays = int(scene_settings.root_number_shadow_rays)

    total_pixels = width * height
    pixels_rendered = 0

    for y in range(height):
        for x in range(width):
            # Generate ray for this pixel (from camera through pixel)
            ray_origin, ray_direction = camera.get_ray(x, y, width, height)

            # Recursively trace this ray through the scene
            # This handles: intersection, shading, reflections, transparency
            color = trace_ray(
                ray_origin,
                ray_direction,
                surfaces,
                materials,
                lights,
                scene_settings,
                num_shadow_rays,
                current_depth=0  # Start at depth 0
            )

            # Clamp color to [0, 1] and convert to [0, 255]
            color = np.clip(color, 0, 1)
            image[y, x] = color * 255

            pixels_rendered += 1
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

    print()

    elapsed_time = time.time() - start_time
    print(f"  Render time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

    return image


def parse_scene_file(file_path):
    objects = []
    camera = None
    scene_settings = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            obj_type = parts[0]
            params = [float(p) for p in parts[1:]]
            if obj_type == "cam":
                camera = Camera(params[:3], params[3:6], params[6:9], params[9], params[10])
            elif obj_type == "set":
                scene_settings = SceneSettings(params[:3], params[3], params[4])
            elif obj_type == "mtl":
                material = Material(params[:3], params[3:6], params[6:9], params[9], params[10])
                objects.append(material)
            elif obj_type == "sph":
                sphere = Sphere(params[:3], params[3], int(params[4]))
                objects.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(params[:3], params[3], int(params[4]))
                objects.append(plane)
            elif obj_type == "box":
                cube = Cube(params[:3], params[3], int(params[4]))
                objects.append(cube)
            elif obj_type == "lgt":
                light = Light(params[:3], params[3:6], params[6], params[7], params[8])
                objects.append(light)
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
    return camera, scene_settings, objects


def save_image(image_array):
    image = Image.fromarray(np.uint8(image_array))

    # Save the image to a file
    image.save("scenes/Spheres.png")


def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)

    # Separate objects by type
    materials = [obj for obj in objects if isinstance(obj, Material)]
    surfaces = [obj for obj in objects if isinstance(obj, (Sphere, InfinitePlane, Cube))]
    lights = [obj for obj in objects if isinstance(obj, Light)]

    print(f"Scene loaded:")
    print(f"  Materials: {len(materials)}")
    print(f"  Surfaces: {len(surfaces)}")
    print(f"  Lights: {len(lights)}")
    print(f"Rendering {args.width}x{args.height} image...")

    # Render the scene
    image_array = render(camera, scene_settings, materials, surfaces, lights,
                        args.width, args.height)

    # Save the output image
    image = Image.fromarray(np.uint8(image_array))
    image.save(args.output_image)
    print(f"Image saved to {args.output_image}")


if __name__ == '__main__':
    main()
