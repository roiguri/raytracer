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
from utils import normalize, EPSILON

def compute_shadow_ray_ratio_vectorized(hit_point, light, scene_settings, num_shadow_rays):
    """
    Vectorized shadow ray computation with transparency support.

    Tests ALL shadow rays against ALL surfaces in batches for massive speedup.
    Accumulates transparency through multiple surfaces.

    Args:
        hit_point: Point on surface (numpy array)
        light: Light object
        scene_settings: SceneSettings object containing surfaces and materials
        num_shadow_rays: Number of shadow rays per axis (N×N total)

    Returns:
        float: Ratio in [0, 1] of light transmission (average of all rays)
    """
    # Direction from hit point to light center
    to_light = light.position - hit_point
    to_light_dir = normalize(to_light)

    # For point lights (shadow_intensity = 0) or single ray, use simple test with transparency
    if light.shadow_intensity == 0 or num_shadow_rays == 1:
        light_factor = 1.0
        distance_to_light = np.linalg.norm(to_light)
        direction_to_light = to_light_dir

        for surface in scene_settings.surfaces:
            t, _ = surface.intersect(hit_point, direction_to_light)
            if t is not None and EPSILON < t < distance_to_light:
                material = scene_settings.materials[surface.material_index - 1]
                light_factor *= material.transparency
                if light_factor == 0.0:
                    break
        return light_factor

    # Area light: Generate all shadow samples at once
    light_right, light_up = light.create_basis(to_light_dir)
    sample_points = light.generate_samples(light_right, light_up, num_shadow_rays)

    total_samples = len(sample_points)

    # Prepare ray origins and directions for batch processing
    ray_origins = np.tile(hit_point, (total_samples, 1))  # (N, 3)
    ray_directions = sample_points - ray_origins  # (N, 3)
    distances = np.linalg.norm(ray_directions, axis=1)  # (N,)
    ray_directions = ray_directions / distances[:, np.newaxis]  # Normalize

    # Initialize light transmission factors (1.0 = full light, 0.0 = blocked)
    light_factors = np.ones(total_samples, dtype=np.float64)

    # Test each surface against all shadow rays
    for surface in scene_settings.surfaces:
        # OPTIMIZATION: Early exit if all rays are fully blocked
        if np.all(light_factors == 0.0):
            return 0.0

        # Get intersection distances for all rays
        t_values = surface.intersect_batch(ray_origins, ray_directions)

        # Create mask: rays that hit this surface before reaching the light
        blocking_mask = (t_values > EPSILON) & (t_values < distances)

        # Accumulate transparency for blocked rays
        if np.any(blocking_mask):
            material = scene_settings.materials[surface.material_index - 1]

            # OPTIMIZATION: Early exit if material is fully opaque
            if material.transparency == 0.0:
                light_factors[blocking_mask] = 0.0
            else:
                light_factors[blocking_mask] *= material.transparency

    # Return average light transmission across all shadow rays
    return np.mean(light_factors)


def calculate_light_intensity(hit_point, light, scene_settings, num_shadow_rays):
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
        scene_settings: SceneSettings object containing surfaces and materials
        num_shadow_rays: Number of shadow rays to cast (N×N grid)

    Returns:
        float: Light intensity in [0, 1] where 1.0 = fully lit, 0.0 = no light
    """
    # Compute what ratio of shadow rays successfully reach the light
    # Use vectorized version with transparency support for better performance
    ray_hit_ratio = compute_shadow_ray_ratio_vectorized(hit_point, light, scene_settings, num_shadow_rays)

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


def calculate_phong_shading(hit_point, normal, view_dir, material, scene_settings, num_shadow_rays):
    """
    Calculate Phong shading (diffuse + specular) for a point with shadows and transparency.

    Includes soft shadows via shadow ray sampling and transparent shadow support.

    Args:
        hit_point: Point on surface (numpy array)
        normal: Surface normal at hit point (normalized)
        view_dir: Direction from hit point to viewer (normalized)
        material: Material object of the surface
        scene_settings: SceneSettings object containing lights, surfaces, and materials
        num_shadow_rays: Number of shadow rays per axis (N×N total samples)

    Returns:
        numpy array: RGB color [0, 1]
    """
    color = np.zeros(3)

    for light in scene_settings.lights:
        # Direction to light
        light_direction = light.position - hit_point
        light_direction = normalize(light_direction)

        # Calculate light intensity accounting for shadows and transparency
        light_intensity = calculate_light_intensity(hit_point, light, scene_settings, num_shadow_rays)

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


def calculate_reflection_contribution(hit_point, ray_direction, normal, material,
                                      scene_settings, num_shadow_rays, current_depth):
    """
    Calculate the color contribution from reflections.

    Casts a reflection ray and recursively traces it to find what's reflected.

    Args:
        hit_point: Point on surface where ray hit (numpy array)
        ray_direction: Direction of incident ray (numpy array)
        normal: Surface normal at hit point (numpy array)
        material: Material of the surface
        scene_settings: SceneSettings object containing all scene data
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
    reflection_origin = hit_point + normal * EPSILON

    # Recursively trace the reflection ray
    reflected_scene_color = trace_ray(
        reflection_origin,
        reflection_dir,
        scene_settings,
        num_shadow_rays,
        current_depth + 1
    )

    # Multiply by material's reflection color (for colored reflections)
    return reflected_scene_color * np.array(material.reflection_color)


def calculate_transparency_contribution(hit_point, ray_direction, material,
                                       scene_settings, num_shadow_rays, current_depth):
    """
    Calculate the color contribution from transparency.

    Casts a ray straight through the surface to find what's behind it.

    Args:
        hit_point: Point on surface where ray hit (numpy array)
        ray_direction: Direction of incident ray (numpy array)
        material: Material of the surface
        scene_settings: SceneSettings object containing all scene data
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
    transparency_origin = hit_point + ray_direction * EPSILON

    # Recursively trace to find what's behind this surface
    background_scene_color = trace_ray(
        transparency_origin,
        ray_direction,  # Same direction - ray continues straight
        scene_settings,
        num_shadow_rays,
        current_depth + 1
    )

    # Scale by material's transparency value
    return background_scene_color * material.transparency


def trace_ray(ray_origin, ray_direction, scene_settings, num_shadow_rays, current_depth=0):
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
        scene_settings: SceneSettings object containing all scene data
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
    _, hit_point, normal, surface = scene_settings.find_nearest_intersection(
        ray_origin, ray_direction
    )

    # BASE CASE 2: Ray doesn't hit anything
    # Return background color
    if surface is None:
        return np.array(scene_settings.background_color)

    # RECURSIVE CASE: Ray hit a surface
    # TODO: verify material index is valid
    material = scene_settings.materials[surface.material_index - 1]

    # ===== STEP 1: Calculate LOCAL color (diffuse + specular with shadows) =====
    view_dir = normalize(ray_origin - hit_point)
    local_color = calculate_phong_shading(
        hit_point, normal, view_dir, material, scene_settings, num_shadow_rays
    )

    # ===== STEP 2: Calculate REFLECTION contribution =====
    reflection_color = calculate_reflection_contribution(
        hit_point, ray_direction, normal, material,
        scene_settings, num_shadow_rays, current_depth
    )

    # ===== STEP 3: Calculate TRANSPARENCY contribution =====
    transparency_color = calculate_transparency_contribution(
        hit_point, ray_direction, material,
        scene_settings, num_shadow_rays, current_depth
    )

    # ===== STEP 4: Combine all components using PDF formula =====
    final_color = material.blend_colors(
        local_color, reflection_color, transparency_color
    )

    return final_color


def render(scene_settings, width, height):
    """
    Render the scene with recursive ray tracing (reflections, transparency, soft shadows).

    This is the main rendering loop that shoots a ray through each pixel and
    uses recursive ray tracing to calculate the final color.

    Args:
        scene_settings: SceneSettings object containing camera, surfaces, materials, lights, and settings
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
            ray_origin, ray_direction = scene_settings.camera.get_ray(x, y, width, height)

            # Recursively trace this ray through the scene
            # This handles: intersection, shading, reflections, transparency
            color = trace_ray(
                ray_origin,
                ray_direction,
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
    background_color = None
    root_number_shadow_rays = None
    max_recursions = None

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
                background_color = params[:3]
                root_number_shadow_rays = params[3]
                max_recursions = params[4]
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

    # Separate objects by type
    materials = [obj for obj in objects if isinstance(obj, Material)]
    surfaces = [obj for obj in objects if isinstance(obj, (Sphere, InfinitePlane, Cube))]
    lights = [obj for obj in objects if isinstance(obj, Light)]

    # Create SceneSettings with all scene objects
    scene_settings = SceneSettings(
        background_color, root_number_shadow_rays, max_recursions,
        camera, surfaces, materials, lights
    )

    return scene_settings


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
    scene_settings = parse_scene_file(args.scene_file)

    print(f"Scene loaded:")
    print(f"  Materials: {len(scene_settings.materials)}")
    print(f"  Surfaces: {len(scene_settings.surfaces)}")
    print(f"  Lights: {len(scene_settings.lights)}")
    print(f"Rendering {args.width}x{args.height} image...")

    # Render the scene
    image_array = render(scene_settings, args.width, args.height)

    # Save the output image
    image = Image.fromarray(np.uint8(image_array))
    image.save(args.output_image)
    print(f"Image saved to {args.output_image}")


if __name__ == '__main__':
    main()
