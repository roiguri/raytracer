import argparse
from PIL import Image
import numpy as np

from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere
from constants import EPSILON, MIN_T
from utils import normalize


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


def calculate_phong_shading(hit_point, normal, view_dir, material, lights):
    """
    Calculate Phong shading (diffuse + specular) for a point.

    Phase 3: No shadows yet - assume all lights reach all points.

    Args:
        hit_point: Point on surface (numpy array)
        normal: Surface normal at hit point (normalized)
        view_dir: Direction from hit point to viewer (normalized)
        material: Material object of the surface
        lights: List of light sources

    Returns:
        numpy array: RGB color [0, 1]
    """
    color = np.zeros(3)

    for light in lights:
        # Direction to light
        light_direction = light.position - hit_point
        light_direction = normalize(light_direction)

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

        color += diffuse_contribution + specular_contribution

    return color


def render(camera, scene_settings, materials, surfaces, lights, width, height):
    """
    Render the scene.

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
    image = np.zeros((height, width, 3))

    for y in range(height):
        for x in range(width):
            # Generate ray for this pixel
            ray_origin, ray_direction = camera.get_ray(x, y, width, height)

            # Find nearest intersection
            t, hit_point, normal, surface = find_nearest_intersection(
                ray_origin, ray_direction, surfaces
            )

            if surface is None:
                # No intersection - use background color
                color = np.array(scene_settings.background_color)
            else:
                # Get material for the surface
                material = materials[surface.material_index - 1]  # Material indices start at 1

                # View direction (from hit point to camera)
                view_dir = normalize(ray_origin - hit_point)

                # Calculate Phong shading (no shadows yet)
                color = calculate_phong_shading(hit_point, normal, view_dir, material, lights)

            # Clamp color to [0, 1] and convert to [0, 255]
            color = np.clip(color, 0, 1)
            image[y, x] = color * 255

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
