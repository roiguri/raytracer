import numpy as np
from utils import EPSILON


class SceneSettings:
    def __init__(self, background_color, root_number_shadow_rays, max_recursions,
                 camera=None, surfaces=None, materials=None, lights=None):
        # Rendering settings
        self.background_color = background_color
        self.root_number_shadow_rays = root_number_shadow_rays
        self.max_recursions = max_recursions

        # Scene objects
        self.camera = camera
        self.surfaces = surfaces if surfaces is not None else []
        self.materials = materials if materials is not None else []
        self.lights = lights if lights is not None else []

    def find_nearest_intersection(self, ray_origin, ray_direction, min_t=EPSILON):
        """
        Find the nearest surface intersection along a ray.

        Args:
            ray_origin: Origin of the ray (numpy array)
            ray_direction: Direction of the ray (normalized numpy array)
            min_t: Minimum distance to consider (prevents self-intersection)

        Returns:
            tuple: (t, hit_point, normal, surface) if hit, else (None, None, None, None)
        """
        nearest_t = float('inf')
        nearest_normal = None
        nearest_surface = None

        for surface in self.surfaces:
            t, normal = surface.intersect(ray_origin, ray_direction)
            if t is not None and t > min_t and t < nearest_t:
                nearest_t = t
                nearest_normal = normal
                nearest_surface = surface

        if nearest_surface is None:
            return None, None, None, None

        hit_point = ray_origin + nearest_t * ray_direction
        return nearest_t, hit_point, nearest_normal, nearest_surface

    def is_occluded(self, hit_point, target_point):
        """
        Check if a point is occluded from a target point by any surface.

        Args:
            hit_point: Starting point (numpy array)
            target_point: Target point to check visibility to (numpy array)

        Returns:
            bool: True if occluded, False if visible
        """
        to_target = target_point - hit_point
        distance = np.linalg.norm(to_target)
        direction = to_target / distance

        for surface in self.surfaces:
            t, _ = surface.intersect(hit_point, direction)
            if t is not None and EPSILON < t < distance:
                return True
        return False
