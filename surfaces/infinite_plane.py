import numpy as np
from constants import EPSILON


class InfinitePlane:
    def __init__(self, normal, offset, material_index):
        self.normal = np.array(normal)
        self.normal = self.normal / np.linalg.norm(self.normal)  # Normalize
        self.offset = offset
        self.material_index = material_index

    def intersect(self, ray_origin, ray_direction):
        """
        Calculate ray-plane intersection.

        Plane equation: P · N = offset
        Ray equation: P = O + t*D

        Substituting: (O + t*D) · N = offset
        Expanding: O·N + t(D·N) = offset
        Solving for t: t = (offset - O·N) / (D·N)

        Args:
            ray_origin: Origin of the ray (numpy array)
            ray_direction: Direction of the ray (normalized numpy array)

        Returns:
            tuple: (t, normal) if intersection exists, (None, None) otherwise
        """
        # Calculate denominator D·N
        denom = np.dot(ray_direction, self.normal)

        # Check if ray is parallel to plane (denominator near zero)
        if abs(denom) < EPSILON:
            return None, None  # Ray parallel to plane, no intersection

        # Calculate t
        t = (self.offset - np.dot(ray_origin, self.normal)) / denom

        # Only return intersection if t is positive (in front of camera)
        if t > 0:
            return t, self.normal
        else:
            return None, None  # Intersection behind camera
