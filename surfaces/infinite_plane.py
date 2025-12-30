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

    def intersect_batch(self, ray_origins, ray_directions):
        """
        Test multiple rays against this plane simultaneously using vectorization.

        Uses NumPy broadcasting to test N rays in parallel, significantly faster
        than testing each ray individually in a loop.

        Args:
            ray_origins: np.ndarray, shape (N, 3) - origins of N rays
            ray_directions: np.ndarray, shape (N, 3) - directions of N rays (normalized)

        Returns:
            t_values: np.ndarray, shape (N,) - distance to nearest intersection for each ray
                      np.inf where no valid intersection occurs
        """
        # Calculate D·N for all rays at once (N,)
        denom = np.sum(ray_directions * self.normal, axis=1)

        # Calculate O·N for all rays at once (N,)
        origin_dot_normal = np.sum(ray_origins * self.normal, axis=1)

        # Calculate t for all rays: t = (offset - O·N) / (D·N)
        # Avoid division by zero by using np.where
        # Where denom is too small (parallel), set t to infinity (no intersection)
        t_values = np.where(
            np.abs(denom) >= EPSILON,
            (self.offset - origin_dot_normal) / denom,
            np.inf
        )

        # Only keep positive t values (intersections in front of rays)
        # Use EPSILON threshold to avoid self-intersection artifacts
        # This matches the non-batch version logic at line 41 (t > 0)
        t_values = np.where(t_values > EPSILON, t_values, np.inf)

        return t_values
