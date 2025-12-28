import numpy as np
from constants import EPSILON


class Sphere:
    def __init__(self, position, radius, material_index):
        self.position = np.array(position)
        self.radius = radius
        self.material_index = material_index

    def intersect(self, ray_origin, ray_direction):
        """
        Calculate ray-sphere intersection using quadratic equation.

        Ray: P(t) = O + t*D
        Sphere: |P - C|² = r²

        Substituting: |O + t*D - C|² = r²
        Expands to: at² + bt + c = 0
        where:
            a = D·D
            b = 2*D·(O-C)
            c = (O-C)·(O-C) - r²

        Args:
            ray_origin: Origin of the ray (numpy array)
            ray_direction: Direction of the ray (normalized numpy array)

        Returns:
            tuple: (t, normal) if intersection exists, (None, None) otherwise
        """
        # Vector from sphere center to ray origin
        oc = ray_origin - self.position

        # Quadratic equation coefficients
        a = np.dot(ray_direction, ray_direction)
        b = 2.0 * np.dot(ray_direction, oc)
        c = np.dot(oc, oc) - self.radius * self.radius

        # Calculate discriminant: b² - 4ac
        discriminant = b * b - 4 * a * c

        # No intersection if discriminant is negative
        if discriminant < 0:
            return None, None

        # Calculate both solutions using quadratic formula
        sqrt_discriminant = np.sqrt(discriminant)
        t1 = (-b - sqrt_discriminant) / (2.0 * a)
        t2 = (-b + sqrt_discriminant) / (2.0 * a)

        # Choose the closest positive t (in front of camera)
        t = None
        if t1 > 0:
            t = t1  # Closer intersection
        elif t2 > 0:
            t = t2  # Farther intersection (ray starts inside sphere)
        else:
            return None, None  # Both intersections behind camera

        # Calculate hit point and surface normal
        hit_point = ray_origin + t * ray_direction
        normal = (hit_point - self.position) / self.radius

        return t, normal

    def intersect_batch(self, ray_origins, ray_directions):
        """
        Test multiple rays against this sphere simultaneously using vectorization.

        Uses NumPy broadcasting to test N rays in parallel, significantly faster
        than testing each ray individually in a loop.

        Args:
            ray_origins: np.ndarray, shape (N, 3) - origins of N rays
            ray_directions: np.ndarray, shape (N, 3) - directions of N rays (normalized)

        Returns:
            t_values: np.ndarray, shape (N,) - distance to nearest intersection for each ray
                      np.inf where no valid intersection occurs
        """
        # Vector from sphere center to each ray origin (N, 3)
        oc = ray_origins - self.position

        # Quadratic coefficients for all rays at once
        # a = D·D for each ray (should be ~1.0 if directions are normalized)
        a = np.sum(ray_directions ** 2, axis=1)  # (N,)

        # b = 2*D·(O-C) for each ray
        b = 2.0 * np.sum(ray_directions * oc, axis=1)  # (N,)

        # c = (O-C)·(O-C) - r² for each ray
        c = np.sum(oc ** 2, axis=1) - self.radius ** 2  # (N,)

        # Calculate discriminant: b² - 4ac
        discriminant = b**2 - 4*a*c  # (N,)

        # Initialize all results as no intersection (inf)
        t_values = np.full(len(ray_origins), np.inf)

        # Only process rays with valid intersections (discriminant >= 0)
        valid_mask = discriminant >= 0

        if np.any(valid_mask):
            # Extract coefficients for valid rays
            sqrt_disc = np.sqrt(discriminant[valid_mask])
            a_valid = a[valid_mask]
            b_valid = b[valid_mask]

            # Calculate both solutions using quadratic formula
            t1 = (-b_valid - sqrt_disc) / (2.0 * a_valid)
            t2 = (-b_valid + sqrt_disc) / (2.0 * a_valid)

            # Choose nearest positive t for each ray
            # If t1 > EPSILON, use t1 (closer intersection)
            # Else if t2 > EPSILON, use t2 (farther intersection, ray inside sphere)
            # Else no valid intersection (both behind ray or too close)
            t_near = np.where(t1 > EPSILON, t1, t2)

            # Only keep intersections that are in front of the ray
            t_values[valid_mask] = np.where(t_near > EPSILON, t_near, np.inf)

        return t_values
