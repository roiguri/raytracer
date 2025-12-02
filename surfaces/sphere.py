import numpy as np


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
