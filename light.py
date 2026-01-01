import numpy as np
from utils import normalize


class Light:
    def __init__(self, position, color, specular_intensity, shadow_intensity, radius):
        self.position = position
        self.color = color
        self.specular_intensity = specular_intensity
        self.shadow_intensity = shadow_intensity
        self.radius = radius

    def create_basis(self, to_light_direction):
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


    def generate_samples(self, light_right, light_up, num_shadow_rays):
        """
        Generate all N×N shadow sample points at once using vectorized operations.

        Args:
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
                u = (i + np.random.random()) / num_shadow_rays
                v = (j + np.random.random()) / num_shadow_rays

                # Shift to center: [0, 1] -> [-0.5, 0.5]
                u = u - 0.5
                v = v - 0.5

                # Sample point on light
                offset = light_right * (u * self.radius) + light_up * (v * self.radius)
                samples[idx] = self.position + offset
                idx += 1

        return samples
