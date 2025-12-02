import numpy as np
from utils import normalize


class Camera:
    def __init__(self, position, look_at, up_vector, screen_distance, screen_width):
        self.position = np.array(position)
        self.look_at = np.array(look_at)
        self.up_vector = np.array(up_vector)
        self.screen_distance = screen_distance
        self.screen_width = screen_width

        # Calculate camera coordinate system
        self.forward = normalize(self.look_at - self.position)
        self.right = normalize(np.cross(self.forward, self.up_vector))
        self.up = normalize(np.cross(self.right, self.forward))

    def get_ray(self, pixel_x, pixel_y, image_width, image_height):
        """
        Generate a ray from the camera through a pixel.

        Args:
            pixel_x: x coordinate of the pixel (0 to image_width-1)
            pixel_y: y coordinate of the pixel (0 to image_height-1)
            image_width: width of the image in pixels
            image_height: height of the image in pixels

        Returns:
            tuple: (ray_origin, ray_direction) both as numpy arrays
        """
        # Calculate screen height based on aspect ratio
        screen_height = self.screen_width * (image_height / image_width)

        # Normalize pixel coordinates to [-0.5, 0.5] range
        norm_x = (pixel_x + 0.5) / image_width - 0.5
        norm_y = (pixel_y + 0.5) / image_height - 0.5

        # Calculate point on screen plane
        screen_center = self.position + self.forward * self.screen_distance
        screen_point = (screen_center +
                       self.right * (norm_x * self.screen_width) +
                       self.up * (-norm_y * screen_height))

        # Calculate ray direction
        ray_direction = normalize(screen_point - self.position)

        return self.position, ray_direction
