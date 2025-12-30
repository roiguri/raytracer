import numpy as np
from constants import EPSILON


class Cube:
    def __init__(self, position, scale, material_index):
        self.position = np.array(position)
        self.scale = scale
        self.material_index = material_index

    def intersect(self, ray_origin, ray_direction):
        """
        Calculate ray-cube intersection using the slab method.

        The cube is axis-aligned (no rotations). The slab method treats
        the cube as the intersection of 3 pairs of parallel planes (slabs):
        - X slab: planes at x_min and x_max
        - Y slab: planes at y_min and y_max
        - Z slab: planes at z_min and z_max

        For each slab, we calculate where the ray enters and exits.
        The ray hits the cube if all three slab intervals overlap.

        Args:
            ray_origin: Origin of the ray (numpy array)
            ray_direction: Direction of the ray (normalized numpy array)

        Returns:
            tuple: (t, normal) if intersection exists, (None, None) otherwise
        """
        # Calculate min and max corners of the cube
        half_scale = self.scale / 2.0
        box_min = self.position - half_scale
        box_max = self.position + half_scale

        # Initialize near and far intersection distances
        t_near = float('-inf')  # Largest entry point
        t_far = float('inf')    # Smallest exit point

        # Track which axis gave us t_near (for normal calculation)
        normal_axis = 0
        normal_sign = 1

        # Check intersection with slabs for each axis (x, y, z)
        for axis in range(3):
            if abs(ray_direction[axis]) < EPSILON:
                # Ray is parallel to this slab
                # Check if ray origin is within the slab bounds
                if ray_origin[axis] < box_min[axis] or ray_origin[axis] > box_max[axis]:
                    return None, None  # Ray is outside slab, no intersection
            else:
                # Calculate intersection distances for this axis's two planes
                t1 = (box_min[axis] - ray_origin[axis]) / ray_direction[axis]
                t2 = (box_max[axis] - ray_origin[axis]) / ray_direction[axis]

                # Ensure t1 is near and t2 is far
                if t1 > t2:
                    t1, t2 = t2, t1

                # Update overall near and far intersections
                if t1 > t_near:
                    t_near = t1
                    normal_axis = axis
                    # Determine which direction we're entering from
                    normal_sign = -1 if ray_direction[axis] > 0 else 1

                if t2 < t_far:
                    t_far = t2

                # If t_near > t_far, ray misses the box
                if t_near > t_far:
                    return None, None

        # If t_far < 0, box is entirely behind the ray
        if t_far < 0:
            return None, None

        # Use t_near if positive, otherwise t_far (ray starts inside box)
        if t_near > 0:
            t = t_near
            # Normal based on entry point
            normal = np.zeros(3)
            normal[normal_axis] = normal_sign
        else:
            # Ray starts inside box, use exit point
            t = t_far
            # Recalculate normal for exit point
            hit_point = ray_origin + t * ray_direction
            # Find which face we're exiting through
            normal = np.zeros(3)
            for axis in range(3):
                if abs(hit_point[axis] - box_min[axis]) < EPSILON:
                    normal[axis] = -1  # Exiting through min face
                    break
                elif abs(hit_point[axis] - box_max[axis]) < EPSILON:
                    normal[axis] = 1   # Exiting through max face
                    break

        return t, normal

    def intersect_batch(self, ray_origins, ray_directions):
        """
        Test multiple rays against this cube simultaneously using vectorization.

        Uses NumPy broadcasting to test N rays in parallel using the slab method,
        significantly faster than testing each ray individually in a loop.

        Args:
            ray_origins: np.ndarray, shape (N, 3) - origins of N rays
            ray_directions: np.ndarray, shape (N, 3) - directions of N rays (normalized)

        Returns:
            t_values: np.ndarray, shape (N,) - distance to nearest intersection for each ray
                      np.inf where no valid intersection occurs
        """
        # Calculate min and max corners of the cube
        half_scale = self.scale / 2.0
        box_min = self.position - half_scale
        box_max = self.position + half_scale

        # Initialize near and far intersection distances for all rays
        N = len(ray_origins)
        t_near = np.full(N, -np.inf)  # Largest entry point
        t_far = np.full(N, np.inf)    # Smallest exit point

        # Check intersection with slabs for each axis (x, y, z)
        for axis in range(3):
            # Check if rays are parallel to this slab
            parallel_mask = np.abs(ray_directions[:, axis]) < EPSILON

            # For parallel rays, check if they're within the slab bounds
            outside_slab = (ray_origins[:, axis] < box_min[axis]) | (ray_origins[:, axis] > box_max[axis])
            invalid_parallel = parallel_mask & outside_slab

            # Mark invalid parallel rays as no intersection
            t_far[invalid_parallel] = -np.inf

            # For non-parallel rays, calculate intersection distances
            non_parallel_mask = ~parallel_mask

            if np.any(non_parallel_mask):
                # Calculate intersection distances for this axis's two planes
                inv_dir = 1.0 / ray_directions[non_parallel_mask, axis]
                t1 = (box_min[axis] - ray_origins[non_parallel_mask, axis]) * inv_dir
                t2 = (box_max[axis] - ray_origins[non_parallel_mask, axis]) * inv_dir

                # Ensure t1 is near and t2 is far
                t_min = np.minimum(t1, t2)
                t_max = np.maximum(t1, t2)

                # Update overall near and far intersections
                t_near[non_parallel_mask] = np.maximum(t_near[non_parallel_mask], t_min)
                t_far[non_parallel_mask] = np.minimum(t_far[non_parallel_mask], t_max)

        # Ray misses box if t_near > t_far
        miss_mask = t_near > t_far

        # Box is entirely behind the ray if t_far < 0
        behind_mask = t_far < 0

        # Combine invalid conditions
        invalid_mask = miss_mask | behind_mask

        # Calculate final t values
        # Use t_near if positive, otherwise t_far (ray starts inside/on box)
        # This matches the non-batch version logic at lines 78-86
        t_values = np.where(t_near > EPSILON, t_near,
                           np.where(t_far > EPSILON, t_far, np.inf))

        # Set invalid intersections to infinity
        t_values[invalid_mask] = np.inf

        return t_values
