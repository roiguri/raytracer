import numpy as np

# Epsilon for floating-point comparisons
# Used to check if values are "close enough" to zero
# Also used as minimum t value to prevent self-intersection
EPSILON = 1e-6


def normalize(v):
    """
    Normalize a vector to unit length.

    Args:
        v: Vector (numpy array)

    Returns:
        Normalized vector (unit length) or original if zero-length
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm
