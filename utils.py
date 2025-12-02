import numpy as np


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
