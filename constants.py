"""
Global constants for the ray tracer.

These constants are used throughout the project to ensure
consistency in numerical comparisons and calculations.
"""

# Epsilon for floating-point comparisons
# Used to check if values are "close enough" to zero
EPSILON = 1e-6

# Minimum positive t value for ray intersections
# Prevents self-intersection due to floating-point errors
MIN_T = 1e-6
