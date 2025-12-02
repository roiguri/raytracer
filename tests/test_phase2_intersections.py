"""
Phase 2 Tests: Geometric Intersections

Tests ray-object intersection calculations for:
- Spheres (quadratic equation)
- Infinite planes (linear equation)
- Cubes (slab method)

Each test validates mathematics, edge cases, and normal calculations.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from surfaces.sphere import Sphere
from surfaces.infinite_plane import InfinitePlane
from surfaces.cube import Cube


def test_sphere_basic_intersection():
    """Test basic sphere intersection from outside."""
    print("\n" + "="*70)
    print("TEST 1: Sphere - Basic Intersection")
    print("="*70)

    # Sphere at origin, radius 1
    sphere = Sphere(position=[0, 0, 0], radius=1.0, material_index=0)

    # Ray from (-5, 0, 0) shooting right toward sphere
    ray_origin = np.array([-5.0, 0.0, 0.0])
    ray_direction = np.array([1.0, 0.0, 0.0])

    t, normal = sphere.intersect(ray_origin, ray_direction)

    print(f"Sphere: center=(0,0,0), radius=1")
    print(f"Ray: origin=(-5,0,0), direction=(1,0,0)")
    print(f"Intersection t: {t}")
    print(f"Hit point: {ray_origin + t * ray_direction}")
    print(f"Normal: {normal}")

    # Should hit at x=-1 (left side of sphere)
    assert t is not None, "Ray should intersect sphere"
    assert abs(t - 4.0) < 1e-6, f"Expected t=4.0, got t={t}"

    # Normal should point left (-1, 0, 0)
    expected_normal = np.array([-1.0, 0.0, 0.0])
    assert np.allclose(normal, expected_normal), f"Expected normal {expected_normal}, got {normal}"

    # Normal should be unit length
    assert abs(np.linalg.norm(normal) - 1.0) < 1e-6, "Normal not unit length"

    print("✓ Hit point correct: (-1, 0, 0)")
    print("✓ Normal correct: (-1, 0, 0)")
    print("✓ Normal is unit length")
    print("\n✅ TEST 1 PASSED: Basic sphere intersection works!\n")


def test_sphere_miss():
    """Test ray missing sphere."""
    print("\n" + "="*70)
    print("TEST 2: Sphere - Ray Miss")
    print("="*70)

    sphere = Sphere(position=[0, 0, 0], radius=1.0, material_index=0)

    # Ray passing above sphere
    ray_origin = np.array([-5.0, 5.0, 0.0])
    ray_direction = np.array([1.0, 0.0, 0.0])

    t, normal = sphere.intersect(ray_origin, ray_direction)

    print(f"Sphere: center=(0,0,0), radius=1")
    print(f"Ray: origin=(-5,5,0), direction=(1,0,0) [passing above]")
    print(f"Result: t={t}, normal={normal}")

    assert t is None, "Ray should miss sphere"
    assert normal is None, "Normal should be None for miss"

    print("✓ Correctly detected miss")
    print("\n✅ TEST 2 PASSED: Sphere miss detection works!\n")


def test_sphere_inside():
    """Test ray starting inside sphere."""
    print("\n" + "="*70)
    print("TEST 3: Sphere - Ray Inside")
    print("="*70)

    sphere = Sphere(position=[0, 0, 0], radius=2.0, material_index=0)

    # Ray starting at origin (inside sphere)
    ray_origin = np.array([0.0, 0.0, 0.0])
    ray_direction = np.array([1.0, 0.0, 0.0])

    t, normal = sphere.intersect(ray_origin, ray_direction)

    print(f"Sphere: center=(0,0,0), radius=2")
    print(f"Ray: origin=(0,0,0) [INSIDE], direction=(1,0,0)")
    print(f"Intersection t: {t}")
    print(f"Hit point: {ray_origin + t * ray_direction}")
    print(f"Normal: {normal}")

    # Should hit at x=2 (exiting right side)
    assert t is not None, "Ray should intersect sphere from inside"
    assert abs(t - 2.0) < 1e-6, f"Expected t=2.0, got t={t}"

    # Normal should point right (1, 0, 0)
    expected_normal = np.array([1.0, 0.0, 0.0])
    assert np.allclose(normal, expected_normal), f"Expected normal {expected_normal}, got {normal}"

    print("✓ Hit point correct: (2, 0, 0)")
    print("✓ Normal correct: (1, 0, 0)")
    print("\n✅ TEST 3 PASSED: Ray inside sphere works!\n")


def test_plane_basic_intersection():
    """Test basic plane intersection."""
    print("\n" + "="*70)
    print("TEST 4: Plane - Basic Intersection")
    print("="*70)

    # Horizontal plane at y=5 (floor at height 5)
    plane = InfinitePlane(normal=[0, 1, 0], offset=5.0, material_index=0)

    # Ray from above shooting down
    ray_origin = np.array([0.0, 10.0, 0.0])
    ray_direction = np.array([0.0, -1.0, 0.0])

    t, normal = plane.intersect(ray_origin, ray_direction)

    print(f"Plane: normal=(0,1,0), offset=5 [horizontal at y=5]")
    print(f"Ray: origin=(0,10,0), direction=(0,-1,0) [shooting down]")
    print(f"Intersection t: {t}")
    print(f"Hit point: {ray_origin + t * ray_direction}")
    print(f"Normal: {normal}")

    # Should hit at t=5 (travel 5 units down from y=10 to y=5)
    assert t is not None, "Ray should intersect plane"
    assert abs(t - 5.0) < 1e-6, f"Expected t=5.0, got t={t}"

    # Normal should be (0, 1, 0) pointing up
    expected_normal = np.array([0.0, 1.0, 0.0])
    assert np.allclose(normal, expected_normal), f"Expected normal {expected_normal}, got {normal}"

    print("✓ Hit point correct: (0, 5, 0)")
    print("✓ Normal correct: (0, 1, 0)")
    print("\n✅ TEST 4 PASSED: Basic plane intersection works!\n")


def test_plane_parallel():
    """Test ray parallel to plane."""
    print("\n" + "="*70)
    print("TEST 5: Plane - Parallel Ray")
    print("="*70)

    # Horizontal plane at y=0
    plane = InfinitePlane(normal=[0, 1, 0], offset=0.0, material_index=0)

    # Ray moving horizontally (parallel to plane)
    ray_origin = np.array([0.0, 5.0, 0.0])
    ray_direction = np.array([1.0, 0.0, 0.0])

    t, normal = plane.intersect(ray_origin, ray_direction)

    print(f"Plane: normal=(0,1,0), offset=0 [horizontal at y=0]")
    print(f"Ray: origin=(0,5,0), direction=(1,0,0) [parallel to plane]")
    print(f"Result: t={t}, normal={normal}")

    assert t is None, "Parallel ray should not intersect plane"
    assert normal is None, "Normal should be None for parallel ray"

    print("✓ Correctly detected parallel ray")
    print("\n✅ TEST 5 PASSED: Plane parallel detection works!\n")


def test_plane_behind():
    """Test plane behind ray."""
    print("\n" + "="*70)
    print("TEST 6: Plane - Behind Ray")
    print("="*70)

    # Plane at y=0
    plane = InfinitePlane(normal=[0, 1, 0], offset=0.0, material_index=0)

    # Ray at y=5 shooting up (away from plane)
    ray_origin = np.array([0.0, 5.0, 0.0])
    ray_direction = np.array([0.0, 1.0, 0.0])

    t, normal = plane.intersect(ray_origin, ray_direction)

    print(f"Plane: normal=(0,1,0), offset=0 [at y=0]")
    print(f"Ray: origin=(0,5,0), direction=(0,1,0) [shooting away]")
    print(f"Result: t={t}, normal={normal}")

    assert t is None, "Plane behind ray should not intersect"
    assert normal is None, "Normal should be None for plane behind"

    print("✓ Correctly detected plane behind ray")
    print("\n✅ TEST 6 PASSED: Plane behind detection works!\n")


def test_plane_normal_normalization():
    """Test that plane normalizes non-unit normals."""
    print("\n" + "="*70)
    print("TEST 7: Plane - Normal Normalization")
    print("="*70)

    # Plane with non-unit normal (length 2)
    plane = InfinitePlane(normal=[0, 2, 0], offset=5.0, material_index=0)

    print(f"Input normal: (0, 2, 0) [length=2]")
    print(f"Stored normal: {plane.normal}")
    print(f"Normal length: {np.linalg.norm(plane.normal)}")

    # Check that normal was normalized
    assert abs(np.linalg.norm(plane.normal) - 1.0) < 1e-6, "Normal should be normalized"
    expected_normal = np.array([0.0, 1.0, 0.0])
    assert np.allclose(plane.normal, expected_normal), "Normal should be (0, 1, 0)"

    print("✓ Normal correctly normalized to unit length")
    print("\n✅ TEST 7 PASSED: Plane normal normalization works!\n")


def test_cube_basic_intersection():
    """Test basic cube intersection."""
    print("\n" + "="*70)
    print("TEST 8: Cube - Basic Intersection")
    print("="*70)

    # Cube at origin, scale 2 (from -1 to 1 on each axis)
    cube = Cube(position=[0, 0, 0], scale=2.0, material_index=0)

    # Ray from left shooting right
    ray_origin = np.array([-5.0, 0.0, 0.0])
    ray_direction = np.array([1.0, 0.0, 0.0])

    t, normal = cube.intersect(ray_origin, ray_direction)

    print(f"Cube: center=(0,0,0), scale=2 [from -1 to 1 on each axis]")
    print(f"Ray: origin=(-5,0,0), direction=(1,0,0)")
    print(f"Intersection t: {t}")
    print(f"Hit point: {ray_origin + t * ray_direction}")
    print(f"Normal: {normal}")

    # Should hit at x=-1 (left face)
    assert t is not None, "Ray should intersect cube"
    assert abs(t - 4.0) < 1e-6, f"Expected t=4.0, got t={t}"

    # Normal should point left (-1, 0, 0)
    expected_normal = np.array([-1.0, 0.0, 0.0])
    assert np.allclose(normal, expected_normal), f"Expected normal {expected_normal}, got {normal}"

    print("✓ Hit point correct: (-1, 0, 0)")
    print("✓ Normal correct: (-1, 0, 0)")
    print("\n✅ TEST 8 PASSED: Basic cube intersection works!\n")


def test_cube_miss():
    """Test ray missing cube."""
    print("\n" + "="*70)
    print("TEST 9: Cube - Ray Miss")
    print("="*70)

    cube = Cube(position=[0, 0, 0], scale=2.0, material_index=0)

    # Ray passing above cube
    ray_origin = np.array([-5.0, 5.0, 0.0])
    ray_direction = np.array([1.0, 0.0, 0.0])

    t, normal = cube.intersect(ray_origin, ray_direction)

    print(f"Cube: center=(0,0,0), scale=2")
    print(f"Ray: origin=(-5,5,0), direction=(1,0,0) [passing above]")
    print(f"Result: t={t}, normal={normal}")

    assert t is None, "Ray should miss cube"
    assert normal is None, "Normal should be None for miss"

    print("✓ Correctly detected miss")
    print("\n✅ TEST 9 PASSED: Cube miss detection works!\n")


def test_cube_inside():
    """Test ray starting inside cube - CRITICAL EDGE CASE."""
    print("\n" + "="*70)
    print("TEST 10: Cube - Ray Inside (Critical Edge Case)")
    print("="*70)

    # Cube from -2 to 2 on each axis
    cube = Cube(position=[0, 0, 0], scale=4.0, material_index=0)

    # Ray starting at origin (inside cube) shooting right
    ray_origin = np.array([0.0, 0.0, 0.0])
    ray_direction = np.array([1.0, 0.0, 0.0])

    t, normal = cube.intersect(ray_origin, ray_direction)

    print(f"Cube: center=(0,0,0), scale=4 [from -2 to 2]")
    print(f"Ray: origin=(0,0,0) [INSIDE], direction=(1,0,0)")
    print(f"Intersection t: {t}")
    print(f"Hit point: {ray_origin + t * ray_direction}")
    print(f"Normal: {normal}")

    # Should hit at x=2 (exiting right face)
    assert t is not None, "Ray should intersect cube from inside"
    assert abs(t - 2.0) < 1e-6, f"Expected t=2.0, got t={t}"

    # Normal should point RIGHT (+1, 0, 0) - exiting right face
    expected_normal = np.array([1.0, 0.0, 0.0])
    assert np.allclose(normal, expected_normal), \
        f"Expected normal {expected_normal}, got {normal} - Normal should point OUTWARD from exit face!"

    print("✓ Hit point correct: (2, 0, 0)")
    print("✓ Normal correct: (1, 0, 0) [outward from exit face]")
    print("✓ Edge case handled: normal recalculated for exit point")
    print("\n✅ TEST 10 PASSED: Ray inside cube works correctly!\n")


def test_cube_corner_hit():
    """Test hitting cube at an angle (multiple slabs)."""
    print("\n" + "="*70)
    print("TEST 11: Cube - Corner/Diagonal Hit")
    print("="*70)

    cube = Cube(position=[0, 0, 0], scale=2.0, material_index=0)

    # Ray from diagonal direction
    ray_origin = np.array([-5.0, -5.0, 0.0])
    ray_direction = np.array([1.0, 1.0, 0.0])
    ray_direction = ray_direction / np.linalg.norm(ray_direction)  # Normalize

    t, normal = cube.intersect(ray_origin, ray_direction)

    print(f"Cube: center=(0,0,0), scale=2")
    print(f"Ray: origin=(-5,-5,0), direction={ray_direction} [diagonal]")
    print(f"Intersection t: {t}")

    if t:
        hit_point = ray_origin + t * ray_direction
        print(f"Hit point: {hit_point}")
        print(f"Normal: {normal}")

        # Verify hit point is on cube surface
        assert np.any(np.abs(np.abs(hit_point) - 1.0) < 1e-3), "Hit point should be on cube surface"

        # Verify normal is axis-aligned
        assert np.sum(np.abs(normal)) == 1.0, "Normal should be axis-aligned"

        print("✓ Hit point on cube surface")
        print("✓ Normal is axis-aligned")
        print("\n✅ TEST 11 PASSED: Diagonal cube hit works!\n")
    else:
        assert False, "Ray should intersect cube"


def test_cube_parallel_slab():
    """Test ray parallel to one slab."""
    print("\n" + "="*70)
    print("TEST 12: Cube - Parallel to Slab")
    print("="*70)

    cube = Cube(position=[0, 0, 0], scale=2.0, material_index=0)

    # Ray moving in XZ plane (parallel to Y slab)
    ray_origin = np.array([-5.0, 0.0, 0.0])
    ray_direction = np.array([1.0, 0.0, 0.0])

    t, normal = cube.intersect(ray_origin, ray_direction)

    print(f"Cube: center=(0,0,0), scale=2")
    print(f"Ray: origin=(-5,0,0), direction=(1,0,0) [parallel to Y slab]")
    print(f"Ray Y position: {ray_origin[1]} (within cube Y bounds: -1 to 1)")

    assert t is not None, "Ray parallel to slab but within bounds should hit"
    print(f"Intersection t: {t}")
    print(f"Hit point: {ray_origin + t * ray_direction}")
    print(f"Normal: {normal}")

    print("✓ Correctly handled parallel slab within bounds")
    print("\n✅ TEST 12 PASSED: Cube parallel slab works!\n")


def run_all_tests():
    """Run all Phase 2 intersection tests."""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  PHASE 2: GEOMETRIC INTERSECTIONS TESTS".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)

    tests = [
        ("Sphere - Basic Intersection", test_sphere_basic_intersection),
        ("Sphere - Ray Miss", test_sphere_miss),
        ("Sphere - Ray Inside", test_sphere_inside),
        ("Plane - Basic Intersection", test_plane_basic_intersection),
        ("Plane - Parallel Ray", test_plane_parallel),
        ("Plane - Behind Ray", test_plane_behind),
        ("Plane - Normal Normalization", test_plane_normal_normalization),
        ("Cube - Basic Intersection", test_cube_basic_intersection),
        ("Cube - Ray Miss", test_cube_miss),
        ("Cube - Ray Inside (CRITICAL)", test_cube_inside),
        ("Cube - Diagonal Hit", test_cube_corner_hit),
        ("Cube - Parallel Slab", test_cube_parallel_slab),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n❌ TEST FAILED: {test_name}")
            print(f"   Error: {e}\n")
            failed += 1
        except Exception as e:
            print(f"\n❌ TEST ERROR: {test_name}")
            print(f"   Exception: {e}\n")
            failed += 1

    # Summary
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  TEST SUMMARY".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    print(f"\n  Total Tests: {passed + failed}")
    print(f"  ✅ Passed: {passed}")
    print(f"  ❌ Failed: {failed}")

    if failed == 0:
        print("\n" + "🎉"*35)
        print("  ALL PHASE 2 TESTS PASSED!")
        print("  Ray-object intersections are working correctly!")
        print("  Ready to move to Phase 3: Shading & Illumination")
        print("🎉"*35 + "\n")
        return True
    else:
        print("\n⚠️  Some tests failed. Please review and fix before proceeding.\n")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
