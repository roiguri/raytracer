"""
Phase 1 Tests: Camera Ray Generation

Tests the camera's ability to:
1. Build correct coordinate system (forward, right, up vectors)
2. Generate rays for center and corner pixels
3. Verify ray directions are correct
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from camera import Camera


def test_camera_basis_vectors():
    """Test that camera basis vectors are orthonormal."""
    print("\n" + "="*70)
    print("TEST 1: Camera Basis Vectors (Orthonormal System)")
    print("="*70)

    # Create camera from pool.txt scene
    position = np.array([0, 10, -2])
    look_at = np.array([0, -100, -4])
    up_hint = np.array([0, 1, 0])

    camera = Camera(position, look_at, up_hint, screen_distance=1.4, screen_width=1)

    print(f"\nCamera Position: {position}")
    print(f"Looking At: {look_at}")
    print(f"Up Hint: {up_hint}")
    print("\nCalculated Basis Vectors:")
    print(f"  Forward: {camera.forward}")
    print(f"  Right:   {camera.right}")
    print(f"  Up:      {camera.up}")

    # Test 1: All vectors should be unit length
    print("\n--- Unit Length Tests ---")
    forward_len = np.linalg.norm(camera.forward)
    right_len = np.linalg.norm(camera.right)
    up_len = np.linalg.norm(camera.up)

    print(f"Forward length: {forward_len:.10f} (expected: 1.0)")
    print(f"Right length:   {right_len:.10f} (expected: 1.0)")
    print(f"Up length:      {up_len:.10f} (expected: 1.0)")

    assert abs(forward_len - 1.0) < 1e-6, f"Forward not normalized: {forward_len}"
    assert abs(right_len - 1.0) < 1e-6, f"Right not normalized: {right_len}"
    assert abs(up_len - 1.0) < 1e-6, f"Up not normalized: {up_len}"
    print("✓ All vectors are unit length")

    # Test 2: Vectors should be perpendicular (dot product = 0)
    print("\n--- Perpendicularity Tests (dot products should be ~0) ---")
    dot_forward_right = np.dot(camera.forward, camera.right)
    dot_forward_up = np.dot(camera.forward, camera.up)
    dot_right_up = np.dot(camera.right, camera.up)

    print(f"Forward · Right: {dot_forward_right:.10f}")
    print(f"Forward · Up:    {dot_forward_up:.10f}")
    print(f"Right · Up:      {dot_right_up:.10f}")

    assert abs(dot_forward_right) < 1e-6, f"Forward and Right not perpendicular: {dot_forward_right}"
    assert abs(dot_forward_up) < 1e-6, f"Forward and Up not perpendicular: {dot_forward_up}"
    assert abs(dot_right_up) < 1e-6, f"Right and Up not perpendicular: {dot_right_up}"
    print("✓ All vectors are mutually perpendicular")

    # Test 3: Forward should point toward look_at
    print("\n--- Forward Direction Test ---")
    expected_forward = (look_at - position) / np.linalg.norm(look_at - position)
    print(f"Expected forward: {expected_forward}")
    print(f"Actual forward:   {camera.forward}")
    print(f"Difference:       {np.linalg.norm(camera.forward - expected_forward):.10f}")

    assert np.allclose(camera.forward, expected_forward), "Forward doesn't point toward look_at"
    print("✓ Forward vector points correctly toward look_at")

    print("\n✅ TEST 1 PASSED: Camera basis vectors are correct!\n")


def test_center_pixel_ray():
    """Test that center pixel ray aligns with forward vector."""
    print("\n" + "="*70)
    print("TEST 2: Center Pixel Ray Alignment")
    print("="*70)

    # Create camera
    position = np.array([0, 10, -2])
    look_at = np.array([0, -100, -4])
    up_hint = np.array([0, 1, 0])
    camera = Camera(position, look_at, up_hint, screen_distance=1.4, screen_width=1)

    # Test with different image sizes
    image_sizes = [(500, 500), (800, 600), (1920, 1080)]

    for width, height in image_sizes:
        print(f"\n--- Image Size: {width}×{height} ---")

        # Center pixel coordinates
        center_x = width // 2
        center_y = height // 2

        print(f"Center pixel: ({center_x}, {center_y})")

        # Get ray through center pixel
        origin, direction = camera.get_ray(center_x, center_y, width, height)

        print(f"Ray origin:    {origin}")
        print(f"Ray direction: {direction}")
        print(f"Forward:       {camera.forward}")

        # Ray direction should match forward vector
        difference = np.linalg.norm(direction - camera.forward)
        print(f"Difference from forward: {difference:.10f}")

        assert np.allclose(direction, camera.forward, atol=1e-3), \
            f"Center ray doesn't align with forward for {width}×{height}"
        print(f"✓ Center ray aligns with forward (diff: {difference:.2e})")

    print("\n✅ TEST 2 PASSED: Center pixel rays align correctly!\n")


def test_corner_pixel_divergence():
    """Test that corner pixel rays diverge symmetrically."""
    print("\n" + "="*70)
    print("TEST 3: Corner Pixel Divergence")
    print("="*70)

    # Create camera
    position = np.array([0, 10, -2])
    look_at = np.array([0, -100, -4])
    up_hint = np.array([0, 1, 0])
    camera = Camera(position, look_at, up_hint, screen_distance=1.4, screen_width=1)

    # Test with square image for simplicity
    width, height = 500, 500
    print(f"\nImage Size: {width}×{height}")

    # Get rays for all four corners
    corners = {
        "Top-Left": (0, 0),
        "Top-Right": (width - 1, 0),
        "Bottom-Left": (0, height - 1),
        "Bottom-Right": (width - 1, height - 1),
    }

    corner_rays = {}
    for name, (x, y) in corners.items():
        origin, direction = camera.get_ray(x, y, width, height)
        corner_rays[name] = direction

        # Calculate angle from forward vector
        dot = np.dot(direction, camera.forward)
        angle_deg = np.degrees(np.arccos(np.clip(dot, -1, 1)))

        print(f"\n{name:15s} ({x:3d}, {y:3d}):")
        print(f"  Direction: {direction}")
        print(f"  Angle from forward: {angle_deg:.2f}°")

    # Test symmetry: opposite corners should have similar angles
    print("\n--- Symmetry Tests ---")

    # Top-left and bottom-right should be similarly angled
    tl_angle = np.degrees(np.arccos(np.dot(corner_rays["Top-Left"], camera.forward)))
    br_angle = np.degrees(np.arccos(np.dot(corner_rays["Bottom-Right"], camera.forward)))
    print(f"Top-Left angle:     {tl_angle:.2f}°")
    print(f"Bottom-Right angle: {br_angle:.2f}°")
    print(f"Difference:         {abs(tl_angle - br_angle):.2f}°")
    assert abs(tl_angle - br_angle) < 0.1, "Top-left and bottom-right not symmetric"
    print("✓ Diagonal corners are symmetric")

    # Top-right and bottom-left should be similarly angled
    tr_angle = np.degrees(np.arccos(np.dot(corner_rays["Top-Right"], camera.forward)))
    bl_angle = np.degrees(np.arccos(np.dot(corner_rays["Bottom-Left"], camera.forward)))
    print(f"\nTop-Right angle:    {tr_angle:.2f}°")
    print(f"Bottom-Left angle:  {bl_angle:.2f}°")
    print(f"Difference:         {abs(tr_angle - bl_angle):.2f}°")
    assert abs(tr_angle - bl_angle) < 0.1, "Top-right and bottom-left not symmetric"
    print("✓ Diagonal corners are symmetric")

    # All corners should have the same angle (for square image)
    all_angles = [tl_angle, tr_angle, bl_angle, br_angle]
    print(f"\nAll corner angles: {[f'{a:.2f}°' for a in all_angles]}")
    angle_variance = max(all_angles) - min(all_angles)
    print(f"Angle variance: {angle_variance:.2f}°")
    assert angle_variance < 0.1, "Corner angles not consistent for square image"
    print("✓ All corners have consistent angles")

    print("\n✅ TEST 3 PASSED: Corner rays diverge symmetrically!\n")


def test_ray_origin():
    """Test that all rays originate from camera position."""
    print("\n" + "="*70)
    print("TEST 4: Ray Origins")
    print("="*70)

    # Create camera
    position = np.array([0, 10, -2])
    look_at = np.array([0, -100, -4])
    up_hint = np.array([0, 1, 0])
    camera = Camera(position, look_at, up_hint, screen_distance=1.4, screen_width=1)

    width, height = 800, 600

    # Test several random pixels
    test_pixels = [
        (0, 0),
        (width // 2, height // 2),
        (width - 1, height - 1),
        (100, 200),
        (700, 400),
    ]

    print(f"\nCamera position: {position}")
    print(f"\nTesting {len(test_pixels)} pixels:")

    for x, y in test_pixels:
        origin, direction = camera.get_ray(x, y, width, height)
        print(f"  Pixel ({x:3d}, {y:3d}): origin = {origin}")

        assert np.allclose(origin, position), f"Ray origin doesn't match camera position for pixel ({x}, {y})"

    print("\n✓ All rays originate from camera position")
    print("\n✅ TEST 4 PASSED: Ray origins are correct!\n")


def test_aspect_ratio():
    """Test that different aspect ratios are handled correctly."""
    print("\n" + "="*70)
    print("TEST 5: Aspect Ratio Handling")
    print("="*70)

    # Create camera
    position = np.array([0, 10, -2])
    look_at = np.array([0, -100, -4])
    up_hint = np.array([0, 1, 0])
    camera = Camera(position, look_at, up_hint, screen_distance=1.4, screen_width=1)

    # Test different aspect ratios
    aspect_ratios = [
        (800, 600, "4:3"),
        (1920, 1080, "16:9"),
        (500, 500, "1:1"),
    ]

    for width, height, ratio_name in aspect_ratios:
        print(f"\n--- Aspect Ratio {ratio_name} ({width}×{height}) ---")

        # Get corner rays
        tl_origin, tl_dir = camera.get_ray(0, 0, width, height)
        tr_origin, tr_dir = camera.get_ray(width - 1, 0, width, height)
        bl_origin, bl_dir = camera.get_ray(0, height - 1, width, height)

        # Calculate horizontal and vertical field of view
        horizontal_angle = np.degrees(np.arccos(np.dot(tl_dir, tr_dir)))
        vertical_angle = np.degrees(np.arccos(np.dot(tl_dir, bl_dir)))

        print(f"Horizontal FOV: {horizontal_angle:.2f}°")
        print(f"Vertical FOV:   {vertical_angle:.2f}°")
        print(f"FOV Ratio:      {horizontal_angle / vertical_angle:.3f}")
        print(f"Aspect Ratio:   {width / height:.3f}")

        # FOV ratio should roughly match aspect ratio
        fov_ratio = horizontal_angle / vertical_angle
        aspect_ratio = width / height
        print(f"Difference:     {abs(fov_ratio - aspect_ratio):.3f}")

        # Allow some tolerance due to the way angles work
        assert abs(fov_ratio - aspect_ratio) < 0.1, \
            f"FOV doesn't match aspect ratio for {ratio_name}"
        print(f"✓ FOV matches aspect ratio")

    print("\n✅ TEST 5 PASSED: Aspect ratios handled correctly!\n")


def run_all_tests():
    """Run all Phase 1 tests."""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  PHASE 1: CAMERA RAY GENERATION TESTS".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)

    tests = [
        ("Camera Basis Vectors", test_camera_basis_vectors),
        ("Center Pixel Ray", test_center_pixel_ray),
        ("Corner Pixel Divergence", test_corner_pixel_divergence),
        ("Ray Origins", test_ray_origin),
        ("Aspect Ratio", test_aspect_ratio),
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
        print("  ALL PHASE 1 TESTS PASSED!")
        print("  Camera ray generation is working correctly!")
        print("  Ready to move to Phase 2: Geometric Intersections")
        print("🎉"*35 + "\n")
        return True
    else:
        print("\n⚠️  Some tests failed. Please review and fix before proceeding.\n")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
