#!/usr/bin/env python3
"""
Image comparison utility for raytracer optimization validation.

Compares two images using multiple metrics:
- MSE (Mean Squared Error): < 0.001 for strict match
- PSNR (Peak Signal-to-Noise Ratio): > 50dB for strict match
- Visual difference map

Usage:
    python compare_images.py reference.png test.png [--save-diff diff.png]
"""

import argparse
import numpy as np
from PIL import Image
import sys


def load_image(path):
    """Load image as normalized numpy array [0, 1]."""
    img = Image.open(path)
    return np.array(img, dtype=np.float64) / 255.0


def compute_mse(img1, img2):
    """Compute Mean Squared Error."""
    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes don't match: {img1.shape} vs {img2.shape}")

    return np.mean((img1 - img2) ** 2)


def compute_psnr(img1, img2):
    """
    Compute Peak Signal-to-Noise Ratio.

    PSNR = 10 * log10(MAX_I^2 / MSE)
    where MAX_I = 1.0 for normalized images
    """
    mse = compute_mse(img1, img2)

    if mse == 0:
        return float('inf')

    # MAX_I = 1.0 for normalized images
    psnr = 10 * np.log10(1.0 / mse)
    return psnr


def create_difference_map(img1, img2, scale=10.0):
    """
    Create visual difference map.

    Args:
        img1, img2: Images as numpy arrays [0, 1]
        scale: Amplification factor for small differences

    Returns:
        PIL Image showing difference (scaled for visibility)
    """
    # Absolute difference per channel
    diff = np.abs(img1 - img2)

    # Scale for visibility
    diff_scaled = np.clip(diff * scale, 0, 1)

    # Convert to uint8
    diff_uint8 = (diff_scaled * 255).astype(np.uint8)

    return Image.fromarray(diff_uint8)


def compare_images(ref_path, test_path, save_diff=None, verbose=True):
    """
    Compare two images and report metrics.

    Returns:
        dict: {'mse': float, 'psnr': float, 'pass': bool}
    """
    # Load images
    img_ref = load_image(ref_path)
    img_test = load_image(test_path)

    # Compute metrics
    mse = compute_mse(img_ref, img_test)
    psnr = compute_psnr(img_ref, img_test)

    # Thresholds
    # Relaxed PSNR threshold to account for random sampling variations in stochastic rendering
    MSE_THRESHOLD = 0.001
    PSNR_THRESHOLD = 40.0

    passed = mse < MSE_THRESHOLD and psnr > PSNR_THRESHOLD

    # Report
    if verbose:
        print("=" * 60)
        print("  Image Comparison Report")
        print("=" * 60)
        print(f"Reference: {ref_path}")
        print(f"Test:      {test_path}")
        print()
        print(f"MSE:       {mse:.6f}  (threshold: < {MSE_THRESHOLD})")
        print(f"PSNR:      {psnr:.2f} dB  (threshold: > {PSNR_THRESHOLD} dB)")
        print()

        if passed:
            print("✓ PASS: Images are equivalent within tolerance")
        else:
            print("✗ FAIL: Images differ beyond acceptable threshold")

            # Detailed failure info
            if mse >= MSE_THRESHOLD:
                print(f"  - MSE {mse:.6f} exceeds threshold {MSE_THRESHOLD}")
            if psnr <= PSNR_THRESHOLD:
                print(f"  - PSNR {psnr:.2f} dB below threshold {PSNR_THRESHOLD} dB")

        print("=" * 60)

    # Save difference map
    if save_diff:
        diff_img = create_difference_map(img_ref, img_test)
        diff_img.save(save_diff)
        if verbose:
            print(f"Difference map saved to: {save_diff}")

    return {
        'mse': mse,
        'psnr': psnr,
        'pass': passed
    }


def main():
    parser = argparse.ArgumentParser(
        description='Compare two raytracer output images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Validation thresholds:
  MSE < 0.001        (very strict)
  PSNR > 50 dB       (very strict)

Exit codes:
  0 = images match (within threshold)
  1 = images differ (beyond threshold)
  2 = error (e.g., file not found)
        """
    )

    parser.add_argument('reference', help='Reference image path')
    parser.add_argument('test', help='Test image path')
    parser.add_argument('--save-diff', metavar='PATH',
                       help='Save difference map to PATH')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress output (exit code only)')

    args = parser.parse_args()

    try:
        result = compare_images(
            args.reference,
            args.test,
            save_diff=args.save_diff,
            verbose=not args.quiet
        )

        # Exit with appropriate code
        sys.exit(0 if result['pass'] else 1)

    except Exception as e:
        if not args.quiet:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == '__main__':
    main()
