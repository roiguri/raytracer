from PIL import Image
import numpy as np
import sys

def compare_images(ref_path, out_path, diff_path):
    print(f"Comparing {ref_path} and {out_path}")
    try:
        ref = Image.open(ref_path).convert('RGB')
        out = Image.open(out_path).convert('RGB')
    except Exception as e:
        print(f"Error opening images: {e}")
        sys.exit(1)

    ref_arr = np.array(ref, dtype=np.float32)
    out_arr = np.array(out, dtype=np.float32)

    if ref_arr.shape != out_arr.shape:
        print(f"Shape mismatch: {ref_arr.shape} vs {out_arr.shape}")
        # Resize out to match ref?
        out = out.resize(ref.size)
        out_arr = np.array(out, dtype=np.float32)
        print("Resized Outcome to match Reference.")

    diff = np.abs(ref_arr - out_arr)
    mae = np.mean(diff)
    max_diff = np.max(diff)
    
    print(f"Mean Absolute Error (per channel): {mae:.2f}")
    print(f"Max Difference: {max_diff:.2f}")

    # Create diff image
    # Scale diff for visibility? If max diff is small.
    # But if max diff is large (e.g. 255), standard is fine.
    
    diff_img = Image.fromarray(diff.astype(np.uint8))
    diff_img.save(diff_path)
    print(f"Saved diff image to {diff_path}")

    # Analyze specific areas
    # Check average brightness
    print(f"Ref Mean Brightness: {np.mean(ref_arr):.2f}")
    print(f"Out Mean Brightness: {np.mean(out_arr):.2f}")

if __name__ == "__main__":
    compare_images("reference.png", "new_outcome2.png", "diff.png")
