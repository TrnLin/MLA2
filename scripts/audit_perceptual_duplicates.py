"""Run the optional label-free perceptual duplicate diagnostic."""

from fashion.eda.perceptual import run_perceptual_audit

if __name__ == "__main__":
    result = run_perceptual_audit()
    print(f"Perceptual audit complete: {result['total_images']:,} images checked.")
