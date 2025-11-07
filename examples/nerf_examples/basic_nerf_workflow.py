"""
Basic NeRF Studio Workflow Example

This script demonstrates a complete workflow for capturing and processing
3D scenes using NeRF Studio with the Azure Kinect camera.

Workflow:
1. Connect to Azure Kinect
2. Capture multi-view images
3. Generate transforms.json
4. Train NeRF model
5. Export results
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.perception.nerf_studio import NeRFDataCapture, NeRFTrainer


def basic_capture_workflow():
    """
    Basic workflow: Manual capture with handheld camera.
    """
    print("Basic NeRF Capture Workflow")
    print("Move the camera around the object while images are captured.\n")

    output_dir = "data/nerf_basic"
    num_images = 100

    capture = NeRFDataCapture(output_dir=output_dir)

    print("Step 1: Connecting to Azure Kinect...")
    capture.connect_kinect(device_id=0)

    print(f"\nStep 2: Capturing {num_images} images...")
    print("Instructions:")
    print("- Keep the object centered in the frame")
    print("- Move smoothly around the object")
    print("- Maintain consistent distance from the object")
    print("- Press 'q' to stop early\n")

    input("Press Enter to start capture...")

    capture.capture_dataset(
        num_images=num_images,
        interval=0.5,
        show_preview=True
    )

    print("\nStep 3: Saving camera transforms...")
    capture.save_transforms()

    print("\nStep 4: Disconnecting...")
    capture.disconnect()

    print(f"\nCapture complete! Data saved to: {output_dir}")
    print("\nNext steps:")
    print(f"1. Review the captured images in {output_dir}/images/")
    print("2. Train the NeRF model using train_nerf_workflow.py")
    print(f"   or run: python examples/nerf_examples/train_nerf_workflow.py {output_dir}")


def quick_capture(output_dir: str = "data/nerf_quick", num_images: int = 50):
    """
    Quick capture workflow with minimal user interaction.

    Args:
        output_dir: Output directory for captured data
        num_images: Number of images to capture
    """
    capture = NeRFDataCapture(output_dir=output_dir)
    capture.connect_kinect()

    print(f"Quick capture: {num_images} images to {output_dir}")
    print("Move the camera now!\n")

    capture.capture_dataset(
        num_images=num_images,
        interval=0.3,
        show_preview=True
    )

    capture.save_transforms()
    capture.disconnect()

    print(f"\nDone! Data in {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NeRF data capture workflow")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use quick capture mode (50 images)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/nerf_basic",
        help="Output directory (default: data/nerf_basic)"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=100,
        help="Number of images to capture (default: 100)"
    )

    args = parser.parse_args()

    if args.quick:
        quick_capture(args.output, min(args.num_images, 50))
    else:
        basic_capture_workflow()
