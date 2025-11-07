"""
NeRF Training and Export Workflow

This script demonstrates how to train a NeRF model using NeRF Studio
and export the results to various formats.

Usage:
    python train_nerf_workflow.py data/nerf_basic
    python train_nerf_workflow.py data/nerf_basic --method instant-ngp
    python train_nerf_workflow.py data/nerf_basic --iterations 50000
"""

import sys
from pathlib import Path
import argparse

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.perception.nerf_studio import NeRFTrainer


def train_and_export(
    data_dir: str,
    method: str = "nerfacto",
    max_iterations: int = 30000,
    export_mesh: bool = True,
    export_pointcloud: bool = False
):
    """
    Complete training and export workflow.

    Args:
        data_dir: Directory containing transforms.json and images
        method: NeRF training method
        max_iterations: Maximum training iterations
        export_mesh: Whether to export mesh after training
        export_pointcloud: Whether to export point cloud after training
    """
    print(f"\nNeRF Training Workflow")
    print(f"Data directory: {data_dir}")
    print(f"Method: {method}")
    print(f"Max iterations: {max_iterations}\n")

    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return

    if not (data_path / "transforms.json").exists():
        print(f"Error: transforms.json not found in {data_dir}")
        print("Run basic_nerf_workflow.py first to capture data.")
        return

    trainer = NeRFTrainer(data_dir=data_dir)

    print("Step 1: Training NeRF model...")
    print("This may take several minutes to hours depending on iterations.\n")

    result = trainer.train(
        method=method,
        max_num_iterations=max_iterations
    )

    if result.returncode != 0:
        print("\nTraining failed. Check the error messages above.")
        return

    print("\nTraining completed!")

    output_path = trainer.output_dir / f"{method}_*" / "config.yml"
    print(f"\nTo view the trained model, run:")
    print(f"ns-viewer --load-config {output_path}")

    if export_mesh or export_pointcloud:
        print("\nNote: Export requires manual path to config.yml from training output.")
        print("Example:")
        print(f"  config_path = 'outputs/{method}_<timestamp>/nerfstudio_models/config.yml'")


def main():
    parser = argparse.ArgumentParser(
        description="Train NeRF model from captured data"
    )
    parser.add_argument(
        "data_dir",
        type=str,
        help="Directory containing transforms.json and images"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="nerfacto",
        choices=["nerfacto", "instant-ngp", "tensorf", "mipnerf"],
        help="NeRF training method (default: nerfacto)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=30000,
        help="Maximum training iterations (default: 30000)"
    )
    parser.add_argument(
        "--export-mesh",
        action="store_true",
        help="Export mesh after training (requires manual config path)"
    )
    parser.add_argument(
        "--export-pointcloud",
        action="store_true",
        help="Export point cloud after training (requires manual config path)"
    )

    args = parser.parse_args()

    train_and_export(
        data_dir=args.data_dir,
        method=args.method,
        max_iterations=args.iterations,
        export_mesh=args.export_mesh,
        export_pointcloud=args.export_pointcloud
    )


if __name__ == "__main__":
    main()
