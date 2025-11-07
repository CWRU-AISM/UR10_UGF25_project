"""
Automated NeRF Capture with Robot Arm

This script demonstrates automated multi-view capture using the UR10 robot
to move the camera through predefined viewpoints around an object.

The robot moves the camera in a hemisphere pattern around the target object,
ensuring consistent coverage from multiple angles.

Usage:
    python robot_nerf_capture.py
    python robot_nerf_capture.py --robot-ip 192.168.1.101 --object-center 0.4 0.3 0.2
"""

import sys
from pathlib import Path
import argparse

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.perception.nerf_studio import RobotNeRFCapture


def automated_capture(
    robot_ip: str,
    object_center: tuple,
    radius: float = 0.5,
    num_views: int = 36,
    output_dir: str = "data/nerf_robot"
):
    """
    Automated capture using robot arm.

    Args:
        robot_ip: IP address of UR10 robot
        object_center: (x, y, z) coordinates of object center in meters
        radius: Distance from object center to camera positions
        num_views: Number of views per elevation level
        output_dir: Output directory for captured data
    """
    print("\nAutomated NeRF Capture with Robot Arm")
    print(f"Robot IP: {robot_ip}")
    print(f"Object center: {object_center}")
    print(f"Capture radius: {radius}m")
    print(f"Views per level: {num_views}")
    print(f"Output: {output_dir}\n")

    print("Safety checklist:")
    print("1. Emergency stop is accessible")
    print("2. Workspace is clear of obstacles")
    print("3. Robot is in normal mode (not protective stop)")
    print("4. Camera is securely mounted to robot end-effector\n")

    response = input("All safety checks complete? (yes/no): ").strip().lower()
    if response != "yes":
        print("Aborting. Complete safety checks before running.")
        return

    robot_capture = RobotNeRFCapture(output_dir=output_dir)

    print("\nConnecting to robot and camera...")
    try:
        robot_capture.connect(
            robot_ip=robot_ip,
            kinect_device_id=0
        )
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    print("\nStarting automated capture...")
    print("The robot will now move through multiple viewpoints.")

    try:
        robot_capture.capture_hemisphere(
            center_point=object_center,
            radius=radius,
            num_views=num_views,
            elevation_angles=[15, 30, 45]
        )
    except Exception as e:
        print(f"\nCapture failed: {e}")
    finally:
        print("\nDisconnecting...")
        robot_capture.disconnect()

    print(f"\nCapture complete! Data saved to: {output_dir}")
    print("\nNext steps:")
    print(f"1. Review captured images in {output_dir}/images/")
    print(f"2. Train NeRF: python examples/nerf_examples/train_nerf_workflow.py {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Automated NeRF capture with robot arm"
    )
    parser.add_argument(
        "--robot-ip",
        type=str,
        default="192.168.1.101",
        help="Robot IP address (default: 192.168.1.101)"
    )
    parser.add_argument(
        "--object-center",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=[0.4, 0.3, 0.2],
        help="Object center coordinates in meters (default: 0.4 0.3 0.2)"
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.5,
        help="Capture radius in meters (default: 0.5)"
    )
    parser.add_argument(
        "--views",
        type=int,
        default=36,
        help="Number of views per elevation (default: 36)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/nerf_robot",
        help="Output directory (default: data/nerf_robot)"
    )

    args = parser.parse_args()

    automated_capture(
        robot_ip=args.robot_ip,
        object_center=tuple(args.object_center),
        radius=args.radius,
        num_views=args.views,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
