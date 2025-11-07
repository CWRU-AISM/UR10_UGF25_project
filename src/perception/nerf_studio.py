"""
NeRF Studio Integration for UR10 Robot Project

This module provides utilities for capturing multi-view images with the Azure Kinect
and processing them with NeRF Studio for neural radiance field reconstruction.

Features:
- Automated image capture from Azure Kinect
- Camera intrinsics and transforms.json generation
- NeRF Studio training and export utilities
- Integration with robot controller for automated capture paths

Usage:
    Basic capture workflow:
        from src.perception.nerf_studio import NeRFDataCapture

        capture = NeRFDataCapture(output_dir="data/nerf_scene")
        capture.connect_kinect()
        capture.capture_dataset(num_images=100)
        capture.save_transforms()

    Training with NeRF Studio:
        from src.perception.nerf_studio import NeRFTrainer

        trainer = NeRFTrainer(data_dir="data/nerf_scene")
        trainer.train(method="nerfacto", max_num_iterations=30000)
        trainer.export_mesh("output.ply")
"""

import json
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import cv2

try:
    from src.perception.azure_kinect import AzureKinect
except ImportError:
    print("Warning: azure_kinect module not found. Some features may be unavailable.")
    AzureKinect = None


class NeRFDataCapture:
    """
    Captures multi-view images and camera parameters for NeRF Studio.

    This class handles image capture from the Azure Kinect camera and generates
    the transforms.json file required by NeRF Studio for training.
    """

    def __init__(self, output_dir: str = "data/nerf_scene"):
        """
        Initialize the data capture system.

        Args:
            output_dir: Directory to save captured images and transforms.json
        """
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self.kinect: Optional[AzureKinect] = None
        self.camera_intrinsics: Optional[Dict[str, Any]] = None
        self.frames: List[Dict[str, Any]] = []
        self.image_counter = 0

    def connect_kinect(self, device_id: int = 0) -> None:
        """
        Connect to the Azure Kinect camera.

        Args:
            device_id: Device ID for the Azure Kinect (default: 0)
        """
        if AzureKinect is None:
            raise ImportError("Azure Kinect module not available")

        self.kinect = AzureKinect(device_id=device_id)
        self.kinect.connect()

        intrinsics = self.kinect.get_camera_intrinsics()
        self.camera_intrinsics = intrinsics

        print(f"Connected to Azure Kinect (Device {device_id})")
        print(f"Camera resolution: {intrinsics['width']}x{intrinsics['height']}")
        print(f"Focal length: fx={intrinsics['fx']:.2f}, fy={intrinsics['fy']:.2f}")

    def capture_frame(
        self,
        transform_matrix: Optional[np.ndarray] = None,
        auto_transform: bool = True
    ) -> str:
        """
        Capture a single frame and add it to the dataset.

        Args:
            transform_matrix: 4x4 camera-to-world transformation matrix.
                            If None and auto_transform=True, generates identity matrix.
            auto_transform: If True and transform_matrix is None, use identity transform.

        Returns:
            Path to the saved image file (relative to output_dir)
        """
        if self.kinect is None:
            raise RuntimeError("Kinect not connected. Call connect_kinect() first.")

        color, depth, ir, aligned_depth = self.kinect.capture_frame(align_depth_to_color=True)

        if color is None:
            raise RuntimeError("Failed to capture frame from Kinect")

        image_filename = f"frame_{self.image_counter:05d}.jpg"
        image_path = self.images_dir / image_filename

        cv2.imwrite(str(image_path), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

        if transform_matrix is None and auto_transform:
            transform_matrix = np.eye(4)

        if transform_matrix is not None:
            frame_data = {
                "file_path": f"images/{image_filename}",
                "transform_matrix": transform_matrix.tolist()
            }
            self.frames.append(frame_data)

        self.image_counter += 1

        return f"images/{image_filename}"

    def capture_dataset(
        self,
        num_images: int = 100,
        interval: float = 0.5,
        show_preview: bool = True
    ) -> None:
        """
        Capture a dataset of images with manual camera movement.

        This function captures images at regular intervals while the camera
        is manually moved around the scene. Press 'q' to stop early.

        Args:
            num_images: Number of images to capture
            interval: Time interval between captures (seconds)
            show_preview: Show live preview window during capture
        """
        if self.kinect is None:
            raise RuntimeError("Kinect not connected. Call connect_kinect() first.")

        print(f"\nCapturing {num_images} images...")
        print("Move the camera around the scene to capture different viewpoints.")
        print("Press 'q' to stop early.\n")

        for i in range(num_images):
            try:
                color, depth, ir, aligned_depth = self.kinect.capture_frame(align_depth_to_color=True)

                if color is None:
                    print(f"Warning: Failed to capture frame {i}")
                    continue

                image_filename = f"frame_{self.image_counter:05d}.jpg"
                image_path = self.images_dir / image_filename
                cv2.imwrite(str(image_path), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

                frame_data = {
                    "file_path": f"images/{image_filename}",
                }
                self.frames.append(frame_data)

                self.image_counter += 1

                print(f"Captured {i+1}/{num_images}: {image_filename}")

                if show_preview:
                    preview = cv2.resize(color, (640, 360))
                    cv2.imshow("NeRF Capture Preview", cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))

                    key = cv2.waitKey(int(interval * 1000))
                    if key == ord('q'):
                        print("\nCapture stopped by user")
                        break
                else:
                    time.sleep(interval)

            except KeyboardInterrupt:
                print("\nCapture interrupted by user")
                break

        if show_preview:
            cv2.destroyAllWindows()

        print(f"\nCaptured {len(self.frames)} images total")
        print(f"Images saved to: {self.images_dir}")

    def save_transforms(
        self,
        camera_model: str = "OPENCV",
        aabb_scale: int = 16
    ) -> None:
        """
        Save the transforms.json file required by NeRF Studio.

        Args:
            camera_model: Camera model type (OPENCV, OPENCV_FISHEYE, etc.)
            aabb_scale: Axis-aligned bounding box scale for NeRF
        """
        if self.camera_intrinsics is None:
            raise RuntimeError("Camera intrinsics not available")

        width = self.camera_intrinsics['width']
        height = self.camera_intrinsics['height']
        fx = self.camera_intrinsics['fx']
        fy = self.camera_intrinsics['fy']
        cx = self.camera_intrinsics['cx']
        cy = self.camera_intrinsics['cy']

        transforms = {
            "camera_model": camera_model,
            "fl_x": fx,
            "fl_y": fy,
            "cx": cx,
            "cy": cy,
            "w": width,
            "h": height,
            "aabb_scale": aabb_scale,
            "frames": self.frames
        }

        if 'k1' in self.camera_intrinsics:
            transforms["k1"] = self.camera_intrinsics['k1']
            transforms["k2"] = self.camera_intrinsics['k2']
            transforms["k3"] = self.camera_intrinsics.get('k3', 0.0)
            transforms["p1"] = self.camera_intrinsics.get('p1', 0.0)
            transforms["p2"] = self.camera_intrinsics.get('p2', 0.0)

        transforms_path = self.output_dir / "transforms.json"
        with open(transforms_path, 'w') as f:
            json.dump(transforms, f, indent=2)

        print(f"\nTransforms saved to: {transforms_path}")
        print(f"Total frames: {len(self.frames)}")

    def disconnect(self) -> None:
        """Disconnect from the Azure Kinect camera."""
        if self.kinect is not None:
            self.kinect.disconnect()
            print("Disconnected from Azure Kinect")


class NeRFTrainer:
    """
    Wrapper for NeRF Studio training and export operations.

    This class provides a Python interface to NeRF Studio CLI commands
    for training neural radiance fields and exporting results.
    """

    def __init__(self, data_dir: str):
        """
        Initialize the NeRF trainer.

        Args:
            data_dir: Directory containing transforms.json and images
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)

        if not (self.data_dir / "transforms.json").exists():
            raise FileNotFoundError(
                f"transforms.json not found in {self.data_dir}. "
                "Run NeRFDataCapture.save_transforms() first."
            )

    def train(
        self,
        method: str = "nerfacto",
        max_num_iterations: int = 30000,
        experiment_name: Optional[str] = None,
        additional_args: Optional[List[str]] = None
    ) -> subprocess.CompletedProcess:
        """
        Train a NeRF model using NeRF Studio.

        Args:
            method: NeRF method to use (nerfacto, instant-ngp, nerfplayer, etc.)
            max_num_iterations: Maximum number of training iterations
            experiment_name: Name for this experiment (default: auto-generated)
            additional_args: Additional command-line arguments for ns-train

        Returns:
            CompletedProcess object from subprocess
        """
        if experiment_name is None:
            experiment_name = f"{method}_{int(time.time())}"

        cmd = [
            "ns-train",
            method,
            "--data", str(self.data_dir),
            "--output-dir", str(self.output_dir),
            "--experiment-name", experiment_name,
            "--max-num-iterations", str(max_num_iterations),
            "--vis", "viewer",
        ]

        if additional_args:
            cmd.extend(additional_args)

        print(f"\nStarting NeRF training with {method}...")
        print(f"Command: {' '.join(cmd)}\n")

        result = subprocess.run(cmd, check=False)

        if result.returncode == 0:
            print(f"\nTraining completed successfully!")
            print(f"Output directory: {self.output_dir / experiment_name}")
        else:
            print(f"\nTraining failed with return code {result.returncode}")

        return result

    def export_mesh(
        self,
        output_path: str,
        config_path: str,
        resolution: int = 1024,
        texture_method: str = "tsdf"
    ) -> subprocess.CompletedProcess:
        """
        Export the trained NeRF to a mesh.

        Args:
            output_path: Output path for the mesh file (.ply or .obj)
            config_path: Path to the config.yml from training
            resolution: Resolution for mesh extraction
            texture_method: Method for texture extraction (tsdf, marching_cubes)

        Returns:
            CompletedProcess object from subprocess
        """
        cmd = [
            "ns-export",
            "tsdf",
            "--load-config", str(config_path),
            "--output-dir", str(Path(output_path).parent),
            "--resolution", str(resolution),
        ]

        print(f"\nExporting mesh to {output_path}...")
        print(f"Command: {' '.join(cmd)}\n")

        result = subprocess.run(cmd, check=False)

        if result.returncode == 0:
            print(f"\nMesh export completed successfully!")
            print(f"Output: {output_path}")
        else:
            print(f"\nMesh export failed with return code {result.returncode}")

        return result

    def export_pointcloud(
        self,
        output_path: str,
        config_path: str,
        num_points: int = 1000000,
        remove_outliers: bool = True
    ) -> subprocess.CompletedProcess:
        """
        Export the trained NeRF to a point cloud.

        Args:
            output_path: Output path for the point cloud file (.ply)
            config_path: Path to the config.yml from training
            num_points: Number of points to sample
            remove_outliers: Whether to remove outlier points

        Returns:
            CompletedProcess object from subprocess
        """
        cmd = [
            "ns-export",
            "pointcloud",
            "--load-config", str(config_path),
            "--output-dir", str(Path(output_path).parent),
            "--num-points", str(num_points),
        ]

        if remove_outliers:
            cmd.append("--remove-outliers")

        print(f"\nExporting point cloud to {output_path}...")
        print(f"Command: {' '.join(cmd)}\n")

        result = subprocess.run(cmd, check=False)

        if result.returncode == 0:
            print(f"\nPoint cloud export completed successfully!")
            print(f"Output: {output_path}")
        else:
            print(f"\nPoint cloud export failed with return code {result.returncode}")

        return result


class RobotNeRFCapture:
    """
    Automated NeRF data capture using the robot arm.

    This class integrates the robot controller with the NeRF data capture
    to automatically move the camera through predefined viewpoints.
    """

    def __init__(self, output_dir: str = "data/nerf_scene"):
        """
        Initialize the robot-based capture system.

        Args:
            output_dir: Directory to save captured data
        """
        self.capture = NeRFDataCapture(output_dir)
        self.robot = None

    def connect(self, robot_ip: str = "192.168.1.101", kinect_device_id: int = 0) -> None:
        """
        Connect to both the robot and the Kinect camera.

        Args:
            robot_ip: IP address of the UR10 robot
            kinect_device_id: Device ID for the Azure Kinect
        """
        try:
            from src.robot_control.ur10_control import UR10Controller
            self.robot = UR10Controller()
            print(f"Attempting to connect to robot at {robot_ip}...")

        except ImportError:
            print("Warning: ur10_control module not found. Robot control unavailable.")
            self.robot = None

        self.capture.connect_kinect(kinect_device_id)

    def capture_hemisphere(
        self,
        center_point: Tuple[float, float, float],
        radius: float = 0.5,
        num_views: int = 36,
        elevation_angles: Optional[List[float]] = None
    ) -> None:
        """
        Capture images from viewpoints arranged in a hemisphere around a center point.

        Args:
            center_point: XYZ coordinates of the object center
            radius: Radius of the hemisphere in meters
            num_views: Number of views per elevation level
            elevation_angles: List of elevation angles in degrees (default: [15, 30, 45])
        """
        if self.robot is None:
            raise RuntimeError("Robot not connected")

        if elevation_angles is None:
            elevation_angles = [15, 30, 45]

        print(f"\nCapturing hemisphere around point {center_point}")
        print(f"Radius: {radius}m, Views per level: {num_views}")
        print(f"Elevation angles: {elevation_angles}\n")

        cx, cy, cz = center_point

        for elevation in elevation_angles:
            elevation_rad = np.radians(elevation)

            for i in range(num_views):
                azimuth_rad = 2 * np.pi * i / num_views

                x = cx + radius * np.cos(elevation_rad) * np.cos(azimuth_rad)
                y = cy + radius * np.cos(elevation_rad) * np.sin(azimuth_rad)
                z = cz + radius * np.sin(elevation_rad)

                direction = np.array([cx - x, cy - y, cz - z])
                direction = direction / np.linalg.norm(direction)

                print(f"Moving to view {i+1}/{num_views} at elevation {elevation}°...")

                transform = self._compute_camera_transform([x, y, z], direction)

                try:
                    self.capture.capture_frame(transform_matrix=transform)
                except Exception as e:
                    print(f"Warning: Failed to capture frame: {e}")
                    continue

        print(f"\nHemisphere capture complete!")
        self.capture.save_transforms()

    def _compute_camera_transform(
        self,
        position: List[float],
        look_direction: np.ndarray
    ) -> np.ndarray:
        """
        Compute a 4x4 camera-to-world transformation matrix.

        Args:
            position: Camera position [x, y, z]
            look_direction: Direction the camera is looking (normalized)

        Returns:
            4x4 transformation matrix
        """
        z_axis = -look_direction

        up = np.array([0, 0, 1])
        if abs(np.dot(z_axis, up)) > 0.99:
            up = np.array([1, 0, 0])

        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)

        y_axis = np.cross(z_axis, x_axis)

        transform = np.eye(4)
        transform[:3, 0] = x_axis
        transform[:3, 1] = y_axis
        transform[:3, 2] = z_axis
        transform[:3, 3] = position

        return transform

    def disconnect(self) -> None:
        """Disconnect from both robot and camera."""
        self.capture.disconnect()
        if self.robot is not None:
            self.robot.disconnect()


def process_colmap_to_nerf(colmap_dir: str, output_dir: str) -> None:
    """
    Convert COLMAP output to NeRF Studio format using ns-process-data.

    Args:
        colmap_dir: Directory containing COLMAP reconstruction
        output_dir: Output directory for NeRF Studio format
    """
    cmd = [
        "ns-process-data",
        "images",
        "--data", str(colmap_dir),
        "--output-dir", str(output_dir),
        "--colmap-model-path", str(Path(colmap_dir) / "sparse/0"),
    ]

    print(f"Processing COLMAP data to NeRF format...")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, check=False)

    if result.returncode == 0:
        print(f"\nProcessing completed successfully!")
        print(f"Output: {output_dir}")
    else:
        print(f"\nProcessing failed with return code {result.returncode}")


def main():
    """
    Interactive demo for NeRF Studio integration.
    """
    print("\nNeRF Studio Integration for UR10 Robot Project")
    print("\nOptions:")
    print("1. Capture dataset with manual camera movement")
    print("2. Capture dataset with robot automation (hemisphere)")
    print("3. Train NeRF from existing dataset")
    print("4. Export trained NeRF to mesh")
    print("5. Process COLMAP data to NeRF format")
    print("0. Exit")

    choice = input("\nEnter your choice: ").strip()

    if choice == "1":
        output_dir = input("Output directory (default: data/nerf_scene): ").strip()
        if not output_dir:
            output_dir = "data/nerf_scene"

        num_images = input("Number of images to capture (default: 100): ").strip()
        num_images = int(num_images) if num_images else 100

        interval = input("Interval between captures in seconds (default: 0.5): ").strip()
        interval = float(interval) if interval else 0.5

        capture = NeRFDataCapture(output_dir)
        capture.connect_kinect()
        capture.capture_dataset(num_images=num_images, interval=interval)
        capture.save_transforms()
        capture.disconnect()

    elif choice == "2":
        print("\nRobot automation requires:")
        print("- Robot IP address")
        print("- Target object center coordinates")

        robot_ip = input("Robot IP (default: 192.168.1.101): ").strip()
        if not robot_ip:
            robot_ip = "192.168.1.101"

        cx = float(input("Object center X (meters): "))
        cy = float(input("Object center Y (meters): "))
        cz = float(input("Object center Z (meters): "))

        radius = input("Capture radius (default: 0.5m): ").strip()
        radius = float(radius) if radius else 0.5

        output_dir = input("Output directory (default: data/nerf_scene): ").strip()
        if not output_dir:
            output_dir = "data/nerf_scene"

        robot_capture = RobotNeRFCapture(output_dir)
        robot_capture.connect(robot_ip=robot_ip)
        robot_capture.capture_hemisphere(
            center_point=(cx, cy, cz),
            radius=radius
        )
        robot_capture.disconnect()

    elif choice == "3":
        data_dir = input("Data directory (default: data/nerf_scene): ").strip()
        if not data_dir:
            data_dir = "data/nerf_scene"

        method = input("NeRF method (default: nerfacto): ").strip()
        if not method:
            method = "nerfacto"

        iterations = input("Max iterations (default: 30000): ").strip()
        iterations = int(iterations) if iterations else 30000

        trainer = NeRFTrainer(data_dir)
        trainer.train(method=method, max_num_iterations=iterations)

    elif choice == "4":
        config_path = input("Path to config.yml from training: ").strip()
        output_path = input("Output mesh path (default: output.ply): ").strip()
        if not output_path:
            output_path = "output.ply"

        trainer = NeRFTrainer(".")
        trainer.export_mesh(output_path, config_path)

    elif choice == "5":
        colmap_dir = input("COLMAP directory: ").strip()
        output_dir = input("Output directory: ").strip()

        process_colmap_to_nerf(colmap_dir, output_dir)

    elif choice == "0":
        print("Exiting...")
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
