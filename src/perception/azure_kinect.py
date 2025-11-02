"""
Azure Kinect DK Integration
Pyk4a for depth sensing and point cloud generation
"""

import numpy as np
import cv2
import pyk4a
from pyk4a import Config, PyK4A, ColorResolution, DepthMode, ImageFormat, FPS
import open3d as o3d
from typing import Optional, Tuple, List
import time
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R


@dataclass
class KinectConfig:
    """Configuration for Azure Kinect DK"""
    color_resolution: ColorResolution = ColorResolution.RES_1080P
    depth_mode: DepthMode = DepthMode.NFOV_UNBINNED
    camera_fps: FPS = FPS.FPS_30
    synchronized_images_only: bool = True
    depth_delay_off_color_usec: int = 0
    wired_sync_mode: pyk4a.WiredSyncMode = pyk4a.WiredSyncMode.STANDALONE
    subordinate_delay_off_master_usec: int = 0
    disable_streaming_indicator: bool = False


class AzureKinect:
    """Main class for Azure Kinect DK operations"""

    def __init__(self, device_id: int = 0, config: Optional[KinectConfig] = None):
        """Initialize Azure Kinect device"""
        self.device_id = device_id
        self.config = config or KinectConfig()
        self.device = None
        self.calibration = None
        self.transformation = None

    def connect(self) -> bool:
        """Connect to Azure Kinect device"""
        try:
            # Create configuration
            k4a_config = Config(
                color_format=ImageFormat.COLOR_BGRA32,
                color_resolution=self.config.color_resolution,
                depth_mode=self.config.depth_mode,
                camera_fps=self.config.camera_fps,
                synchronized_images_only=self.config.synchronized_images_only,
                depth_delay_off_color_usec=self.config.depth_delay_off_color_usec,
                wired_sync_mode=self.config.wired_sync_mode,
                subordinate_delay_off_master_usec=self.config.subordinate_delay_off_master_usec,
                disable_streaming_indicator=self.config.disable_streaming_indicator
            )

            # Initialize device
            self.device = PyK4A(config=k4a_config, device_id=self.device_id)
            self.device.start()

            # Get calibration (factory intrinsics)
            self.calibration = self.device.calibration

            # Wait for device to stabilize
            time.sleep(0.5)

            print(f"Connected to Azure Kinect device {self.device_id}")
            return True

        except Exception as e:
            print(f"Failed to connect to Azure Kinect: {e}")
            return False

    def get_camera_intrinsics(self) -> dict:
        """
        Get factory camera intrinsics from device calibration

        Returns:
            Dictionary with color and depth camera intrinsics
        """
        if not self.calibration:
            return None

        try:
            # Get color camera intrinsics
            color_intrinsics = self.calibration.get_camera_matrix(pyk4a.CalibrationType.COLOR)
            color_distortion = self.calibration.get_distortion_coefficients(pyk4a.CalibrationType.COLOR)

            # Get depth camera intrinsics
            depth_intrinsics = self.calibration.get_camera_matrix(pyk4a.CalibrationType.DEPTH)
            depth_distortion = self.calibration.get_distortion_coefficients(pyk4a.CalibrationType.DEPTH)

            return {
                'color': {
                    'intrinsics': color_intrinsics,
                    'distortion': color_distortion,
                    'fx': color_intrinsics[0, 0],
                    'fy': color_intrinsics[1, 1],
                    'cx': color_intrinsics[0, 2],
                    'cy': color_intrinsics[1, 2]
                },
                'depth': {
                    'intrinsics': depth_intrinsics,
                    'distortion': depth_distortion,
                    'fx': depth_intrinsics[0, 0],
                    'fy': depth_intrinsics[1, 1],
                    'cx': depth_intrinsics[0, 2],
                    'cy': depth_intrinsics[1, 2]
                }
            }
        except Exception as e:
            print(f"Failed to get intrinsics: {e}")
            return None

    def disconnect(self):
        """Disconnect from Azure Kinect device"""
        if self.device:
            self.device.stop()
            self.device = None
            print("Disconnected from Azure Kinect")

    def capture_frame(self, align_depth_to_color: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Capture a single frame from the device

        Args:
            align_depth_to_color: If True, return depth aligned to color camera

        Returns: (color_image, depth_image, ir_image, aligned_depth_image)
        """
        if not self.device:
            print("Device not connected")
            return None, None, None, None

        try:
            capture = self.device.get_capture()

            # Get color image (BGRA -> BGR)
            color = capture.color
            if color is not None:
                color = color[:, :, :3]  # Remove alpha channel

            # Get depth image
            depth = capture.depth

            # Get IR image
            ir = capture.ir

            # Get aligned depth if requested
            aligned_depth = None
            if align_depth_to_color:
                try:
                    aligned_depth = capture.transformed_depth
                except:
                    aligned_depth = None

            return color, depth, ir, aligned_depth

        except Exception as e:
            print(f"Failed to capture frame: {e}")
            return None, None, None, None


    def get_point_cloud(self, color_image: np.ndarray, depth_image: np.ndarray) -> o3d.geometry.PointCloud:
        """
        Generate colored point cloud from color and depth images
        """
        if not self.device:
            raise RuntimeError("Device not connected")

        # Get camera intrinsics
        intrinsics = self.calibration.get_camera_matrix(pyk4a.CalibrationType.COLOR)

        # Create point cloud
        height, width = depth_image.shape

        # Generate mesh grid
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))

        # Convert to 3D points
        z = depth_image.astype(float) / 1000.0  # Convert mm to meters
        x = (xx - intrinsics[0, 2]) * z / intrinsics[0, 0]
        y = (yy - intrinsics[1, 2]) * z / intrinsics[1, 1]

        # Stack points
        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        colors = color_image.reshape(-1, 3) / 255.0

        # Remove invalid points (z=0)
        valid_mask = points[:, 2] > 0
        points = points[valid_mask]
        colors = colors[valid_mask]

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    def get_transformed_point_cloud(self) -> Optional[o3d.geometry.PointCloud]:
        """
        Get point cloud with depth aligned to color camera
        """
        if not self.device:
            print("Device not connected")
            return None

        try:
            capture = self.device.get_capture()

            # Get transformed depth (aligned to color camera)
            depth_transformed = capture.transformed_depth
            color = capture.color[:, :, :3]  # Remove alpha

            if depth_transformed is not None and color is not None:
                return self.get_point_cloud(color, depth_transformed)

            return None

        except Exception as e:
            print(f"Failed to get transformed point cloud: {e}")
            return None

    def save_point_cloud(self, filename: str, pcd: o3d.geometry.PointCloud) -> bool:
        """Save point cloud to file (PLY, PCD, etc.)"""
        try:
            o3d.io.write_point_cloud(filename, pcd)
            print(f"Saved point cloud to {filename}")
            return True
        except Exception as e:
            print(f"Failed to save point cloud: {e}")
            return False


class CameraIntrinsicCalibration:
    """Handle camera intrinsic calibration using chessboard or ArUco markers"""

    def __init__(self):
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibration_images = []
        self.object_points = []  # 3D points in real world
        self.image_points = []   # 2D points in image plane

    def calibrate_with_chessboard(self,
                                  images: list,
                                  board_size: tuple = (9, 6),
                                  square_size: float = 0.025) -> bool:
        """
        Calibrate camera using chessboard pattern

        Args:
            images: List of calibration images (color images)
            board_size: Number of internal corners (width, height)
            square_size: Size of chessboard square in meters
        """
        # Prepare object points (0,0,0), (1,0,0), (2,0,0) ...
        objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
        objp *= square_size

        object_points = []
        image_points = []

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        print(f"Processing {len(images)} calibration images...")

        for i, img in enumerate(images):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, board_size, None)

            if ret:
                object_points.append(objp)

                # Refine corner positions
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                image_points.append(corners_refined)

                print(f"  Image {i+1}: Chessboard found")
            else:
                print(f"  Image {i+1}: Chessboard not found")

        if len(object_points) < 3:
            print("Insufficient calibration images. Need at least 3 with detected chessboard.")
            return False

        # Calibrate camera
        print(f"\nCalibrating with {len(object_points)} images...")
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points, gray.shape[::-1], None, None
        )

        if ret:
            self.camera_matrix = camera_matrix
            self.dist_coeffs = dist_coeffs
            self.object_points = object_points
            self.image_points = image_points

            print("\nCalibration successful!")
            print(f"Camera matrix:\n{camera_matrix}")
            print(f"Distortion coefficients: {dist_coeffs.ravel()}")
            return True
        else:
            print("Calibration failed")
            return False

    def calibrate_with_aruco(self,
                            images: list,
                            board_size: tuple = (5, 7),
                            square_length: float = 0.04,
                            marker_length: float = 0.03,
                            dictionary_type=cv2.aruco.DICT_6X6_250) -> bool:
        """
        Calibrate camera using ArUco ChArUco board

        Args:
            images: List of calibration images
            board_size: Number of chessboard squares (width, height)
            square_length: Chessboard square side length in meters
            marker_length: ArUco marker side length in meters
            dictionary_type: ArUco dictionary type
        """
        # Create ArUco dictionary and CharUco board
        aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_type)
        charuco_board = cv2.aruco.CharucoBoard(
            board_size, square_length, marker_length, aruco_dict
        )

        all_charuco_corners = []
        all_charuco_ids = []

        print(f"Processing {len(images)} ArUco calibration images...")

        for i, img in enumerate(images):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect ArUco markers
            detector_params = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
            corners, ids, rejected = detector.detectMarkers(gray)

            if ids is not None and len(ids) > 0:
                # Interpolate ChArUco corners
                ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    corners, ids, gray, charuco_board
                )

                if ret:
                    all_charuco_corners.append(charuco_corners)
                    all_charuco_ids.append(charuco_ids)
                    print(f"  Image {i+1}: Found {len(charuco_ids)} ChArUco corners")
                else:
                    print(f"  Image {i+1}: ChArUco interpolation failed")
            else:
                print(f"  Image {i+1}: No ArUco markers detected")

        if len(all_charuco_corners) < 3:
            print("Insufficient calibration images. Need at least 3 with detected markers.")
            return False

        # Calibrate camera
        print(f"\nCalibrating with {len(all_charuco_corners)} images...")
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            all_charuco_corners, all_charuco_ids, charuco_board, gray.shape[::-1], None, None
        )

        if ret:
            self.camera_matrix = camera_matrix
            self.dist_coeffs = dist_coeffs

            print("\nCalibration successful!")
            print(f"Camera matrix:\n{camera_matrix}")
            print(f"Distortion coefficients: {dist_coeffs.ravel()}")
            return True
        else:
            print("Calibration failed")
            return False

    def save_calibration(self, filename: str) -> bool:
        """Save intrinsic calibration to file"""
        if self.camera_matrix is None:
            print("No calibration to save")
            return False

        try:
            np.savez(filename,
                    camera_matrix=self.camera_matrix,
                    dist_coeffs=self.dist_coeffs)
            print(f"Saved intrinsic calibration to {filename}")
            return True
        except Exception as e:
            print(f"Failed to save calibration: {e}")
            return False

    def load_calibration(self, filename: str) -> bool:
        """Load intrinsic calibration from file"""
        try:
            data = np.load(filename)
            self.camera_matrix = data['camera_matrix']
            self.dist_coeffs = data['dist_coeffs']
            print(f"Loaded intrinsic calibration from {filename}")
            return True
        except Exception as e:
            print(f"Failed to load calibration: {e}")
            return False


class HandEyeCalibration:
    """
    Handle hand-eye calibration for robot-camera transformation
    Uses OpenCV's hand-eye calibration methods
    """

    def __init__(self):
        self.camera_to_gripper = None  # Transformation from camera to robot gripper
        self.base_to_camera = None     # Transformation from robot base to camera

    def calibrate_eye_in_hand(self,
                             robot_poses: list,
                             marker_poses: list,
                             method=cv2.CALIB_HAND_EYE_TSAI) -> np.ndarray:
        """
        Eye-in-hand calibration (camera mounted on robot)

        Args:
            robot_poses: List of 4x4 transformation matrices (gripper to robot base)
            marker_poses: List of 4x4 transformation matrices (marker to camera)
            method: Hand-eye calibration method (TSAI, PARK, HORAUD, ANDREFF, DANIILIDIS)

        Returns:
            4x4 transformation matrix (camera to gripper)
        """
        if len(robot_poses) != len(marker_poses):
            raise ValueError("Number of robot poses must match marker poses")

        if len(robot_poses) < 3:
            raise ValueError("Need at least 3 pose pairs for calibration")

        # Convert to rotation and translation vectors
        R_gripper2base = []
        t_gripper2base = []
        R_marker2cam = []
        t_marker2cam = []

        for robot_pose, marker_pose in zip(robot_poses, marker_poses):
            # Robot gripper to base
            R_gripper2base.append(robot_pose[:3, :3])
            t_gripper2base.append(robot_pose[:3, 3].reshape(3, 1))

            # Marker to camera
            R_marker2cam.append(marker_pose[:3, :3])
            t_marker2cam.append(marker_pose[:3, 3].reshape(3, 1))

        # Perform hand-eye calibration
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper2base, t_gripper2base,
            R_marker2cam, t_marker2cam,
            method=method
        )

        # Build transformation matrix
        self.camera_to_gripper = np.eye(4)
        self.camera_to_gripper[:3, :3] = R_cam2gripper
        self.camera_to_gripper[:3, 3] = t_cam2gripper.flatten()

        print("\nHand-eye calibration complete!")
        print("Camera to gripper transformation:")
        print(self.camera_to_gripper)

        return self.camera_to_gripper

    def calibrate_eye_to_hand(self,
                             robot_poses: list,
                             marker_poses: list,
                             method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH) -> np.ndarray:
        """
        Eye-to-hand calibration (camera fixed in world)

        Args:
            robot_poses: List of 4x4 transformation matrices (gripper to robot base)
            marker_poses: List of 4x4 transformation matrices (marker to camera)
            method: Robot-world hand-eye calibration method (SHAH, LI)

        Returns:
            4x4 transformation matrix (robot base to camera)
        """
        if len(robot_poses) != len(marker_poses):
            raise ValueError("Number of robot poses must match marker poses")

        if len(robot_poses) < 3:
            raise ValueError("Need at least 3 pose pairs for calibration")

        # Convert to rotation and translation vectors
        R_gripper2base = []
        t_gripper2base = []
        R_marker2cam = []
        t_marker2cam = []

        for robot_pose, marker_pose in zip(robot_poses, marker_poses):
            R_gripper2base.append(robot_pose[:3, :3])
            t_gripper2base.append(robot_pose[:3, 3].reshape(3, 1))

            R_marker2cam.append(marker_pose[:3, :3])
            t_marker2cam.append(marker_pose[:3, 3].reshape(3, 1))

        # Perform robot-world hand-eye calibration
        R_base2cam, t_base2cam = cv2.calibrateRobotWorldHandEye(
            R_gripper2base, t_gripper2base,
            R_marker2cam, t_marker2cam,
            method=method
        )

        # Build transformation matrix
        self.base_to_camera = np.eye(4)
        self.base_to_camera[:3, :3] = R_base2cam
        self.base_to_camera[:3, 3] = t_base2cam.flatten()

        print("\nRobot-world hand-eye calibration complete!")
        print("Robot base to camera transformation:")
        print(self.base_to_camera)

        return self.base_to_camera

    def transform_point_camera_to_robot(self, camera_point: np.ndarray, current_gripper_pose: np.ndarray = None) -> np.ndarray:
        """
        Transform point from camera to robot base coordinates

        Args:
            camera_point: 3D point in camera frame
            current_gripper_pose: Current gripper pose (for eye-in-hand)
        """
        point_homo = np.append(camera_point, 1)

        if self.camera_to_gripper is not None and current_gripper_pose is not None:
            # Eye-in-hand: camera -> gripper -> base
            point_gripper = self.camera_to_gripper @ point_homo
            point_base = current_gripper_pose @ point_gripper
            return point_base[:3]

        elif self.base_to_camera is not None:
            # Eye-to-hand: camera -> base
            T_cam2base = np.linalg.inv(self.base_to_camera)
            point_base = T_cam2base @ point_homo
            return point_base[:3]

        else:
            raise RuntimeError("No calibration available")

    def save_calibration(self, filename: str) -> bool:
        """Save hand-eye calibration to file"""
        data = {}
        if self.camera_to_gripper is not None:
            data['camera_to_gripper'] = self.camera_to_gripper
        if self.base_to_camera is not None:
            data['base_to_camera'] = self.base_to_camera

        if not data:
            print("No calibration to save")
            return False

        try:
            np.savez(filename, **data)
            print(f"Saved hand-eye calibration to {filename}")
            return True
        except Exception as e:
            print(f"Failed to save calibration: {e}")
            return False

    def load_calibration(self, filename: str) -> bool:
        """Load hand-eye calibration from file"""
        try:
            data = np.load(filename)
            if 'camera_to_gripper' in data:
                self.camera_to_gripper = data['camera_to_gripper']
            if 'base_to_camera' in data:
                self.base_to_camera = data['base_to_camera']
            print(f"Loaded hand-eye calibration from {filename}")
            return True
        except Exception as e:
            print(f"Failed to load calibration: {e}")
            return False


class KinectViewer:
    """
    Real-time viewer for Azure Kinect similar to k4aviewer
    Displays color, depth, IR, and point cloud streams
    """

    def __init__(self, kinect: AzureKinect):
        self.kinect = kinect
        self.running = False

        # Visualization settings
        self.depth_min = 500  # mm
        self.depth_max = 5000  # mm
        self.colormap = cv2.COLORMAP_JET
        self.colormap_name = "JET"

        # Available colormaps
        self.colormaps = {
            'JET': cv2.COLORMAP_JET,
            'HSV': cv2.COLORMAP_HSV,
            'HOT': cv2.COLORMAP_HOT,
            'COOL': cv2.COLORMAP_COOL,
            'RAINBOW': cv2.COLORMAP_RAINBOW,
            'VIRIDIS': cv2.COLORMAP_VIRIDIS,
            'PLASMA': cv2.COLORMAP_PLASMA,
            'MAGMA': cv2.COLORMAP_MAGMA
        }
        self.colormap_list = list(self.colormaps.keys())
        self.colormap_index = 0

        # Depth modes
        self.depth_modes = {
            'NFOV Unbinned': DepthMode.NFOV_UNBINNED,
            'NFOV Binned': DepthMode.NFOV_2X2BINNED,
            'WFOV Unbinned': DepthMode.WFOV_UNBINNED,
            'WFOV Binned': DepthMode.WFOV_2X2BINNED,
            'Passive IR': DepthMode.PASSIVE_IR
        }
        self.depth_mode_names = list(self.depth_modes.keys())
        self.depth_mode_index = 0
        self.current_depth_mode = 'NFOV Unbinned'

        # Display modes
        self.view_mode = 'all'  # 'all', 'color', 'depth', 'ir'

        # Statistics
        self.fps = 0
        self.frame_count = 0
        self.start_time = None
        self.last_frame_time = None

        # Point cloud visualization
        self.show_pointcloud = False
        self.pointcloud_mode = 'rgb'  # 'rgb' or 'depth'
        self.point_size = 2.0
        self.vis_3d = None
        self.pcd = None
        self.pointcloud_initialized = False

        # Video recording
        self.is_recording = False
        self.video_writer_color = None
        self.video_writer_depth = None
        self.recording_start_time = None

        # Window name
        self.window_name = "Azure Kinect Viewer"

    def _change_depth_mode(self):
        """Change depth mode and reconnect camera with proper FPS"""
        self.depth_mode_index = (self.depth_mode_index + 1) % len(self.depth_mode_names)
        self.current_depth_mode = self.depth_mode_names[self.depth_mode_index]

        print(f"\nChanging depth mode to: {self.current_depth_mode}")
        print("Reconnecting camera...")

        # Disconnect
        self.kinect.disconnect()

        # Update config
        self.kinect.config.depth_mode = self.depth_modes[self.current_depth_mode]

        # Set appropriate FPS based on depth mode
        # WFOV Unbinned only supports 15 FPS, others support 30 FPS
        if self.current_depth_mode == 'WFOV Unbinned':
            self.kinect.config.camera_fps = FPS.FPS_15
            print("  FPS: 15 (required for WFOV Unbinned)")
        else:
            self.kinect.config.camera_fps = FPS.FPS_30
            print("  FPS: 30")

        # Reconnect
        if self.kinect.connect():
            print(f"Switched to {self.current_depth_mode}")

            # Reset point cloud if active
            if self.show_pointcloud and self.vis_3d:
                self.vis_3d.destroy_window()
                self.vis_3d = None
                self.pointcloud_initialized = False
        else:
            print("Failed to reconnect camera")
            self.running = False

    def _start_recording(self):
        """Start video recording"""
        if self.is_recording:
            return

        import os
        from datetime import datetime

        # Create recordings directory
        os.makedirs("kinect_recordings", exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Video settings
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = 30.0

        # Initialize video writers
        color_path = f"kinect_recordings/color_{timestamp}.avi"
        depth_path = f"kinect_recordings/depth_{timestamp}.avi"

        self.video_writer_color = cv2.VideoWriter(color_path, fourcc, fps, (1920, 1080))
        self.video_writer_depth = cv2.VideoWriter(depth_path, fourcc, fps, (640, 576))

        self.is_recording = True
        self.recording_start_time = time.time()

        print(f"Recording started: {color_path}")

    def _stop_recording(self):
        """Stop video recording"""
        if not self.is_recording:
            return

        if self.video_writer_color:
            self.video_writer_color.release()
            self.video_writer_color = None

        if self.video_writer_depth:
            self.video_writer_depth.release()
            self.video_writer_depth = None

        self.is_recording = False
        elapsed = time.time() - self.recording_start_time
        print(f"Recording stopped. Duration: {elapsed:.1f}s")

    def _draw_info_overlay(self, image: np.ndarray, title: str, extra_info: dict = None) -> np.ndarray:
        """Draw information overlay on image"""
        overlay = image.copy()
        h, w = overlay.shape[:2]

        # Semi-transparent background for text
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)

        # Title
        cv2.putText(image, title, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # FPS
        cv2.putText(image, f"FPS: {self.fps:.1f}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Depth mode
        cv2.putText(image, f"Depth: {self.current_depth_mode}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # Recording indicator
        if self.is_recording:
            elapsed = time.time() - self.recording_start_time
            cv2.putText(image, f"REC {elapsed:.1f}s", (w - 120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Red dot
            cv2.circle(image, (w - 140, 25), 8, (0, 0, 255), -1)

        # Extra info
        if extra_info:
            y_offset = 90
            for key, value in extra_info.items():
                text = f"{key}: {value}"
                cv2.putText(image, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                y_offset += 20

        return image

    def _draw_controls_help(self, image: np.ndarray) -> np.ndarray:
        """Draw control instructions"""
        h, w = image.shape[:2]

        controls = [
            "Controls:",
            "1-4: View modes",
            "M: Depth mode",
            "C: Colormap",
            "P: 3D mode",
            "R: RGB/Depth (3D)",
            "+/-: Size/Range",
            "V: Record",
            "S: Snapshot",
            "Q: Quit"
        ]

        # Semi-transparent background
        overlay = image.copy()
        text_height = len(controls) * 18 + 10
        cv2.rectangle(overlay, (w - 200, h - text_height - 10), (w - 5, h - 5), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)

        # Draw controls
        y_offset = h - text_height
        for i, control in enumerate(controls):
            color = (100, 200, 255) if i == 0 else (220, 220, 220)
            cv2.putText(image, control, (w - 190, y_offset + i * 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        return image

    def _process_depth_for_display(self, depth_image: np.ndarray) -> np.ndarray:
        """Process depth image for visualization"""
        # Clip to range
        depth_clipped = depth_image.copy()
        depth_clipped[depth_clipped < self.depth_min] = 0
        depth_clipped[depth_clipped > self.depth_max] = 0

        # Normalize to 0-255
        if self.depth_max > self.depth_min:
            depth_normalized = cv2.normalize(
                depth_clipped, None, 0, 255,
                cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
        else:
            depth_normalized = np.zeros_like(depth_clipped, dtype=np.uint8)

        # Apply colormap
        depth_colormap = cv2.applyColorMap(depth_normalized, self.colormap)

        # Add depth info overlay
        extra_info = {
            'Min Depth': f'{self.depth_min}mm',
            'Max Depth': f'{self.depth_max}mm',
            'Colormap': self.colormap_name
        }
        depth_colormap = self._draw_info_overlay(depth_colormap, "Depth", extra_info)

        return depth_colormap

    def _process_ir_for_display(self, ir_image: np.ndarray) -> np.ndarray:
        """Process IR image for visualization"""
        if ir_image is None:
            return np.zeros((576, 640, 3), dtype=np.uint8)

        # Normalize IR to 0-255
        ir_normalized = cv2.normalize(ir_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Convert to BGR for display
        ir_bgr = cv2.cvtColor(ir_normalized, cv2.COLOR_GRAY2BGR)

        # Add overlay
        ir_bgr = self._draw_info_overlay(ir_bgr, "Infrared", {})

        return ir_bgr

    def _create_combined_view(self, color: np.ndarray, depth: np.ndarray, ir: np.ndarray) -> np.ndarray:
        """Create a 2x2 grid view of all streams"""
        # Resize images to consistent size
        target_size = (640, 480)

        # Process color
        if color is not None:
            color_resized = cv2.resize(color, target_size)
            color_display = self._draw_info_overlay(color_resized, "Color (1920x1080)", {})
        else:
            color_display = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

        # Process depth
        if depth is not None:
            depth_display = self._process_depth_for_display(depth)
            depth_display = cv2.resize(depth_display, target_size)
        else:
            depth_display = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

        # Process IR
        if ir is not None:
            ir_display = self._process_ir_for_display(ir)
            ir_display = cv2.resize(ir_display, target_size)
        else:
            ir_display = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

        # Create info panel
        info_panel = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        info_lines = [
            ("Azure Kinect Viewer", 40, (0, 255, 255), 0.8),
            (f"FPS: {self.fps:.1f}", 80, (0, 255, 0), 0.6),
            (f"Frames: {self.frame_count}", 110, (0, 255, 0), 0.6),
            ("", 140, (255, 255, 255), 0.5),
            ("View Modes:", 170, (255, 255, 0), 0.5),
            ("  1: All views", 195, (200, 200, 200), 0.5),
            ("  2: Color only", 220, (200, 200, 200), 0.5),
            ("  3: Depth only", 245, (200, 200, 200), 0.5),
            ("  4: IR only", 270, (200, 200, 200), 0.5),
            ("", 300, (255, 255, 255), 0.5),
            ("C: Change colormap", 330, (200, 200, 200), 0.5),
            ("+/-: Depth range", 355, (200, 200, 200), 0.5),
            ("P: Point cloud", 380, (200, 200, 200), 0.5),
            ("S: Save snapshot", 405, (200, 200, 200), 0.5),
            ("Q: Quit", 430, (200, 200, 200), 0.5),
        ]

        for text, y, color, scale in info_lines:
            cv2.putText(info_panel, text, (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2 if scale > 0.6 else 1)

        # Combine into 2x2 grid
        top_row = np.hstack([color_display, depth_display])
        bottom_row = np.hstack([ir_display, info_panel])
        combined = np.vstack([top_row, bottom_row])

        return combined

    def _update_point_cloud(self, color: np.ndarray, aligned_depth: np.ndarray):
        """Update point cloud visualization with integrated controls"""
        try:
            if aligned_depth is None:
                return

            # Generate point cloud using aligned depth (same size as color)
            pcd = self.kinect.get_point_cloud(color, aligned_depth)

            # Fix inversion: flip Y axis (camera coordinate system)
            points = np.asarray(pcd.points)
            points[:, 1] = -points[:, 1]  # Flip Y to fix upside-down
            pcd.points = o3d.utility.Vector3dVector(points)

            # Apply coloring based on mode
            if self.pointcloud_mode == 'depth':
                # Depth-based coloring (blue=close, red=far)
                depths = points[:, 2]  # Z values

                # Normalize depths to 0-1 range
                min_depth = np.min(depths[depths > 0]) if np.any(depths > 0) else 0
                max_depth = np.max(depths)

                if max_depth > min_depth:
                    normalized_depths = (depths - min_depth) / (max_depth - min_depth)

                    # Create color map: blue (close) to red (far)
                    colors = np.zeros((len(depths), 3))
                    colors[:, 0] = normalized_depths  # Red channel
                    colors[:, 2] = 1.0 - normalized_depths  # Blue channel

                    pcd.colors = o3d.utility.Vector3dVector(colors)

            # Initialize visualizer if needed
            if not self.pointcloud_initialized:
                self.vis_3d = o3d.visualization.Visualizer()

                # Create window with controls info in title
                title = f"Point Cloud [{self.current_depth_mode}] | R:RGB/Depth | +/- Size"
                self.vis_3d.create_window(window_name=title, width=1280, height=960)

                self.pcd = pcd
                self.vis_3d.add_geometry(self.pcd)

                # Set view options
                opt = self.vis_3d.get_render_option()
                opt.point_size = self.point_size
                opt.background_color = np.asarray([0.05, 0.05, 0.05])
                opt.show_coordinate_frame = True

                self.pointcloud_initialized = True
            else:
                # Update existing point cloud
                self.pcd.points = pcd.points
                self.pcd.colors = pcd.colors
                self.vis_3d.update_geometry(self.pcd)

            # Always update render options (for point size changes)
            opt = self.vis_3d.get_render_option()
            opt.point_size = self.point_size

            # Update visualization
            self.vis_3d.poll_events()
            self.vis_3d.update_renderer()

        except Exception as e:
            print(f"Point cloud update error: {e}")

    def _save_snapshot(self, color: np.ndarray, depth: np.ndarray, ir: np.ndarray):
        """Save current frame snapshot"""
        import os
        from datetime import datetime

        # Create snapshot directory
        snapshot_dir = "kinect_snapshots"
        os.makedirs(snapshot_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save color
        if color is not None:
            color_path = os.path.join(snapshot_dir, f"color_{timestamp}.jpg")
            cv2.imwrite(color_path, color)

        # Save depth
        if depth is not None:
            depth_path = os.path.join(snapshot_dir, f"depth_{timestamp}.png")
            cv2.imwrite(depth_path, depth)

        # Save IR
        if ir is not None:
            ir_path = os.path.join(snapshot_dir, f"ir_{timestamp}.png")
            cv2.imwrite(ir_path, ir)

        print(f"Snapshot saved: {timestamp}")

    def run(self):
        """Run the viewer"""
        if not self.kinect.device:
            print("Kinect not connected. Call kinect.connect() first.")
            return

        # Print factory intrinsics
        intrinsics = self.kinect.get_camera_intrinsics()
        if intrinsics:
            print("\n" + "="*60)
            print("Factory Camera Intrinsics")
            print("="*60)
            print(f"Color Camera:")
            print(f"  fx: {intrinsics['color']['fx']:.2f}, fy: {intrinsics['color']['fy']:.2f}")
            print(f"  cx: {intrinsics['color']['cx']:.2f}, cy: {intrinsics['color']['cy']:.2f}")
            print(f"Depth Camera:")
            print(f"  fx: {intrinsics['depth']['fx']:.2f}, fy: {intrinsics['depth']['fy']:.2f}")
            print(f"  cx: {intrinsics['depth']['cx']:.2f}, cy: {intrinsics['depth']['cy']:.2f}")

        print("\n" + "="*60)
        print("Azure Kinect Viewer")
        print("="*60)
        print("2D Mode Controls:")
        print("  1-4: Switch view modes (All/Color/Depth/IR)")
        print("  M: Change depth mode (NFOV Binned/Unbinned, WFOV Binned/Unbinned, Passive IR)")
        print("  C: Cycle depth colormaps")
        print("  +/-: Adjust depth range visualization")
        print("\n3D Mode (Point Cloud):")
        print("  P: Toggle 3D mode (hides 2D view)")
        print("  R: Toggle RGB colors / Depth gradient (blue=close, red=far)")
        print("  +/-: Adjust point size (0.5 to 10.0)")
        print("\nRecording:")
        print("  V: Start/Stop video recording")
        print("  S: Save snapshot (color + depth + IR)")
        print("\nQ/ESC/X: Quit")
        print("="*60)

        self.running = True
        self.start_time = time.time()
        self.frame_count = 0
        self.last_frame_time = time.time()

        # Create main window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        try:
            while self.running:
                # Check if window was closed (X button)
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    self.running = False
                    break

                # Capture frame - only get aligned depth if point cloud is enabled for better FPS
                color, depth, ir, aligned_depth = self.kinect.capture_frame(align_depth_to_color=self.show_pointcloud)

                if color is None and depth is None:
                    continue

                # Update frame count and FPS (instant FPS, not average)
                current_time = time.time()
                if self.last_frame_time:
                    frame_time = current_time - self.last_frame_time
                    if frame_time > 0:
                        self.fps = 1.0 / frame_time
                self.last_frame_time = current_time
                self.frame_count += 1

                # Record video if enabled
                if self.is_recording:
                    if self.video_writer_color and color is not None:
                        self.video_writer_color.write(color)
                    if self.video_writer_depth and depth is not None:
                        depth_vis = self._process_depth_for_display(depth)
                        depth_vis = cv2.resize(depth_vis, (640, 576))
                        self.video_writer_depth.write(depth_vis)

                # In point cloud mode, show minimal control panel
                if self.show_pointcloud:
                    # Create small control panel for keyboard input
                    control_panel = np.zeros((300, 400, 3), dtype=np.uint8)

                    cv2.putText(control_panel, "3D MODE ACTIVE", (70, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                    cv2.putText(control_panel, f"FPS: {self.fps:.1f}", (10, 80),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    cv2.putText(control_panel, f"Depth Mode: {self.current_depth_mode}", (10, 110),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

                    cv2.putText(control_panel, f"Point Size: {self.point_size:.1f}", (10, 140),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                    cv2.putText(control_panel, f"Mode: {self.pointcloud_mode.upper()}", (10, 170),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                    cv2.putText(control_panel, "Controls:", (10, 210),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
                    cv2.putText(control_panel, "P: Back to 2D", (10, 235),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                    cv2.putText(control_panel, "R: RGB/Depth", (10, 255),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                    cv2.putText(control_panel, "+/-: Point size", (10, 275),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

                    cv2.imshow(self.window_name, control_panel)

                    # Update point cloud (less frequently for performance)
                    if color is not None and aligned_depth is not None and self.frame_count % 2 == 0:
                        self._update_point_cloud(color, aligned_depth)
                else:
                    # Normal 2D view mode
                    # Create display based on mode
                    if self.view_mode == 'all':
                        display = self._create_combined_view(color, depth, ir)
                    elif self.view_mode == 'color':
                        display = color.copy() if color is not None else np.zeros((1080, 1920, 3), dtype=np.uint8)
                        display = self._draw_info_overlay(display, "Color Stream", {})
                        display = self._draw_controls_help(display)
                    elif self.view_mode == 'depth':
                        if depth is not None:
                            display = self._process_depth_for_display(depth)
                            display = cv2.resize(display, (1280, 960))
                        else:
                            display = np.zeros((960, 1280, 3), dtype=np.uint8)
                        display = self._draw_controls_help(display)
                    elif self.view_mode == 'ir':
                        display = self._process_ir_for_display(ir)
                        display = cv2.resize(display, (1280, 960))
                        display = self._draw_controls_help(display)

                    # Show display
                    cv2.imshow(self.window_name, display)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:  # q or ESC
                    self.running = False
                elif key == ord('1'):
                    self.view_mode = 'all'
                    print("View mode: All")
                elif key == ord('2'):
                    self.view_mode = 'color'
                    print("View mode: Color")
                elif key == ord('3'):
                    self.view_mode = 'depth'
                    print("View mode: Depth")
                elif key == ord('4'):
                    self.view_mode = 'ir'
                    print("View mode: IR")
                elif key == ord('m'):
                    self._change_depth_mode()
                elif key == ord('c'):
                    self.colormap_index = (self.colormap_index + 1) % len(self.colormap_list)
                    self.colormap_name = self.colormap_list[self.colormap_index]
                    self.colormap = self.colormaps[self.colormap_name]
                    print(f"Colormap: {self.colormap_name}")
                elif key == ord('p'):
                    self.show_pointcloud = not self.show_pointcloud
                    if self.show_pointcloud:
                        print("\n3D Mode ENABLED - 2D window hidden")
                        print("  Press R to toggle RGB/Depth coloring")
                        print("  Press +/- to adjust point size")
                        print("  Press P again to return to 2D mode")
                    else:
                        print("3D Mode disabled - returning to 2D view")
                        if self.vis_3d:
                            self.vis_3d.destroy_window()
                            self.vis_3d = None
                            self.pointcloud_initialized = False
                elif key == ord('r'):
                    # Toggle point cloud coloring mode
                    if self.show_pointcloud:
                        self.pointcloud_mode = 'depth' if self.pointcloud_mode == 'rgb' else 'rgb'
                        mode_name = "Depth Gradient (Blue=Close, Red=Far)" if self.pointcloud_mode == 'depth' else "RGB Colors"
                        print(f"Point cloud mode: {mode_name}")
                    else:
                        print("Enable point cloud first (press P)")
                elif key == ord('=') or key == ord('+'):  # = key (increase)
                    if self.show_pointcloud:
                        # Adjust point cloud size
                        self.point_size = min(10.0, self.point_size + 0.5)
                        print(f"Point size: {self.point_size:.1f}")
                        # Force render option update
                        if self.vis_3d:
                            opt = self.vis_3d.get_render_option()
                            opt.point_size = self.point_size
                    else:
                        # Adjust depth range
                        self.depth_max = min(self.depth_max + 500, 10000)
                        print(f"Depth range: {self.depth_min}-{self.depth_max}mm")
                elif key == ord('-') or key == ord('_'):  # - key (decrease)
                    if self.show_pointcloud:
                        # Adjust point cloud size
                        self.point_size = max(0.5, self.point_size - 0.5)
                        print(f"Point size: {self.point_size:.1f}")
                        # Force render option update
                        if self.vis_3d:
                            opt = self.vis_3d.get_render_option()
                            opt.point_size = self.point_size
                    else:
                        # Adjust depth range
                        self.depth_max = max(self.depth_max - 500, self.depth_min + 500)
                        print(f"Depth range: {self.depth_min}-{self.depth_max}mm")
                elif key == ord('v'):
                    if self.is_recording:
                        self._stop_recording()
                    else:
                        self._start_recording()
                elif key == ord('s'):
                    self._save_snapshot(color, depth, ir)

        except KeyboardInterrupt:
            print("\nViewer interrupted by user")

        finally:
            # Stop recording if active
            if self.is_recording:
                self._stop_recording()

            # Cleanup
            cv2.destroyAllWindows()
            if self.vis_3d:
                try:
                    self.vis_3d.destroy_window()
                except:
                    pass

            print(f"\nViewer closed. Total frames: {self.frame_count}")


class DepthProcessor:
    """Process depth images for object detection and segmentation"""

    @staticmethod
    def segment_plane(depth_image: np.ndarray,
                     distance_threshold: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment dominant plane (e.g., table) from depth image

        Returns:
            plane_mask: Binary mask of plane pixels
            objects_mask: Binary mask of non-plane pixels (potential objects)
        """
        # Convert depth to point cloud
        h, w = depth_image.shape
        fx = fy = w  # Approximate focal length
        cx, cy = w // 2, h // 2

        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        z = depth_image.astype(float) / 1000.0
        x = (xx - cx) * z / fx
        y = (yy - cy) * z / fy

        # Stack into points
        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

        # Remove invalid points
        valid_mask = z.reshape(-1) > 0
        valid_points = points[valid_mask]

        if len(valid_points) < 100:
            return np.zeros_like(depth_image, dtype=bool), np.zeros_like(depth_image, dtype=bool)

        # Use RANSAC to fit plane
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(valid_points)

        plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold/1000.0,
                                                 ransac_n=3,
                                                 num_iterations=1000)

        # Create masks
        plane_mask_1d = np.zeros(len(points), dtype=bool)
        plane_mask_1d[valid_mask] = np.array([i in inliers for i in range(len(valid_points))])
        plane_mask = plane_mask_1d.reshape(h, w)

        objects_mask = ~plane_mask & (depth_image > 0)

        return plane_mask, objects_mask

    @staticmethod
    def find_objects(depth_image: np.ndarray,
                    min_size: int = 100,
                    max_distance: float = 2000.0) -> List[Tuple[int, int, int, int]]:
        """
        Find objects in depth image

        Returns:
            List of bounding boxes (x, y, width, height)
        """
        # Threshold depth
        mask = (depth_image > 100) & (depth_image < max_distance)

        # Find connected components
        num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))

        boxes = []
        for i in range(1, num_labels):
            component_mask = (labels == i)

            # Check size
            if np.sum(component_mask) < min_size:
                continue

            # Get bounding box
            y_coords, x_coords = np.where(component_mask)
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)

            boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))

        return boxes


# Helper functions for calibration

def capture_calibration_images(kinect: AzureKinect, num_images: int = 20, output_dir: str = "calibration_images") -> list:
    """
    Interactive calibration image capture

    Args:
        kinect: Connected Azure Kinect device
        num_images: Number of images to capture
        output_dir: Directory to save images

    Returns:
        List of captured images
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    images = []
    count = 0

    print(f"\nCalibration Image Capture")
    print(f"Target: {num_images} images")
    print("Press SPACE to capture, 'q' to quit")

    cv2.namedWindow("Calibration Capture", cv2.WINDOW_NORMAL)

    while count < num_images:
        color, depth, ir, _ = kinect.capture_frame()

        if color is not None:
            # Display with counter
            display = color.copy()
            cv2.putText(display, f"Captured: {count}/{num_images}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display, "Press SPACE to capture",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Calibration Capture", cv2.resize(display, (960, 540)))

        key = cv2.waitKey(30) & 0xFF

        if key == ord(' '):  # Space bar
            if color is not None:
                images.append(color.copy())
                # Save image
                filename = os.path.join(output_dir, f"calib_{count:03d}.jpg")
                cv2.imwrite(filename, color)
                print(f"Captured image {count+1}/{num_images}")
                count += 1

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    print(f"\nCaptured {len(images)} calibration images")
    return images


def detect_aruco_pose(image: np.ndarray,
                     camera_matrix: np.ndarray,
                     dist_coeffs: np.ndarray,
                     marker_size: float = 0.05,
                     dictionary_type=cv2.aruco.DICT_6X6_250) -> dict:
    """
    Detect ArUco markers and estimate their pose

    Args:
        image: Input image
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        marker_size: Size of ArUco marker in meters
        dictionary_type: ArUco dictionary type

    Returns:
        Dictionary with marker IDs and their poses
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_type)
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    marker_poses = {}

    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            # Estimate pose
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners[i]], marker_size, camera_matrix, dist_coeffs
            )

            # Convert to transformation matrix
            R, _ = cv2.Rodrigues(rvec[0])
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = tvec[0].flatten()

            marker_poses[marker_id] = {
                'rvec': rvec[0],
                'tvec': tvec[0],
                'transform': T,
                'corners': corners[i]
            }

    return marker_poses


def calibration_workflow_chessboard(kinect: AzureKinect,
                                   board_size: tuple = (9, 6),
                                   square_size: float = 0.025,
                                   num_images: int = 20) -> CameraIntrinsicCalibration:
    """
    Complete chessboard calibration workflow

    Args:
        kinect: Connected Azure Kinect device
        board_size: Chessboard internal corners (width, height)
        square_size: Square size in meters
        num_images: Number of calibration images to capture

    Returns:
        Calibrated CameraIntrinsicCalibration object
    """
    print("="*60)
    print("Chessboard Calibration Workflow")
    print("="*60)
    print(f"Board size: {board_size[0]}x{board_size[1]} internal corners")
    print(f"Square size: {square_size}m")
    print(f"Target images: {num_images}")
    print("\nInstructions:")
    print("- Move the chessboard to different positions and angles")
    print("- Cover different areas of the image")
    print("- Press SPACE when chessboard is clearly visible")
    print("="*60)

    # Capture calibration images
    images = capture_calibration_images(kinect, num_images)

    if len(images) < 3:
        print("Insufficient images captured")
        return None

    # Perform calibration
    calib = CameraIntrinsicCalibration()
    success = calib.calibrate_with_chessboard(images, board_size, square_size)

    if success:
        # Save calibration
        calib.save_calibration("camera_calibration.npz")
        print("\nCalibration complete and saved!")

    return calib if success else None


def hand_eye_calibration_workflow(kinect: AzureKinect,
                                  robot_controller,
                                  camera_calib: CameraIntrinsicCalibration,
                                  marker_size: float = 0.05,
                                  num_poses: int = 10,
                                  eye_in_hand: bool = True) -> HandEyeCalibration:
    """
    Complete hand-eye calibration workflow

    Args:
        kinect: Connected Azure Kinect device
        robot_controller: Robot controller object with get_tcp_pose() method
        camera_calib: Camera intrinsic calibration
        marker_size: ArUco marker size in meters
        num_poses: Number of calibration poses
        eye_in_hand: True for eye-in-hand, False for eye-to-hand

    Returns:
        Calibrated HandEyeCalibration object
    """
    print("="*60)
    print("Hand-Eye Calibration Workflow")
    print("="*60)
    print(f"Mode: {'Eye-in-hand' if eye_in_hand else 'Eye-to-hand'}")
    print(f"Target poses: {num_poses}")
    print("\nInstructions:")
    print("- Move robot to different positions")
    print("- Ensure ArUco marker is visible in camera")
    print("- Press SPACE to capture pose pair")
    print("="*60)

    robot_poses = []
    marker_poses = []
    count = 0

    cv2.namedWindow("Hand-Eye Calibration", cv2.WINDOW_NORMAL)

    while count < num_poses:
        color, depth, ir, _ = kinect.capture_frame()

        if color is not None:
            # Detect ArUco marker
            marker_data = detect_aruco_pose(
                color,
                camera_calib.camera_matrix,
                camera_calib.dist_coeffs,
                marker_size
            )

            # Display
            display = color.copy()
            cv2.putText(display, f"Poses: {count}/{num_poses}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if marker_data:
                cv2.putText(display, f"Marker detected: {list(marker_data.keys())}",
                          (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # Draw markers
                for marker_id, data in marker_data.items():
                    cv2.drawFrameAxes(display, camera_calib.camera_matrix, camera_calib.dist_coeffs,
                                     data['rvec'], data['tvec'], marker_size * 0.5)
            else:
                cv2.putText(display, "No marker detected",
                          (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(display, "Press SPACE to capture",
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Hand-Eye Calibration", cv2.resize(display, (960, 540)))

        key = cv2.waitKey(30) & 0xFF

        if key == ord(' '):  # Space bar
            if marker_data:
                # Get robot pose
                robot_pose = robot_controller.get_tcp_pose()  # Should return 4x4 matrix

                # Use first detected marker
                marker_id = list(marker_data.keys())[0]
                marker_pose = marker_data[marker_id]['transform']

                robot_poses.append(robot_pose)
                marker_poses.append(marker_pose)

                print(f"Captured pose pair {count+1}/{num_poses}")
                count += 1
            else:
                print("No marker detected, capture skipped")

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

    if len(robot_poses) < 3:
        print("Insufficient pose pairs captured")
        return None

    # Perform hand-eye calibration
    hand_eye_calib = HandEyeCalibration()

    if eye_in_hand:
        hand_eye_calib.calibrate_eye_in_hand(robot_poses, marker_poses)
    else:
        hand_eye_calib.calibrate_eye_to_hand(robot_poses, marker_poses)

    # Save calibration
    hand_eye_calib.save_calibration("hand_eye_calibration.npz")
    print("\nHand-eye calibration complete and saved!")

    return hand_eye_calib


# Example usage functions
def test_kinect_connection():
    """Test basic Kinect functionality"""
    kinect = AzureKinect()

    if kinect.connect():
        print("Successfully connected to Azure Kinect")

        # Get and print intrinsics
        intrinsics = kinect.get_camera_intrinsics()
        if intrinsics:
            print("\nFactory Intrinsics:")
            print(f"Color: fx={intrinsics['color']['fx']:.2f}, fy={intrinsics['color']['fy']:.2f}")
            print(f"Depth: fx={intrinsics['depth']['fx']:.2f}, fy={intrinsics['depth']['fy']:.2f}")

        # Capture a few frames
        print("\nCapturing frames...")
        for i in range(5):
            color, depth, ir, aligned_depth = kinect.capture_frame(align_depth_to_color=False)
            if color is not None:
                print(f"Frame {i+1}: Color shape: {color.shape}, Depth shape: {depth.shape}")
            time.sleep(0.5)

        kinect.disconnect()
    else:
        print("Failed to connect to Azure Kinect")


def capture_point_cloud(save_path: str = "point_cloud.ply"):
    """Capture and save a point cloud"""
    kinect = AzureKinect()

    if kinect.connect():
        print("Capturing point cloud...")

        # Get transformed point cloud
        pcd = kinect.get_transformed_point_cloud()

        if pcd:
            # Save point cloud
            kinect.save_point_cloud(save_path, pcd)

            # Visualize (optional)
            o3d.visualization.draw_geometries([pcd])

        kinect.disconnect()


def launch_viewer():
    """Launch the interactive Kinect viewer"""
    kinect = AzureKinect()

    if kinect.connect():
        viewer = KinectViewer(kinect)
        viewer.run()
        kinect.disconnect()
    else:
        print("Failed to connect to Azure Kinect")


if __name__ == "__main__":
    print("Azure Kinect DK")
    print("Select an option:")
    print("1. Launch Interactive Viewer (k4aviewer-like)")
    print("2. Test connection")
    print("3. Capture and save point cloud")
    print("4. Camera calibration (chessboard)")
    print("0. Exit")

    choice = input("\nEnter choice: ").strip()

    if choice == "1":
        launch_viewer()
    elif choice == "2":
        test_kinect_connection()
    elif choice == "3":
        capture_point_cloud()
    elif choice == "4":
        kinect = AzureKinect()
        if kinect.connect():
            calibration_workflow_chessboard(kinect)
            kinect.disconnect()
    elif choice == "0":
        print("Exiting...")
    else:
        print("Invalid choice. Run the script again.")