"""
RGB Image to Point Cloud Conversion
Creates 3D point clouds from RGB images using structure-from-motion or multi-view stereo
"""

import numpy as np
import cv2
import open3d as o3d
from typing import List, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt


class RGBToPointCloud:
    """Convert RGB images to 3D point clouds using various methods"""

    def __init__(self):
        self.images = []
        self.camera_params = []
        self.pcd = None

    def create_from_stereo_pair(self,
                               left_image: np.ndarray,
                               right_image: np.ndarray,
                               focal_length: float,
                               baseline: float,
                               min_disparity: int = 0,
                               num_disparities: int = 64,
                               block_size: int = 11) -> o3d.geometry.PointCloud:
        """
        Create point cloud from stereo image pair

        Args:
            left_image: Left camera RGB image
            right_image: Right camera RGB image
            focal_length: Camera focal length in pixels
            baseline: Distance between cameras in meters
            min_disparity: Minimum disparity value
            num_disparities: Maximum disparity minus minimum disparity (must be divisible by 16)
            block_size: Size of the block for matching (odd number, typically 5-21)

        Returns:
            Point cloud from stereo reconstruction
        """
        print("Computing stereo disparity...")

        # Convert to grayscale
        left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

        # Create stereo matcher
        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disparity,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * 3 * block_size ** 2,
            P2=32 * 3 * block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        # Compute disparity
        disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

        # Filter invalid disparities
        disparity[disparity <= min_disparity] = 0.1
        disparity[disparity > num_disparities] = 0.1

        # Compute depth from disparity: depth = (focal_length * baseline) / disparity
        depth = (focal_length * baseline) / (disparity + 1e-6)

        # Create point cloud
        h, w = depth.shape
        cx, cy = w / 2, h / 2

        # Generate mesh grid
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))

        # Compute 3D coordinates
        z = depth
        x = (xx - cx) * z / focal_length
        y = (yy - cy) * z / focal_length

        # Stack points
        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        colors = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB).reshape(-1, 3) / 255.0

        # Filter invalid points
        valid_mask = (z.reshape(-1) > 0.1) & (z.reshape(-1) < 10.0)
        points = points[valid_mask]
        colors = colors[valid_mask]

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Flip for correct orientation
        pcd.transform([[1, 0, 0, 0],
                       [0, -1, 0, 0],
                       [0, 0, -1, 0],
                       [0, 0, 0, 1]])

        self.pcd = pcd
        print(f"Created point cloud: {len(pcd.points)} points")

        return pcd

    def create_from_multiview_sfm(self,
                                  image_paths: List[str],
                                  camera_intrinsics: Optional[np.ndarray] = None) -> o3d.geometry.PointCloud:
        """
        Create point cloud from multiple RGB images using Structure from Motion (SfM)

        This is a simplified SfM pipeline. For production use, consider:
        - COLMAP (state-of-the-art SfM)
        - OpenMVG
        - AliceVision

        Args:
            image_paths: List of paths to RGB images
            camera_intrinsics: 3x3 camera matrix (if None, will estimate)

        Returns:
            Point cloud from multi-view reconstruction
        """
        print(f"\nStructure from Motion with {len(image_paths)} images")
        print("Note: This is a simplified SfM. For better results, use COLMAP or OpenMVG")

        if len(image_paths) < 2:
            raise ValueError("Need at least 2 images for SfM")

        # Load images
        images = []
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                print(f"Warning: Could not load {path}")
                continue
            images.append(img)

        if len(images) < 2:
            raise ValueError("Failed to load sufficient images")

        # Feature detection and matching
        print("\n1. Detecting features...")
        sift = cv2.SIFT_create()
        keypoints_list = []
        descriptors_list = []

        for i, img in enumerate(images):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, desc = sift.detectAndCompute(gray, None)
            keypoints_list.append(kp)
            descriptors_list.append(desc)
            print(f"   Image {i}: {len(kp)} keypoints")

        # Match features between image pairs
        print("\n2. Matching features...")
        bf = cv2.BFMatcher()
        matches_list = []

        for i in range(len(images) - 1):
            matches = bf.knnMatch(descriptors_list[i], descriptors_list[i+1], k=2)

            # Lowe's ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

            matches_list.append(good_matches)
            print(f"   Image {i} <-> {i+1}: {len(good_matches)} good matches")

        # Estimate camera poses and triangulate points
        print("\n3. Estimating camera poses...")

        # For simplicity, use first two images
        img1, img2 = images[0], images[1]
        kp1, kp2 = keypoints_list[0], keypoints_list[1]
        matches = matches_list[0]

        # Get matched points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        # Estimate camera intrinsics if not provided
        if camera_intrinsics is None:
            h, w = img1.shape[:2]
            focal_length = max(w, h)
            camera_intrinsics = np.array([
                [focal_length, 0, w/2],
                [0, focal_length, h/2],
                [0, 0, 1]
            ])

        # Estimate essential matrix
        E, mask = cv2.findEssentialMat(pts1, pts2, camera_intrinsics, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        pts1 = pts1[mask.ravel() == 1]
        pts2 = pts2[mask.ravel() == 1]

        # Recover pose
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, camera_intrinsics)
        pts1 = pts1[mask.ravel() > 0]
        pts2 = pts2[mask.ravel() > 0]

        print(f"   Inliers: {len(pts1)}")

        # Triangulate points
        print("\n4. Triangulating 3D points...")

        # Projection matrices
        P1 = camera_intrinsics @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = camera_intrinsics @ np.hstack([R, t])

        # Triangulate
        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points_3d = (points_4d[:3] / points_4d[3]).T

        # Filter points behind camera or too far
        valid_mask = (points_3d[:, 2] > 0) & (points_3d[:, 2] < 100)
        points_3d = points_3d[valid_mask]
        pts1 = pts1[valid_mask]

        # Get colors from first image
        colors = []
        for pt in pts1:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < img1.shape[1] and 0 <= y < img1.shape[0]:
                color = img1[y, x][::-1] / 255.0  # BGR to RGB
                colors.append(color)
            else:
                colors.append([0, 0, 0])

        colors = np.array(colors)

        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        self.pcd = pcd
        print(f"\nCreated point cloud: {len(pcd.points)} points")

        return pcd

    def create_from_depth_estimation(self,
                                    rgb_image: np.ndarray,
                                    depth_estimator: str = 'midas') -> o3d.geometry.PointCloud:
        """
        Create point cloud from single RGB image using monocular depth estimation

        Requires additional dependencies:
        - For MiDaS: torch, torchvision
        - pip install torch torchvision

        Args:
            rgb_image: RGB image
            depth_estimator: 'midas' or 'dpt' (Deep depth estimation models)

        Returns:
            Point cloud from estimated depth
        """
        try:
            import torch
        except ImportError:
            print("Error: PyTorch not installed. Install with: pip install torch torchvision")
            return None

        print(f"Using {depth_estimator.upper()} for depth estimation...")

        # Load MiDaS model
        if depth_estimator == 'midas':
            # Load a pretrained model
            model_type = "DPT_Large"  # Options: DPT_Large, DPT_Hybrid, MiDaS_small
            midas = torch.hub.load("intel-isl/MiDaS", model_type)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            midas.to(device)
            midas.eval()

            # Load transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            transform = midas_transforms.dpt_transform

            # Prepare image
            img_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            input_batch = transform(img_rgb).to(device)

            # Predict depth
            print("Estimating depth...")
            with torch.no_grad():
                prediction = midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            depth = prediction.cpu().numpy()

            # Normalize depth (MiDaS outputs inverse depth)
            depth = depth.max() - depth
            depth = depth / depth.max()

            # Convert to metric depth (approximate)
            depth = depth * 10.0  # Scale to ~10 meters max

        else:
            raise ValueError(f"Unknown depth estimator: {depth_estimator}")

        # Create point cloud
        h, w = depth.shape
        focal_length = max(h, w)
        cx, cy = w / 2, h / 2

        xx, yy = np.meshgrid(np.arange(w), np.arange(h))

        z = depth
        x = (xx - cx) * z / focal_length
        y = (yy - cy) * z / focal_length

        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        colors = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB).reshape(-1, 3) / 255.0

        # Filter invalid points
        valid_mask = points[:, 2] > 0.1
        points = points[valid_mask]
        colors = colors[valid_mask]

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Flip for correct orientation
        pcd.transform([[1, 0, 0, 0],
                       [0, -1, 0, 0],
                       [0, 0, -1, 0],
                       [0, 0, 0, 1]])

        self.pcd = pcd
        print(f"Created point cloud: {len(pcd.points)} points")

        return pcd

    def visualize_disparity(self, left_image: np.ndarray, right_image: np.ndarray):
        """
        Visualize stereo disparity map

        Args:
            left_image: Left camera image
            right_image: Right camera image
        """
        # Convert to grayscale
        left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

        # Create stereo matcher
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=64,
            blockSize=11
        )

        # Compute disparity
        disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

        # Visualize
        plt.figure(figsize=(15, 5))

        plt.subplot(131)
        plt.imshow(cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB))
        plt.title("Left Image")
        plt.axis('off')

        plt.subplot(132)
        plt.imshow(cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB))
        plt.title("Right Image")
        plt.axis('off')

        plt.subplot(133)
        plt.imshow(disparity, cmap='jet')
        plt.title("Disparity Map")
        plt.colorbar()
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    def save(self, filename: str) -> bool:
        """Save point cloud to file"""
        if self.pcd is None:
            print("No point cloud to save")
            return False

        success = o3d.io.write_point_cloud(filename, self.pcd)
        if success:
            print(f"Saved point cloud to {filename}")
        return success

    def visualize(self, window_name: str = "Point Cloud from RGB"):
        """Visualize the point cloud"""
        if self.pcd is None:
            print("No point cloud to visualize")
            return

        # Add coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0]
        )

        o3d.visualization.draw_geometries(
            [self.pcd, coord_frame],
            window_name=window_name,
            point_show_normal=False
        )


def demo_stereo_reconstruction(left_path: str, right_path: str):
    """
    Demo stereo reconstruction from image pair

    Args:
        left_path: Path to left camera image
        right_path: Path to right camera image
    """
    print("\n" + "="*60)
    print("Demo: Stereo Reconstruction")
    print("="*60)

    # Load images
    left_img = cv2.imread(left_path)
    right_img = cv2.imread(right_path)

    if left_img is None or right_img is None:
        print("Error: Could not load images")
        return

    # Create converter
    converter = RGBToPointCloud()

    # Visualize disparity first
    print("\nVisualizing disparity map...")
    converter.visualize_disparity(left_img, right_img)

    # Create point cloud
    # These are example parameters - adjust based on your stereo setup
    focal_length = 700.0  # pixels
    baseline = 0.1  # meters (10 cm between cameras)

    pcd = converter.create_from_stereo_pair(
        left_img, right_img,
        focal_length=focal_length,
        baseline=baseline,
        num_disparities=64,
        block_size=11
    )

    # Post-process
    print("\nPost-processing...")
    pcd = pcd.voxel_down_sample(voxel_size=0.01)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

    # Visualize
    converter.visualize("Stereo Point Cloud")

    # Save
    converter.save("stereo_pointcloud.ply")


def demo_multiview_sfm(image_dir: str):
    """
    Demo multi-view structure from motion

    Args:
        image_dir: Directory containing sequence of RGB images
    """
    print("\n" + "="*60)
    print("Demo: Multi-View Structure from Motion")
    print("="*60)

    # Get all images
    image_paths = sorted(Path(image_dir).glob("*.jpg")) + sorted(Path(image_dir).glob("*.png"))
    image_paths = [str(p) for p in image_paths]

    if len(image_paths) < 2:
        print(f"Error: Need at least 2 images in {image_dir}")
        return

    print(f"Found {len(image_paths)} images")

    # Create converter
    converter = RGBToPointCloud()

    # Run SfM
    pcd = converter.create_from_multiview_sfm(image_paths)

    # Post-process
    print("\nPost-processing...")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.voxel_down_sample(voxel_size=0.05)

    # Visualize
    converter.visualize("Multi-View Point Cloud")

    # Save
    converter.save("multiview_pointcloud.ply")


def demo_monocular_depth(image_path: str):
    """
    Demo monocular depth estimation

    Args:
        image_path: Path to RGB image
    """
    print("\n" + "="*60)
    print("Demo: Monocular Depth Estimation")
    print("="*60)

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return

    # Create converter
    converter = RGBToPointCloud()

    # Estimate depth and create point cloud
    try:
        pcd = converter.create_from_depth_estimation(img, depth_estimator='midas')

        if pcd is not None:
            # Post-process
            print("\nPost-processing...")
            pcd = pcd.voxel_down_sample(voxel_size=0.02)

            # Visualize
            converter.visualize("Monocular Depth Point Cloud")

            # Save
            converter.save("monocular_pointcloud.ply")
    except Exception as e:
        print(f"Error during depth estimation: {e}")
        print("Make sure PyTorch is installed: pip install torch torchvision")


if __name__ == "__main__":
    print("RGB to Point Cloud Conversion")
    print("\nSelect a method:")
    print("1. Stereo reconstruction (requires stereo image pair)")
    print("2. Multi-view SfM (requires multiple images of same scene)")
    print("3. Monocular depth estimation (single RGB image, requires PyTorch)")
    print("0. Exit")

    choice = input("\nEnter choice: ").strip()

    if choice == "1":
        left = input("Enter left image path: ").strip()
        right = input("Enter right image path: ").strip()
        demo_stereo_reconstruction(left, right)

    elif choice == "2":
        image_dir = input("Enter directory with image sequence: ").strip()
        demo_multiview_sfm(image_dir)

    elif choice == "3":
        image_path = input("Enter RGB image path: ").strip()
        demo_monocular_depth(image_path)

    elif choice == "0":
        print("Exiting...")
    else:
        print("Invalid choice")
