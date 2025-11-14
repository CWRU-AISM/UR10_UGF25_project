"""
3D Geometry Processing and Visualization
Handles point cloud creation, normals, denoising, downsampling, and mesh operations using Open3D
"""

import numpy as np
import open3d as o3d
import cv2
from typing import Optional, Tuple, Union
from pathlib import Path


class PointCloudProcessor:
    """Process and visualize point clouds with various operations"""

    def __init__(self):
        self.pcd = None

    def create_from_depth_rgb(self,
                             depth_image: np.ndarray,
                             rgb_image: np.ndarray,
                             fx: float, fy: float,
                             cx: float, cy: float,
                             depth_scale: float = 1000.0,
                             depth_trunc: float = 3.0) -> o3d.geometry.PointCloud:
        """
        Create point cloud from depth and RGB images using camera intrinsics
        """
        # Convert to Open3D images
        if rgb_image.shape[2] == 3 and rgb_image.dtype == np.uint8:
            # Assuming BGR from OpenCV, convert to RGB
            rgb_o3d = o3d.geometry.Image(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
        else:
            rgb_o3d = o3d.geometry.Image(rgb_image)

        depth_o3d = o3d.geometry.Image(depth_image)

        # Create RGBD image
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d,
            depth_o3d,
            depth_scale=depth_scale,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False
        )

        # Create camera intrinsic matrix
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=rgb_image.shape[1],
            height=rgb_image.shape[0],
            fx=fx, fy=fy,
            cx=cx, cy=cy
        )

        # Create point cloud from RGBD image
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

        # Flip it for correct visualization (camera coordinate convention)
        pcd.transform([[1, 0, 0, 0],
                       [0, -1, 0, 0],
                       [0, 0, -1, 0],
                       [0, 0, 0, 1]])

        self.pcd = pcd
        return pcd

    def create_from_numpy(self,
                         points: np.ndarray,
                         colors: Optional[np.ndarray] = None) -> o3d.geometry.PointCloud:
        """
        Create point cloud from numpy arrays

        Args:
            points: Nx3 numpy array of 3D points
            colors: Nx3 numpy array of RGB colors (0-1 range)

        Returns:
            Open3D PointCloud object
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if colors is not None:
            if colors.max() > 1.0:
                colors = colors / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)

        self.pcd = pcd
        return pcd

    def create_from_azure_kinect(self, kinect, align_depth: bool = True) -> o3d.geometry.PointCloud:
        """
        Create point cloud directly from Azure Kinect capture

        Args:
            kinect: AzureKinect instance (from azure_kinect.py)
            align_depth: Whether to align depth to color camera

        Returns:
            Open3D PointCloud object
        """
        if align_depth:
            pcd = kinect.get_transformed_point_cloud()
        else:
            color, depth, _, _ = kinect.capture_frame()
            if color is not None and depth is not None:
                pcd = kinect.get_point_cloud(color, depth)
            else:
                return None

        self.pcd = pcd
        return pcd

    def estimate_normals(self,
                        search_param: Optional[Union[int, float]] = None,
                        method: str = 'knn') -> o3d.geometry.PointCloud:
        """
        Estimate normals for the point cloud
        """
        if self.pcd is None:
            raise ValueError("No point cloud loaded. Create one first.")

        if method == 'knn':
            k = search_param if search_param is not None else 30
            self.pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k)
            )
        elif method == 'hybrid':
            radius = search_param if search_param is not None else 0.1
            self.pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=radius, max_nn=30
                )
            )
        else:
            raise ValueError("Method must be 'knn' or 'hybrid'")

        # Orient normals consistently (towards camera/viewpoint)
        self.pcd.orient_normals_consistent_tangent_plane(k=15)

        return self.pcd

    def denoise_statistical(self,
                           nb_neighbors: int = 20,
                           std_ratio: float = 2.0) -> o3d.geometry.PointCloud:
        """
        Remove statistical outliers from point cloud
        """
        if self.pcd is None:
            raise ValueError("No point cloud loaded. Create one first.")

        cleaned_pcd, ind = self.pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )

        print(f"Statistical outlier removal: {len(self.pcd.points)} -> {len(cleaned_pcd.points)} points")

        self.pcd = cleaned_pcd
        return self.pcd

    def denoise_radius(self,
                      radius: float = 0.05,
                      min_neighbors: int = 10) -> o3d.geometry.PointCloud:
        """
        Remove points with few neighbors in a radius
        """
        if self.pcd is None:
            raise ValueError("No point cloud loaded. Create one first.")

        cleaned_pcd, ind = self.pcd.remove_radius_outlier(
            nb_points=min_neighbors,
            radius=radius
        )

        print(f"Radius outlier removal: {len(self.pcd.points)} -> {len(cleaned_pcd.points)} points")

        self.pcd = cleaned_pcd
        return self.pcd

    def downsample_voxel(self, voxel_size: float = 0.01) -> o3d.geometry.PointCloud:
        """
        Downsample point cloud using voxel grid
        """
        if self.pcd is None:
            raise ValueError("No point cloud loaded. Create one first.")

        downsampled = self.pcd.voxel_down_sample(voxel_size=voxel_size)

        print(f"Voxel downsampling: {len(self.pcd.points)} -> {len(downsampled.points)} points")

        self.pcd = downsampled
        return self.pcd

    def downsample_uniform(self, every_k_points: int = 5) -> o3d.geometry.PointCloud:
        """
        Downsample by selecting every k-th point
        """
        if self.pcd is None:
            raise ValueError("No point cloud loaded. Create one first.")

        downsampled = self.pcd.uniform_down_sample(every_k_points=every_k_points)

        print(f"Uniform downsampling: {len(self.pcd.points)} -> {len(downsampled.points)} points")

        self.pcd = downsampled
        return self.pcd

    def visualize(self,
                 show_normals: bool = False,
                 normal_length: float = 0.02,
                 point_size: float = 1.0,
                 window_name: str = "Point Cloud Visualization") -> None:
        """
        Visualize the point cloud with optional normals
        """
        if self.pcd is None:
            raise ValueError("No point cloud loaded. Create one first.")

        geometries = [self.pcd]

        # Add coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0]
        )
        geometries.append(coord_frame)

        # Visualize with custom settings
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name)

        for geom in geometries:
            vis.add_geometry(geom)

        # Set render options
        opt = vis.get_render_option()
        opt.point_size = point_size
        opt.background_color = np.asarray([0.1, 0.1, 0.1])
        opt.show_coordinate_frame = True

        if show_normals and self.pcd.has_normals():
            opt.point_show_normal = True

        vis.run()
        vis.destroy_window()

    def save(self, filename: str) -> bool:
        """
        Save point cloud to file (supports .ply, .pcd, .xyz, etc.)
        """
        if self.pcd is None:
            raise ValueError("No point cloud loaded. Create one first.")

        success = o3d.io.write_point_cloud(filename, self.pcd)
        if success:
            print(f"Saved point cloud to {filename}")
        else:
            print(f"Failed to save point cloud to {filename}")

        return success

    def load(self, filename: str) -> o3d.geometry.PointCloud:
        """
        Load point cloud from file

        Args:
            filename: Input filename

        Returns:
            Loaded point cloud
        """
        self.pcd = o3d.io.read_point_cloud(filename)
        print(f"Loaded point cloud: {len(self.pcd.points)} points")
        return self.pcd


class MeshProcessor:
    """Process and visualize 3D meshes"""

    def __init__(self):
        self.mesh = None

    def load_mesh(self, filename: str) -> o3d.geometry.TriangleMesh:
        """
        Load mesh from file (.obj, .ply, .stl, etc.)
        """
        if not Path(filename).exists():
            raise FileNotFoundError(f"Mesh file not found: {filename}")

        self.mesh = o3d.io.read_triangle_mesh(filename)
        self.mesh.compute_vertex_normals()

        print(f"Loaded mesh: {len(self.mesh.vertices)} vertices, {len(self.mesh.triangles)} triangles")
        return self.mesh

    def create_from_point_cloud(self,
                                pcd: o3d.geometry.PointCloud,
                                method: str = 'poisson') -> o3d.geometry.TriangleMesh:
        """
        Create mesh from point cloud using surface reconstruction
        """
        if not pcd.has_normals():
            print("Point cloud doesn't have normals. Estimating...")
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )

        if method == 'poisson':
            # Poisson surface reconstruction
            self.mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=9
            )

            # Remove low density vertices (outliers)
            vertices_to_remove = densities < np.quantile(densities, 0.01)
            self.mesh.remove_vertices_by_mask(vertices_to_remove)

        elif method == 'ball_pivoting':
            # Ball pivoting algorithm
            radii = [0.005, 0.01, 0.02, 0.04]
            self.mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd,
                o3d.utility.DoubleVector(radii)
            )
        else:
            raise ValueError("Method must be 'poisson' or 'ball_pivoting'")

        self.mesh.compute_vertex_normals()
        print(f"Created mesh: {len(self.mesh.vertices)} vertices, {len(self.mesh.triangles)} triangles")

        return self.mesh

    def apply_laplacian_smoothing(self,
                                  iterations: int = 5,
                                  lambda_filter: float = 0.5) -> o3d.geometry.TriangleMesh:
        """
        Apply Laplacian smoothing filter to mesh (each vertex towards avg of neighbors)
        """
        if self.mesh is None:
            raise ValueError("No mesh loaded. Load or create one first.")

        # Apply Laplacian smoothing
        self.mesh = self.mesh.filter_smooth_laplacian(
            number_of_iterations=iterations,
            lambda_filter=lambda_filter
        )

        self.mesh.compute_vertex_normals()
        print(f"Applied Laplacian smoothing: {iterations} iterations, lambda={lambda_filter}")

        return self.mesh

    def apply_taubin_smoothing(self,
                               iterations: int = 10,
                               lambda_filter: float = 0.5,
                               mu: float = -0.53) -> o3d.geometry.TriangleMesh:
        """
        Apply Taubin smoothing filter (better at preserving volume than Laplacian)
        """
        if self.mesh is None:
            raise ValueError("No mesh loaded. Load or create one first.")

        self.mesh = self.mesh.filter_smooth_taubin(
            number_of_iterations=iterations,
            lambda_filter=lambda_filter,
            mu=mu
        )

        self.mesh.compute_vertex_normals()
        print(f"Applied Taubin smoothing: {iterations} iterations")

        return self.mesh

    def simplify_mesh(self,
                     target_triangles: int = 10000) -> o3d.geometry.TriangleMesh:
        """
        Simplify mesh by reducing number of triangles
        """
        if self.mesh is None:
            raise ValueError("No mesh loaded. Load or create one first.")

        simplified = self.mesh.simplify_quadric_decimation(
            target_number_of_triangles=target_triangles
        )

        print(f"Mesh simplification: {len(self.mesh.triangles)} -> {len(simplified.triangles)} triangles")

        self.mesh = simplified
        self.mesh.compute_vertex_normals()

        return self.mesh

    def visualize(self,
                 show_wireframe: bool = False,
                 window_name: str = "Mesh Visualization") -> None:
        """
        Visualize the mesh
        """
        if self.mesh is None:
            raise ValueError("No mesh loaded. Load or create one first.")

        geometries = [self.mesh]

        # Add coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0]
        )
        geometries.append(coord_frame)

        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name)

        for geom in geometries:
            vis.add_geometry(geom)

        # Set render options
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.1, 0.1, 0.1])
        opt.mesh_show_wireframe = show_wireframe
        opt.mesh_show_back_face = True

        vis.run()
        vis.destroy_window()

    def save(self, filename: str) -> bool:
        """
        Save mesh to file

        """
        if self.mesh is None:
            raise ValueError("No mesh loaded. Load or create one first.")

        success = o3d.io.write_triangle_mesh(filename, self.mesh)
        if success:
            print(f"Saved mesh to {filename}")
        else:
            print(f"Failed to save mesh to {filename}")

        return success


def demo_point_cloud_from_numpy():
    # Create synthetic point cloud (sphere)
    n_points = 5000
    theta = np.random.uniform(0, 2*np.pi, n_points)
    phi = np.random.uniform(0, np.pi, n_points)
    r = 0.5

    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    points = np.column_stack([x, y, z])

    # Create colors (gradient based on Z)
    colors = np.zeros((n_points, 3))
    colors[:, 0] = (z + r) / (2*r)  # Red gradient
    colors[:, 2] = 1.0 - (z + r) / (2*r)  # Blue gradient

    # Process
    processor = PointCloudProcessor()
    processor.create_from_numpy(points, colors)
    processor.estimate_normals(search_param=30, method='knn')

    print(f"Created sphere point cloud: {len(points)} points")

    # Visualize with normals
    processor.visualize(show_normals=True, normal_length=0.05, point_size=2.0)


def demo_point_cloud_from_depth_rgb(depth_path: str, rgb_path: str):
    # Load images
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    rgb = cv2.imread(rgb_path)

    if depth is None or rgb is None:
        print("Error: Could not load images")
        return

    # Example intrinsics
    fx, fy = 920.0, 920.0
    cx, cy = rgb.shape[1] / 2, rgb.shape[0] / 2

    # Create point cloud
    processor = PointCloudProcessor()
    processor.create_from_depth_rgb(depth, rgb, fx, fy, cx, cy)

    print(f"Created point cloud from images: {len(processor.pcd.points)} points")

    # Denoise
    processor.denoise_statistical(nb_neighbors=20, std_ratio=2.0)

    # Downsample
    processor.downsample_voxel(voxel_size=0.005)

    # Estimate normals
    processor.estimate_normals(search_param=30, method='knn')

    # Visualize
    processor.visualize(show_normals=False, point_size=2.0)

    # Save
    processor.save("output_pointcloud.ply")


def demo_mesh_laplacian_filter(mesh_path: str):

    # Load mesh
    processor = MeshProcessor()
    processor.load_mesh(mesh_path)

    # Visualize original
    print("\nVisualizing original mesh")
    processor.visualize(show_wireframe=True, window_name="Original Mesh")

    # Apply Laplacian smoothing
    processor.apply_laplacian_smoothing(iterations=10, lambda_filter=0.5)

    # Visualize smoothed
    print("\nVisualizing smoothed mesh...")
    processor.visualize(show_wireframe=True, window_name="Smoothed Mesh (Laplacian)")

    # Save result
    processor.save("smoothed_mesh.ply")


def demo_voxel_downsampling():
    # Create dense point cloud (bunny shape approximation)
    n_points = 50000
    points = np.random.randn(n_points, 3) * 0.1
    points[:, 1] += 0.05  # Shift up

    # Create processor
    processor = PointCloudProcessor()
    processor.create_from_numpy(points)

    print(f"Original: {len(processor.pcd.points)} points")

    # Visualize original
    processor.visualize(point_size=1.0, window_name="Original Dense Point Cloud")

    # Apply voxel downsampling with different sizes
    for voxel_size in [0.01, 0.02, 0.05]:
        # Reload original
        processor.create_from_numpy(points)

        # Downsample
        processor.downsample_voxel(voxel_size=voxel_size)

        # Visualize
        processor.visualize(
            point_size=2.0,
            window_name=f"Voxel Downsampled (size={voxel_size})"
        )


if __name__ == "__main__":
    print("3D Geometry Processing Demos")
    print("\nSelect an option:")
    print("1. Point cloud from numpy arrays (synthetic sphere)")
    print("2. Point cloud from depth + RGB images")
    print("3. Mesh Laplacian filtering")
    print("4. Voxel downsampling demo")
    print("5. Point cloud denoising comparison")
    print("0. Exit")

    choice = input("\nEnter choice: ").strip()

    if choice == "1":
        demo_point_cloud_from_numpy()

    elif choice == "2":
        depth_path = input("Enter depth image path (16-bit PNG): ").strip()
        rgb_path = input("Enter RGB image path: ").strip()
        demo_point_cloud_from_depth_rgb(depth_path, rgb_path)

    elif choice == "3":
        print("\nMesh file locations:")
        print("Project meshes: ros2_ws/src/*/meshes/")
        print("Open3D samples: Use o3d.data.BunnyMesh() for testing")
        mesh_path = input("Enter mesh file path (.obj, .ply, .stl): ").strip()
        demo_mesh_laplacian_filter(mesh_path)

    elif choice == "4":
        demo_voxel_downsampling()

    elif choice == "5":
        # Create clean sphere
        n_points = 5000
        theta = np.random.uniform(0, 2*np.pi, n_points)
        phi = np.random.uniform(0, np.pi, n_points)
        r = 0.5

        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)

        points = np.column_stack([x, y, z])

        # Add noise
        noise = np.random.randn(n_points, 3) * 0.01
        noisy_points = points + noise

        # Add outliers
        n_outliers = 500
        outliers = np.random.randn(n_outliers, 3) * 2.0
        noisy_points = np.vstack([noisy_points, outliers])

        processor = PointCloudProcessor()
        processor.create_from_numpy(noisy_points)

        print("Visualizing noisy point cloud...")
        processor.visualize(window_name="Noisy Point Cloud")

        # Statistical denoising
        processor.create_from_numpy(noisy_points)
        processor.denoise_statistical(nb_neighbors=20, std_ratio=2.0)
        print("Visualizing statistical denoising...")
        processor.visualize(window_name="Statistical Denoising")

        # Radius denoising
        processor.create_from_numpy(noisy_points)
        processor.denoise_radius(radius=0.05, min_neighbors=10)
        print("Visualizing radius denoising...")
        processor.visualize(window_name="Radius Denoising")

    elif choice == "0":
        print("Exiting...")
    else:
        print("Invalid choice")
