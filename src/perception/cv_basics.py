"""
Azure Kinect + OpenCV Basics
"""

import cv2
import numpy as np
import pyk4a
from pyk4a import Config, PyK4A, ColorResolution, DepthMode, ImageFormat, FPS
import os
from datetime import datetime
import json
import open3d as o3d

# Basic Azure setup
class KinectCamera:
    """Wrapper class for Azure Kinect operations"""
    
    def __init__(self, device_id=0):
        """Initialize Azure Kinect with standard configuration"""
        self.device_id = device_id

        # Configure camera settings
        self.config = Config(
            color_format=ImageFormat.COLOR_BGRA32,       # BGRA format
            color_resolution=ColorResolution.RES_1080P,  # 1920x1080
            depth_mode=DepthMode.NFOV_UNBINNED,          # 640x576
            camera_fps=FPS.FPS_30,
            synchronized_images_only=True,
        )

        # Initialize camera
        self.camera = PyK4A(config=self.config, device_id=device_id)
        self.camera.start()

        # Get calibration data
        self.calibration = self.camera.calibration

        print(f"Kinect {device_id} initialized successfully!")
        print(f"Color resolution: {self.config.color_resolution}")
        print(f"Depth mode: {self.config.depth_mode}")
    
    def capture_frame(self):
        """Capture synchronized color and depth frames"""
        capture = self.camera.get_capture()
        
        if capture.color is not None and capture.depth is not None:
            # Convert BGRA to BGR for OpenCV
            color_image = capture.color[:, :, :3]
            depth_image = capture.depth
            
            return color_image, depth_image
        return None, None
    
    def get_point_cloud(self, depth_image):
        """Convert depth image to 3D point cloud using camera intrinsics"""
        # Get depth camera intrinsics
        intrinsics = self.calibration.get_camera_matrix(pyk4a.CalibrationType.DEPTH)

        # Get image dimensions
        height, width = depth_image.shape

        # Create mesh grid of pixel coordinates
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))

        # Convert depth to meters
        z = depth_image.astype(float) / 1000.0

        # Back-project to 3D using camera intrinsics
        # x = (u - cx) * z / fx
        # y = (v - cy) * z / fy
        x = (xx - intrinsics[0, 2]) * z / intrinsics[0, 0]
        y = (yy - intrinsics[1, 2]) * z / intrinsics[1, 1]

        # Stack into Nx3 array of 3D points
        points_3d = np.stack([x, y, z], axis=-1)

        return points_3d
    
    def close(self):
        """Clean up camera resources"""
        self.camera.stop()


# Basic image processing

def process_color_image(image):
    """Basic OpenCV processing on color image"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    return gray, blurred, edges


def process_depth_image(depth_image, min_depth=500, max_depth=2000):
    """Process and visualize depth data"""
    # Create a copy for processing
    depth_vis = depth_image.copy()
    
    # Clip depth values to range of interest
    depth_vis[depth_vis < min_depth] = 0
    depth_vis[depth_vis > max_depth] = 0
    
    # Normalize for visualization (0-255)
    depth_normalized = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Apply colormap for better visualization
    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    
    return depth_normalized, depth_colormap


# Data capture and saving

def save_frame_data(color_image, depth_image, frame_id, output_dir="captured_data"):
    """Save color and depth images with metadata"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save color image
    color_path = os.path.join(output_dir, f"color_{frame_id:04d}_{timestamp}.jpg")
    cv2.imwrite(color_path, color_image)
    
    # Save depth image (as 16-bit PNG to preserve depth values)
    depth_path = os.path.join(output_dir, f"depth_{frame_id:04d}_{timestamp}.png")
    cv2.imwrite(depth_path, depth_image)
    
    # Save metadata
    metadata = {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "color_path": color_path,
        "depth_path": depth_path,
        "depth_unit": "millimeters"
    }
    
    metadata_path = os.path.join(output_dir, f"metadata_{frame_id:04d}_{timestamp}.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata


# Multi-camera detection

def detect_kinect_devices():
    """Detect number of connected Azure Kinect devices"""
    device_count = pyk4a.connected_device_count()
    print(f"Found {device_count} Azure Kinect device(s)")

    devices = []
    for i in range(device_count):
        try:
            # Try to get device info
            temp_camera = PyK4A(device_id=i)
            # Note: serial is a property, not serial_number
            serial = temp_camera.serial if hasattr(temp_camera, 'serial') else f"Device_{i}"
            temp_camera.stop()
            devices.append({"id": i, "serial": serial})
            print(f"  Device {i}: Serial {serial}")
        except Exception as e:
            print(f"  Device {i}: Error - {e}")
            # Still add the device even if we can't get serial
            devices.append({"id": i, "serial": f"Device_{i}"})

    return devices


# Main demo

def demo_single_camera():
    """Demo: Single camera capture and processing"""
    
    # Initialize camera
    kinect = KinectCamera(device_id=0)
    
    # Create windows
    cv2.namedWindow("Color Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Depth Colormap", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Edges", cv2.WINDOW_NORMAL)
    
    frame_count = 0
    save_next = False
    
    print("Press 's' to save current frame, 'q' to quit")
    
    try:
        while True:
            # Capture frame
            color_image, depth_image = kinect.capture_frame()
            
            if color_image is not None and depth_image is not None:
                # Process images
                gray, blurred, edges = process_color_image(color_image)
                depth_norm, depth_colormap = process_depth_image(depth_image)
                
                # Display images
                cv2.imshow("Color Image", cv2.resize(color_image, (960, 540)))
                cv2.imshow("Depth Colormap", depth_colormap)
                cv2.imshow("Edges", edges)
                
                # Save frame if requested
                if save_next:
                    metadata = save_frame_data(color_image, depth_image, frame_count)
                    print(f"Saved frame {frame_count}: {metadata['timestamp']}")
                    frame_count += 1
                    save_next = False
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_next = True
    
    finally:
        kinect.close()
        cv2.destroyAllWindows()


def demo_frame_buffer():
    """Buffering frames for batch processing"""
    
    from collections import deque
    
    # Initialize camera
    kinect = KinectCamera(device_id=0)
    
    # Create frame buffer (stores last N frames)
    buffer_size = 30  # Store last 30 frames (1 second at 30fps)
    color_buffer = deque(maxlen=buffer_size)
    depth_buffer = deque(maxlen=buffer_size)
    
    print(f"Filling buffer with {buffer_size} frames...")
    
    try:
        # Fill buffer
        for i in range(buffer_size):
            color_image, depth_image = kinect.capture_frame()
            if color_image is not None and depth_image is not None:
                color_buffer.append(color_image)
                depth_buffer.append(depth_image)
                print(f"  Captured frame {i+1}/{buffer_size}")
        
        print("\nBuffer filled! Processing frames...")
        
        # Compute average depth over buffered frames
        depth_stack = np.stack(depth_buffer, axis=0)
        avg_depth = np.mean(depth_stack, axis=0)
        
        # Visualize average depth
        _, avg_depth_colormap = process_depth_image(avg_depth.astype(np.uint16))
        cv2.imshow("Average Depth (1 second)", avg_depth_colormap)
        cv2.waitKey(0)
        
        # Detect motion by comparing first and last frame
        frame_diff = cv2.absdiff(color_buffer[0], color_buffer[-1])
        gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
        _, motion_mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
        
        cv2.imshow("Motion Mask", motion_mask)
        cv2.waitKey(0)
    
    finally:
        kinect.close()
        cv2.destroyAllWindows()


def demo_point_cloud():
    """Generate and save point cloud from depth with visualization"""

    # Initialize camera
    kinect = KinectCamera(device_id=0)

    try:
        print("Capturing frame for point cloud...")
        color_image, depth_image = kinect.capture_frame()

        if depth_image is not None and color_image is not None:
            # Generate point cloud
            print("Generating point cloud...")
            point_cloud = kinect.get_point_cloud(depth_image)

            # Reshape to list of 3D points
            points_3d = point_cloud.reshape(-1, 3)

            # Remove invalid points (depth = 0)
            valid_mask = points_3d[:, 2] > 0
            valid_points = points_3d[valid_mask]

            print(f"Generated {len(valid_points)} valid 3D points")
            print(f"Point cloud range:")
            print(f"  X: {valid_points[:, 0].min():.3f} to {valid_points[:, 0].max():.3f} m")
            print(f"  Y: {valid_points[:, 1].min():.3f} to {valid_points[:, 1].max():.3f} m")
            print(f"  Z: {valid_points[:, 2].min():.3f} to {valid_points[:, 2].max():.3f} m")

            # Save as text file (XYZ format)
            output_file = "point_cloud.xyz"
            np.savetxt(output_file, valid_points, fmt='%.6f', header='X Y Z', comments='')
            print(f"\nSaved point cloud to {output_file}")

            # Create 3D point cloud visualizations
            print("\nCreating point cloud visualizations...")

            # Flip Y-axis to fix inversion
            valid_points[:, 1] = -valid_points[:, 1]

            # 1. Point cloud without color (geometry only)
            pcd_geometry = o3d.geometry.PointCloud()
            pcd_geometry.points = o3d.utility.Vector3dVector(valid_points)
            # Set uniform gray color
            pcd_geometry.paint_uniform_color([0.5, 0.5, 0.5])

            # 2. Point cloud with RGB colors
            pcd_rgb = o3d.geometry.PointCloud()
            pcd_rgb.points = o3d.utility.Vector3dVector(valid_points)

            # Get corresponding RGB colors for valid points
            # Resize color image to match depth image dimensions
            height, width = depth_image.shape
            color_resized = cv2.resize(color_image, (width, height))

            # Get valid colors matching valid depth points
            valid_mask_2d = (depth_image > 0).flatten()
            color_flat = color_resized.reshape(-1, 3)
            valid_colors = color_flat[valid_mask_2d]

            # Normalize colors to 0-1 range and convert BGR to RGB
            colors_normalized = valid_colors[:, ::-1] / 255.0  # BGR to RGB and normalize
            pcd_rgb.colors = o3d.utility.Vector3dVector(colors_normalized)

            # Save point clouds
            o3d.io.write_point_cloud("point_cloud_geometry.ply", pcd_geometry)
            o3d.io.write_point_cloud("point_cloud_rgb.ply", pcd_rgb)

            print("\nSaved point clouds:")
            print("  - point_cloud_geometry.ply (geometry only)")
            print("  - point_cloud_rgb.ply (with RGB colors)")

            # Display point cloud with geometry only
            print("\nDisplaying geometry-only point cloud...")
            print("Close the window to see RGB point cloud")
            o3d.visualization.draw_geometries(
                [pcd_geometry],
                window_name="Point Cloud - Geometry Only",
                width=1024,
                height=768
            )

            # Display point cloud with RGB colors
            print("\nDisplaying RGB point cloud...")
            print("Close the window to continue")
            o3d.visualization.draw_geometries(
                [pcd_rgb],
                window_name="Point Cloud - RGB Colors",
                width=1024,
                height=768
            )

    finally:
        kinect.close()
        cv2.destroyAllWindows()



def continuous_capture_loop(mode='images', interval_seconds=5.0, output_dir="continuous_capture"):
    """
    Continuous capture loop for long-running data collection
    """
    kinect = KinectCamera(device_id=0)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create display window
    cv2.namedWindow("Continuous Capture", cv2.WINDOW_NORMAL)

    if mode == 'video':
        # Video recording mode
        print("Starting continuous video recording...")
        print("Press 'q' to stop recording")

        # Get frame dimensions
        color_image, depth_image = kinect.capture_frame()
        if color_image is None:
            print("Failed to capture initial frame")
            kinect.close()
            return

        height, width = color_image.shape[:2]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create video writers
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        color_video_path = os.path.join(output_dir, f"color_video_{timestamp}.avi")
        depth_video_path = os.path.join(output_dir, f"depth_video_{timestamp}.avi")

        color_writer = cv2.VideoWriter(color_video_path, fourcc, 30.0, (width, height))
        depth_writer = cv2.VideoWriter(depth_video_path, fourcc, 30.0, depth_image.shape[::-1])

        print(f"Recording to: {color_video_path}")

        frame_count = 0
        start_time = datetime.now()

        try:
            while True:
                color_image, depth_image = kinect.capture_frame()

                if color_image is not None and depth_image is not None:
                    # Write frames
                    color_writer.write(color_image)

                    # Convert depth to 8-bit for video
                    depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                    depth_writer.write(depth_colormap)

                    frame_count += 1

                    # Display with info overlay
                    elapsed = (datetime.now() - start_time).total_seconds()
                    display_img = color_image.copy()
                    cv2.putText(display_img, f"Recording: {elapsed:.1f}s | Frames: {frame_count}",
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(display_img, "Press 'q' to stop", (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    cv2.imshow("Continuous Capture", cv2.resize(display_img, (960, 540)))

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        finally:
            color_writer.release()
            depth_writer.release()
            print(f"\nRecording stopped. Captured {frame_count} frames in {elapsed:.1f} seconds")

    elif mode == 'images':
        # Periodic image capture mode
        print(f"Starting periodic image capture (every {interval_seconds} seconds)...")
        print("Press 'q' to stop")

        frame_count = 0
        import time
        last_capture_time = time.time()

        try:
            while True:
                current_time = time.time()
                color_image, depth_image = kinect.capture_frame()

                if color_image is not None:
                    # Check if it's time to save
                    time_since_last = current_time - last_capture_time

                    # Display with countdown
                    display_img = color_image.copy()
                    time_until_next = max(0, interval_seconds - time_since_last)
                    cv2.putText(display_img, f"Next capture in: {time_until_next:.1f}s",
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display_img, f"Captured: {frame_count} frames",
                              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display_img, "Press 'q' to stop", (10, 90),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    cv2.imshow("Continuous Capture", cv2.resize(display_img, (960, 540)))

                    # Save if interval has passed
                    if time_since_last >= interval_seconds:
                        if depth_image is not None:
                            metadata = save_frame_data(color_image, depth_image, frame_count, output_dir)
                            print(f"Captured frame {frame_count}: {metadata['timestamp']}")
                            frame_count += 1
                            last_capture_time = current_time

                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    break

        finally:
            print(f"\nCapture stopped. Total frames captured: {frame_count}")

    else:
        print(f"Unknown mode: {mode}. Use 'images' or 'video'")

    kinect.close()
    cv2.destroyAllWindows()


def sync_frames():
    """Capture and save 10 synchronized frames"""
    kinect = KinectCamera(device_id=0)

    try:
        print("Capturing 10 frames with 1-second intervals...")

        for i in range(10):
            print(f"\nCapturing frame {i+1}/10...")
            color_image, depth_image = kinect.capture_frame()

            if color_image is not None and depth_image is not None:
                # Save frame data
                metadata = save_frame_data(color_image, depth_image, i)
                print(f"Saved: {metadata['timestamp']}")

                # Display current frame
                cv2.imshow("Current Frame", cv2.resize(color_image, (960, 540)))
                cv2.waitKey(100)  # Brief display
            else:
                print(f"Failed to capture frame {i+1}")

            # Wait 1 second before next capture (except for last frame)
            if i < 9:
                import time
                time.sleep(1.0)

        print("\All frames saved to 'captured_data' directory.")
        cv2.waitKey(2000)  # Show last frame for 2 seconds

    finally:
        kinect.close()
        cv2.destroyAllWindows()


def image_proc():
    """Apply custom image processing"""
    kinect = KinectCamera(device_id=0)

    # Create windows
    cv2.namedWindow("Color Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Depth Segmentation", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Color Filter (HSV)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Contours Detected", cv2.WINDOW_NORMAL)

    print("Custom Image Processing Pipeline")
    print("Press 'q' to quit")

    try:
        while True:
            color_image, depth_image = kinect.capture_frame()

            if color_image is not None and depth_image is not None:
                # 1. Depth-based segmentation (focus on objects 500-1500mm away)
                depth_vis, depth_colormap = process_depth_image(depth_image, min_depth=500, max_depth=1500)

                # 2. Color filtering (example: detect red objects in HSV space)
                hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
                # Red color range (wraps around in HSV)
                lower_red1 = np.array([0, 100, 100])
                upper_red1 = np.array([10, 255, 255])
                lower_red2 = np.array([160, 100, 100])
                upper_red2 = np.array([180, 255, 255])

                mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                color_mask = cv2.bitwise_or(mask1, mask2)

                # Apply morphological operations to clean up mask
                kernel = np.ones((5, 5), np.uint8)
                color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
                color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)

                # 3. Object detection using contours
                contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Draw results on copy of original image
                result_image = color_image.copy()

                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 500:  # Filter small contours
                        # Get bounding box
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

                        # Get center point
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])

                            # Get depth at center point (scale to depth image size if needed)
                            depth_h, depth_w = depth_image.shape
                            color_h, color_w = color_image.shape[:2]
                            dx = int(cx * depth_w / color_w)
                            dy = int(cy * depth_h / color_h)

                            if 0 <= dx < depth_w and 0 <= dy < depth_h:
                                depth_value = depth_image[dy, dx]
                                cv2.circle(result_image, (cx, cy), 5, (255, 0, 0), -1)
                                cv2.putText(result_image, f"{depth_value}mm", (cx+10, cy),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Display results
                cv2.imshow("Color Image", cv2.resize(color_image, (960, 540)))
                cv2.imshow("Depth Segmentation", depth_colormap)
                cv2.imshow("Color Filter (HSV)", color_mask)
                cv2.imshow("Contours Detected", cv2.resize(result_image, (960, 540)))

            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        kinect.close()
        cv2.destroyAllWindows()


def multi_sync():
    """Multi camera synchronization"""
    # Detect available devices
    devices = detect_kinect_devices()

    if len(devices) < 2:
        print("Multi-camera mode requires at least 2 Azure Kinect devices.")
        print(f"Only {len(devices)} device(s) detected.")
        if len(devices) == 1:
            print("Running with single camera...")
            kinect = KinectCamera(device_id=0)

            try:
                print("Press 's' to save synchronized frame, 'q' to quit")

                frame_count = 0
                while True:
                    color_image, depth_image = kinect.capture_frame()

                    if color_image is not None:
                        cv2.imshow("Camera 0", cv2.resize(color_image, (960, 540)))

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        if color_image is not None and depth_image is not None:
                            save_frame_data(color_image, depth_image, frame_count, output_dir="multi_cam_data/cam_0")
                            print(f"Saved frame {frame_count} from camera 0")
                            frame_count += 1

            finally:
                kinect.close()
                cv2.destroyAllWindows()

        return

    # Multi-camera mode
    print(f"\nInitializing {len(devices)} cameras for synchronized capture...")

    cameras = []
    try:
        # Initialize all cameras
        for device_info in devices:
            cam = KinectCamera(device_id=device_info["id"])
            cameras.append(cam)
            print(f"  Initialized camera {device_info['id']}")

        print("\nMulti-camera capture ready!")
        print("Press 's' to save synchronized frames from all cameras, 'q' to quit")

        frame_count = 0

        while True:
            # Capture from all cameras
            frames = []
            for cam in cameras:
                color, depth = cam.capture_frame()
                frames.append((color, depth))

            # Display all camera feeds
            for i, (color, depth) in enumerate(frames):
                if color is not None:
                    cv2.imshow(f"Camera {i}", cv2.resize(color, (640, 360)))

            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save synchronized frames from all cameras
                print(f"\nSaving synchronized frame set {frame_count}...")
                for i, (color, depth) in enumerate(frames):
                    if color is not None and depth is not None:
                        save_frame_data(color, depth, frame_count,
                                      output_dir=f"multi_cam_data/cam_{i}")
                        print(f"  Saved camera {i}")
                print(f"Frame set {frame_count} saved!")
                frame_count += 1

    finally:
        # Clean up all cameras
        for cam in cameras:
            cam.close()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    print("Azure Kinect + OpenCV Starter Code")
    
    # Detect available devices
    devices = detect_kinect_devices()
    
    if len(devices) == 0:
        print("No Azure Kinect devices found!")
        print("Check USB connection and permissions")
        exit(1)
    
    # Run demos
    while True:
        print("Azure Kinect Capture & Processing")
        print("1. Single camera capture and processing")
        print("2. Frame buffer demo")
        print("3. Point cloud generation")
        print("4. Continuous capture: every n seconds")
        print("5. Continuous capture: video recording")
        print("6. Capture 10 frames with intervals")
        print("7. Custom image processing pipeline")
        print("8. Multi-camera synchronization")
        print("0. Exit")

        choice = input("\nEnter choice: ")

        if choice == "1":
            demo_single_camera()
        elif choice == "2":
            demo_frame_buffer()
        elif choice == "3":
            demo_point_cloud()
        elif choice == "4":
            interval = input("Enter interval in seconds (default 5): ")
            try:
                interval = float(interval) if interval else 5.0
            except:
                interval = 5.0
            continuous_capture_loop(mode='images', interval_seconds=interval)
        elif choice == "5":
            continuous_capture_loop(mode='video')
        elif choice == "6":
            sync_frames()
        elif choice == "7":
            image_proc()
        elif choice == "8":
            multi_sync()
        elif choice == "0":
            break
        else:
            print("Invalid choice!")
    
