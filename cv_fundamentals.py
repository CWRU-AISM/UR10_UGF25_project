"""
Computer Vision Group - Week 1 Starter Code
Azure Kinect + OpenCV Fundamentals
"""

import cv2
import numpy as np
import pyk4a
from pyk4a import Config, PyK4A, ColorResolution, DepthMode, ImageFormat, FPS
import os
from datetime import datetime
import json

# Basic Azure setup

class KinectCamera:
    """Wrapper class for Azure Kinect operations"""
    
    def __init__(self, device_id=0):
        """Initialize Azure Kinect with standard configuration"""
        self.device_id = device_id
        
        # Configure camera settings
        self.config = Config(
            color_resolution=ColorResolution.RES_1080P,  # 1920x1080
            depth_mode=DepthMode.NFOV_UNBINNED,          # 640x576
            image_format=ImageFormat.COLOR_BGRA32,
            camera_fps=FPS.FPS_30,
            synchronized_images_only=True,  # Ensure color and depth are synced
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
        """Convert depth image to 3D point cloud"""
        return self.camera.calibration.convert_2d_to_3d(
            depth_image, 
            self.camera.calibration.K_DEPTH
        )
    
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
            serial = temp_camera.serial_number
            temp_camera.close()
            devices.append({"id": i, "serial": serial})
            print(f"  Device {i}: Serial {serial}")
        except Exception as e:
            print(f"  Device {i}: Could not get info - {e}")
    
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
    """Demo: Buffering frames for batch processing"""
    
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
        
        # Example: Compute average depth over buffered frames
        depth_stack = np.stack(depth_buffer, axis=0)
        avg_depth = np.mean(depth_stack, axis=0)
        
        # Visualize average depth
        _, avg_depth_colormap = process_depth_image(avg_depth.astype(np.uint16))
        cv2.imshow("Average Depth (1 second)", avg_depth_colormap)
        cv2.waitKey(0)
        
        # Example: Detect motion by comparing first and last frame
        frame_diff = cv2.absdiff(color_buffer[0], color_buffer[-1])
        gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
        _, motion_mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
        
        cv2.imshow("Motion Mask", motion_mask)
        cv2.waitKey(0)
    
    finally:
        kinect.close()
        cv2.destroyAllWindows()


def demo_point_cloud():
    """Demo: Generate and save point cloud from depth"""
    
    # Initialize camera
    kinect = KinectCamera(device_id=0)
    
    try:
        print("Capturing frame for point cloud...")
        color_image, depth_image = kinect.capture_frame()
        
        if depth_image is not None:
            # Generate point cloud
            print("Generating point cloud...")
            point_cloud = kinect.get_point_cloud(depth_image)
            
            # Reshape to list of 3D points
            points_3d = point_cloud.reshape(-1, 3)
            
            # Remove invalid points (depth = 0)
            valid_mask = points_3d[:, 2] > 0
            valid_points = points_3d[valid_mask]
            
            print(f"Generated {len(valid_points)} valid 3D points")
            
            # Save as simple text file (for visualization in MeshLab/CloudCompare)
            output_file = "point_cloud.xyz"
            np.savetxt(output_file, valid_points, fmt='%.3f')
            print(f"Saved point cloud to {output_file}")
            
    
    finally:
        kinect.close()



def exercise_1():
    """Exercise 1: Capture and save 10 synchronized frames"""
    # TODO: Implement capturing and saving 10 frames with 1-second intervals
    pass


def exercise_2():
    """Exercise 2: Apply custom image processing"""
    # TODO: Implement custom processing pipeline
    # Ideas: Object detection, color filtering, depth-based segmentation
    pass


def exercise_3():
    """Exercise 3: Multi-camera synchronization"""
    # TODO: If multiple cameras available, capture from all simultaneously
    pass



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
        print("\nSelect demo to run:")
        print("1. Single camera capture and processing")
        print("2. Frame buffer demo")
        print("3. Point cloud generation")
        print("4. Exercise 1: Capture 10 frames")
        print("5. Exercise 2: Custom processing")
        print("6. Exercise 3: Multi-camera")
        print("0. Exit")
        
        choice = input("\nEnter choice: ")
        
        if choice == "1":
            demo_single_camera()
        elif choice == "2":
            demo_frame_buffer()
        elif choice == "3":
            demo_point_cloud()
        elif choice == "4":
            exercise_1()
        elif choice == "5":
            exercise_2()
        elif choice == "6":
            exercise_3()
        elif choice == "0":
            break
        else:
            print("Invalid choice!")
    
