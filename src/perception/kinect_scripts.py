"""
Advanced Azure Kinect Scripts
"""

import cv2
import numpy as np
import pyk4a
from pyk4a import Config, PyK4A, ColorResolution, DepthMode
import open3d as o3d
from scipy.spatial.transform import Rotation
import time
from collections import deque
import threading
import json

# Object segmentation with depth

class DepthSegmentation:
    """
    Segments objects from background using depth information.
    """
    
    def __init__(self, kinect):
        self.kinect = kinect
        self.background_depth = None
        
    def capture_background(self, num_frames=30):
        """Capture background depth for subtraction"""
        print("Capturing background (no objects)...")
        depth_frames = []
        
        for i in range(num_frames):
            _, depth = self.kinect.capture_frame()
            if depth is not None:
                depth_frames.append(depth)
                print(f"  Frame {i+1}/{num_frames}")
            time.sleep(0.1)
        
        # Take median to reduce noise
        self.background_depth = np.median(depth_frames, axis=0)
        print("Background captured!")
        return self.background_depth
    
    def segment_objects(self, depth_image, threshold_mm=50):
        """
        Segment objects that are closer than background
        Returns binary mask of objects
        """
        if self.background_depth is None:
            print("No background captured!")
            return None
        
        # Objects are closer than background
        diff = self.background_depth.astype(np.float32) - depth_image.astype(np.float32)
        
        # Threshold to create binary mask
        object_mask = diff > threshold_mm
        
        # Clean up mask with morphological operations
        kernel = np.ones((5,5), np.uint8)
        object_mask = cv2.morphologyEx(object_mask.astype(np.uint8), 
                                       cv2.MORPH_CLOSE, kernel)
        object_mask = cv2.morphologyEx(object_mask, 
                                       cv2.MORPH_OPEN, kernel)
        
        return object_mask
    
    def find_object_contours(self, mask, min_area=1000):
        """Find individual objects from mask"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Get center of mass
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = x + w//2, y + h//2
                
                objects.append({
                    'contour': contour,
                    'bbox': (x, y, w, h),
                    'center': (cx, cy),
                    'area': area
                })
        
        return objects
    
    def get_object_depth(self, depth_image, mask):
        """Get average depth of segmented object"""
        masked_depth = depth_image[mask > 0]
        if len(masked_depth) > 0:
            # Remove zeros and outliers
            masked_depth = masked_depth[masked_depth > 0]
            if len(masked_depth) > 0:
                return np.median(masked_depth)
        return None


# Plane detection
class PlaneDetector:
    """
    Detects and removes planes (tables, walls) from point clouds.
    """
    
    def __init__(self):
        self.table_height = None
        self.table_normal = None
        
    def depth_to_point_cloud(self, depth_image, camera_matrix):
        """Convert depth image to point cloud"""
        h, w = depth_image.shape
        
        # Create mesh grid
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        
        # Valid depth mask
        valid = depth_image > 0
        
        # Compute 3D points
        z = depth_image[valid]
        x = (xx[valid] - camera_matrix[0, 2]) * z / camera_matrix[0, 0]
        y = (yy[valid] - camera_matrix[1, 2]) * z / camera_matrix[1, 1]
        
        points = np.stack([x, y, z], axis=-1)
        return points
    
    def detect_plane_ransac(self, points, threshold=10, min_points=1000):
        """
        Detect dominant plane using RANSAC
        Returns plane equation coefficients [a, b, c, d] where ax+by+cz+d=0
        """
        if len(points) < min_points:
            return None
            
        # Convert to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # RANSAC plane detection
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=threshold,
            ransac_n=3,
            num_iterations=1000
        )
        
        if len(inliers) < min_points:
            return None
            
        return plane_model, inliers
    
    def remove_plane_points(self, points, plane_model, threshold=10):
        """Remove points belonging to plane"""
        a, b, c, d = plane_model
        
        # Calculate distance to plane for all points
        distances = np.abs(a * points[:, 0] + b * points[:, 1] + 
                          c * points[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)
        
        # Keep points not on plane
        mask = distances > threshold
        return points[mask]
    
    def detect_table_height(self, depth_image, camera_matrix):
        """
        Detect table surface height
        Useful for defining manipulation workspace
        """
        # Convert to point cloud
        points = self.depth_to_point_cloud(depth_image, camera_matrix)
        
        # Detect dominant plane
        result = self.detect_plane_ransac(points)
        if result is None:
            return None
            
        plane_model, inliers = result
        
        # Get average height of plane points
        plane_points = points[inliers]
        table_height = np.mean(plane_points[:, 2])  # Z coordinate
        
        self.table_height = table_height
        self.table_normal = plane_model[:3]  # Normal vector
        
        return table_height
    
    def segment_objects_on_table(self, depth_image, camera_matrix, margin_mm=20):
        """
        Segment objects sitting on detected table
        Returns mask of objects above table
        """
        if self.table_height is None:
            print("Table not detected! Run detect_table_height first")
            return None
        
        # Convert to point cloud
        h, w = depth_image.shape
        points = self.depth_to_point_cloud(depth_image, camera_matrix)
        
        # Create full-size mask
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Find points above table
        valid = depth_image > 0
        z_values = depth_image[valid]
        
        # Points above table (closer to camera, so smaller z values)
        above_table = z_values < (self.table_height - margin_mm)
        
        # Map back to image coordinates
        valid_coords = np.where(valid)
        above_coords = (valid_coords[0][above_table], valid_coords[1][above_table])
        mask[above_coords] = 255
        
        return mask


# Motion detection and tracking
class MotionTracker:
    """
    Tracks motion and moving objects using temporal information.
    """
    
    def __init__(self, history_size=10):
        self.history = deque(maxlen=history_size)
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True
        )
        self.optical_flow = None
        self.prev_gray = None
        
    def update(self, color_image, depth_image):
        """Update motion tracker with new frame"""
        self.history.append({
            'color': color_image.copy(),
            'depth': depth_image.copy(),
            'timestamp': time.time()
        })
        
    def detect_motion_mask(self, color_image):
        """Get motion mask using background subtraction"""
        # Apply background subtractor
        motion_mask = self.background_subtractor.apply(color_image)
        
        # Remove shadows
        motion_mask[motion_mask < 255] = 0
        
        # Clean up noise
        kernel = np.ones((5,5), np.uint8)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
        
        return motion_mask
    
    def compute_optical_flow(self, color_image):
        """Compute dense optical flow between frames"""
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is not None:
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            # Compute magnitude and angle
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            self.optical_flow = {
                'flow': flow,
                'magnitude': magnitude,
                'angle': angle
            }
        
        self.prev_gray = gray
        return self.optical_flow
    
    def track_moving_object(self, color_image, depth_image):
        """
        Track the primary moving object in scene
        Returns bounding box and average depth
        """
        # Get motion mask
        motion_mask = self.detect_motion_mask(color_image)
        
        # Find contours
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find largest moving object
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Get average depth in bounding box
        roi_depth = depth_image[y:y+h, x:x+w]
        valid_depths = roi_depth[roi_depth > 0]
        
        if len(valid_depths) > 0:
            avg_depth = np.mean(valid_depths)
        else:
            avg_depth = None
        
        return {
            'bbox': (x, y, w, h),
            'contour': largest_contour,
            'depth': avg_depth,
            'area': cv2.contourArea(largest_contour)
        }
    
    def detect_gesture_swipe(self, min_displacement=100, time_window=1.0):
        """
        Detect horizontal swipe gestures
        Returns 'left', 'right', or None
        """
        if len(self.history) < 2:
            return None
        
        # Get frames from time window
        current_time = time.time()
        recent_frames = [f for f in self.history 
                        if current_time - f['timestamp'] < time_window]
        
        if len(recent_frames) < 2:
            return None
        
        # Track motion between first and last frame
        first_motion = self.detect_motion_mask(recent_frames[0]['color'])
        last_motion = self.detect_motion_mask(recent_frames[-1]['color'])
        
        # Get center of mass for each
        def get_center(mask):
            M = cv2.moments(mask)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                return cx
            return None
        
        first_cx = get_center(first_motion)
        last_cx = get_center(last_motion)
        
        if first_cx and last_cx:
            displacement = last_cx - first_cx
            if displacement > min_displacement:
                return 'right'
            elif displacement < -min_displacement:
                return 'left'
        
        return None


# Multi camera calibration

class MultiCameraCalibration:
    """
    Calibrate multiple cameras to common coordinate frame
    """
    
    def __init__(self, num_cameras):
        self.num_cameras = num_cameras
        self.calibrations = {}
        self.transforms = {}  # Transforms between cameras
        
        # Checkerboard parameters
        self.checkerboard_size = (9, 6)
        self.square_size = 0.025  # 25mm squares
        
    def calibrate_intrinsics(self, camera_id, images):
        """Calibrate single camera intrinsics"""
        # Prepare object points
        objp = np.zeros((self.checkerboard_size[0] * self.checkerboard_size[1], 3), 
                       np.float32)
        objp[:, :2] = np.mgrid[0:self.checkerboard_size[0], 
                              0:self.checkerboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        
        objpoints = []
        imgpoints = []
        
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find checkerboard corners
            ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
            
            if ret:
                # Refine corners
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                          criteria=(cv2.TERM_CRITERIA_EPS + 
                                                   cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                objpoints.append(objp)
                imgpoints.append(corners)
        
        # Calibrate
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )
        
        self.calibrations[camera_id] = {
            'matrix': mtx,
            'distortion': dist,
            'error': ret
        }
        
        return ret, mtx, dist
    
    def calibrate_stereo_pair(self, cam1_id, cam2_id, image_pairs):
        """
        Calibrate transform between two cameras
        NOTE: requires synchronized images of checkerboard from both cameras
        """
        # Get intrinsics
        K1 = self.calibrations[cam1_id]['matrix']
        D1 = self.calibrations[cam1_id]['distortion']
        K2 = self.calibrations[cam2_id]['matrix']
        D2 = self.calibrations[cam2_id]['distortion']
        
        # Prepare points
        objp = np.zeros((self.checkerboard_size[0] * self.checkerboard_size[1], 3), 
                       np.float32)
        objp[:, :2] = np.mgrid[0:self.checkerboard_size[0], 
                              0:self.checkerboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        
        objpoints = []
        imgpoints1 = []
        imgpoints2 = []
        
        for img1, img2 in image_pairs:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            ret1, corners1 = cv2.findChessboardCorners(gray1, self.checkerboard_size, None)
            ret2, corners2 = cv2.findChessboardCorners(gray2, self.checkerboard_size, None)
            
            if ret1 and ret2:
                corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1),
                                           criteria=(cv2.TERM_CRITERIA_EPS + 
                                                    cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1),
                                           criteria=(cv2.TERM_CRITERIA_EPS + 
                                                    cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                
                objpoints.append(objp)
                imgpoints1.append(corners1)
                imgpoints2.append(corners2)
        
        # Stereo calibration
        retval, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints1, imgpoints2,
            K1, D1, K2, D2,
            gray1.shape[::-1],
            flags=cv2.CALIB_FIX_INTRINSIC
        )
        
        # Store transform from cam1 to cam2
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = T.flatten()
        
        self.transforms[f"{cam1_id}_to_{cam2_id}"] = transform
        
        return R, T
    
    def get_global_transforms(self, reference_camera=0):
        """
        Compute all camera transforms relative to reference camera
        """
        global_transforms = {reference_camera: np.eye(4)}
        
        for i in range(self.num_cameras):
            if i != reference_camera:
                key = f"{reference_camera}_to_{i}"
                if key in self.transforms:
                    global_transforms[i] = self.transforms[key]
        
        return global_transforms


# Demo functions

def demo_depth_segmentation():
    """Object segmentation using depth"""

    # Initialize camera
    from cv_basics import KinectCamera
    kinect = KinectCamera(device_id=0)
    segmenter = DepthSegmentation(kinect)

    try:
        # Capture background
        input("Remove all objects and press Enter...")
        segmenter.capture_background()

        input("Place objects and press Enter...")

        print("Press 'q' to quit")

        while True:
            color, depth = kinect.capture_frame()
            if color is not None and depth is not None:
                # Segment objects
                mask = segmenter.segment_objects(depth)

                if mask is not None:
                    # Find individual objects
                    objects = segmenter.find_object_contours(mask)

                    # Visualize
                    vis = color.copy()
                    for obj in objects:
                        x, y, w, h = obj['bbox']
                        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)

                        # Get depth
                        obj_depth = segmenter.get_object_depth(depth, mask)
                        if obj_depth:
                            cv2.putText(vis, f"{obj_depth:.0f}mm",
                                      (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                                      0.5, (0, 255, 0), 2)

                    cv2.imshow("Segmented Objects", cv2.resize(vis, (960, 540)))
                    cv2.imshow("Mask", cv2.resize(mask * 255, (640, 576)))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        kinect.close()
        cv2.destroyAllWindows()


def demo_plane_detection():
    """Table plane detection and object segmentation"""

    # Initialize camera
    from src.perception.cv_basics import KinectCamera
    kinect = KinectCamera(device_id=0)
    plane_detector = PlaneDetector()

    try:
        print("Capturing frame to detect table plane...")
        input("Ensure camera can see a flat surface (table/floor) and press Enter...")

        color, depth = kinect.capture_frame()

        if depth is not None:
            # Get camera intrinsics
            calibration = kinect.calibration
            K_depth = calibration.get_camera_matrix(pyk4a.CalibrationType.DEPTH)

            # Detect table
            print("Detecting table plane using RANSAC...")
            table_height = plane_detector.detect_table_height(depth, K_depth)

            if table_height:
                print(f"Table detected at height: {table_height:.1f}mm")

                # Now segment objects on table
                print("\nPlace objects on the table...")
                input("Press Enter to start segmentation...")

                print("Press 'q' to quit")

                while True:
                    color, depth = kinect.capture_frame()

                    if color is not None and depth is not None:
                        # Segment objects above table
                        object_mask = plane_detector.segment_objects_on_table(depth, K_depth, margin_mm=20)

                        if object_mask is not None:
                            # Find contours
                            contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL,
                                                          cv2.CHAIN_APPROX_SIMPLE)

                            # Visualize
                            vis = color.copy()

                            for contour in contours:
                                area = cv2.contourArea(contour)
                                if area > 500:  # Filter small noise
                                    x, y, w, h = cv2.boundingRect(contour)
                                    cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                    cv2.putText(vis, f"Area: {int(area)}", (x, y-10),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                            cv2.imshow("Objects on Table", cv2.resize(vis, (960, 540)))
                            cv2.imshow("Object Mask", cv2.resize(object_mask, (640, 576)))

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            else:
                print("Could not detect table plane!")

    finally:
        kinect.close()
        cv2.destroyAllWindows()


def demo_motion_tracking():
    """Motion detection and tracking"""

    # Initialize camera
    from src.perception.cv_basics import KinectCamera
    kinect = KinectCamera(device_id=0)
    motion_tracker = MotionTracker(history_size=30)

    try:
        print("Motion Tracking Demo")
        print("Press 'q' to quit")
        print("\nWait a few seconds for background model to stabilize...")

        cv2.namedWindow("Color Stream", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Motion Mask", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Optical Flow", cv2.WINDOW_NORMAL)

        frame_count = 0

        while True:
            color, depth = kinect.capture_frame()

            if color is not None and depth is not None:
                # Update motion tracker
                motion_tracker.update(color, depth)

                # Detect motion
                motion_mask = motion_tracker.detect_motion_mask(color)

                # Compute optical flow
                flow_data = motion_tracker.compute_optical_flow(color)

                # Track moving object
                tracked_object = motion_tracker.track_moving_object(color, depth)

                # Visualize
                vis = color.copy()

                if tracked_object:
                    x, y, w, h = tracked_object['bbox']
                    cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 3)

                    depth_val = tracked_object['depth']
                    if depth_val:
                        cv2.putText(vis, f"Depth: {depth_val:.0f}mm", (x, y-30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    cv2.putText(vis, f"Area: {int(tracked_object['area'])}", (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Optical flow visualization
                if flow_data:
                    magnitude = flow_data['magnitude']
                    angle = flow_data['angle']

                    # Create HSV image for flow visualization
                    hsv = np.zeros((magnitude.shape[0], magnitude.shape[1], 3), dtype=np.uint8)
                    hsv[..., 0] = angle * 180 / np.pi / 2  # Hue = direction
                    hsv[..., 1] = 255  # Saturation = max
                    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value = magnitude

                    flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                    cv2.imshow("Optical Flow", cv2.resize(flow_vis, (640, 360)))

                # Detect swipe gestures
                if frame_count % 10 == 0:  # Check every 10 frames
                    gesture = motion_tracker.detect_gesture_swipe(min_displacement=150, time_window=1.0)
                    if gesture:
                        print(f"Gesture detected: {gesture.upper()} SWIPE!")

                cv2.imshow("Color Stream", cv2.resize(vis, (960, 540)))
                cv2.imshow("Motion Mask", cv2.resize(motion_mask, (640, 360)))

                frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        kinect.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Advanced Azure Kinect Scripts")

    while True:
        print("\nSelect demo:")
        print("1. Depth-based object segmentation")
        print("2. Table plane detection")
        print("3. Motion tracking")
        print("0. Exit")

        choice = input("\nChoice: ")

        if choice == "1":
            demo_depth_segmentation()
        elif choice == "2":
            demo_plane_detection()
        elif choice == "3":
            demo_motion_tracking()
        elif choice == "0":
            break
        else:
            print("Invalid choice!")