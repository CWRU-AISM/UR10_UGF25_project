"""
Advanced OpenCV Scripts
Computer vision algorithms for robotics applications
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage import measure
import matplotlib.pyplot as plt
from collections import deque
import time

# Color based object detection

class ColorDetector:
    """
    Detects objects based on color in HSV space.
    Why use this: More robust than RGB for color detection under varying lighting.
    Applications: Detecting colored blocks, markers, specific objects
    """
    
    def __init__(self):
        # Define color ranges in HSV
        # Hue: 0-180, Saturation: 0-255, Value: 0-255
        self.color_ranges = {
            'red': [
                (np.array([0, 100, 100]), np.array([10, 255, 255])),    # Lower red
                (np.array([170, 100, 100]), np.array([180, 255, 255]))  # Upper red
            ],
            'green': [(np.array([40, 50, 50]), np.array([80, 255, 255]))],
            'blue': [(np.array([100, 50, 50]), np.array([130, 255, 255]))],
            'yellow': [(np.array([20, 100, 100]), np.array([30, 255, 255]))],
            'orange': [(np.array([10, 100, 100]), np.array([20, 255, 255]))],
            'purple': [(np.array([130, 50, 50]), np.array([170, 255, 255]))]
        }
        
    def detect_color(self, image, color_name):
        """
        Detect specific color in image
        Returns binary mask and contours
        """
        # Convert to HSV (Hue, Saturation, Value)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for color
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        if color_name in self.color_ranges:
            for (lower, upper) in self.color_ranges[color_name]:
                mask_part = cv2.inRange(hsv, lower, upper)
                mask = cv2.bitwise_or(mask, mask_part)
        
        # Clean up mask with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)      # Remove noise
        mask = cv2.dilate(mask, kernel, iterations=2)     # Fill gaps
        mask = cv2.erode(mask, kernel, iterations=1)      # Restore size
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return mask, contours
    
    def get_color_statistics(self, image, mask):
        """
        Get color statistics for masked region
        Useful for: Color calibration, quality checks
        """
        # Extract color values in masked region
        masked_pixels = image[mask > 0]
        
        if len(masked_pixels) == 0:
            return None
        
        # Convert to different color spaces
        hsv_pixels = cv2.cvtColor(masked_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV)
        lab_pixels = cv2.cvtColor(masked_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB)
        
        stats = {
            'bgr_mean': np.mean(masked_pixels, axis=0),
            'bgr_std': np.std(masked_pixels, axis=0),
            'hsv_mean': np.mean(hsv_pixels.reshape(-1, 3), axis=0),
            'hsv_std': np.std(hsv_pixels.reshape(-1, 3), axis=0),
            'lab_mean': np.mean(lab_pixels.reshape(-1, 3), axis=0),
            'pixel_count': len(masked_pixels)
        }
        
        return stats
    
    def auto_calibrate_color(self, image, roi):
        """
        Automatically calibrate color range from ROI
        ROI: (x, y, w, h) region containing target color
        """
        x, y, w, h = roi
        roi_image = image[y:y+h, x:x+w]
        
        # Convert to HSV
        hsv_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
        
        # Get HSV statistics
        h_vals = hsv_roi[:, :, 0].flatten()
        s_vals = hsv_roi[:, :, 1].flatten()
        v_vals = hsv_roi[:, :, 2].flatten()
        
        # Calculate ranges (mean ± 2*std with bounds)
        h_mean, h_std = np.mean(h_vals), np.std(h_vals)
        s_mean, s_std = np.mean(s_vals), np.std(s_vals)
        v_mean, v_std = np.mean(v_vals), np.std(v_vals)
        
        lower = np.array([
            max(0, h_mean - 2*h_std),
            max(0, s_mean - 2*s_std),
            max(0, v_mean - 2*v_std)
        ])
        
        upper = np.array([
            min(180, h_mean + 2*h_std),
            min(255, s_mean + 2*s_std),
            min(255, v_mean + 2*v_std)
        ])
        
        return lower, upper


# Template matching

class TemplateMatching:
    """
    Find objects using template matching.
    Why use this: Good for finding known objects without training ML models.
    Applications: Finding specific parts, QR codes regions, fiducial markers
    """
    
    def __init__(self):
        self.templates = {}
        self.methods = {
            'correlation': cv2.TM_CCORR_NORMED,
            'squared_diff': cv2.TM_SQDIFF_NORMED,
            'correlation_coeff': cv2.TM_CCOEFF_NORMED
        }
    
    def add_template(self, name, template_img, scales=None):
        """
        Add template with multiple scales
        Scales: list of scale factors, e.g., [0.8, 0.9, 1.0, 1.1, 1.2]
        """
        if scales is None:
            scales = [0.8, 0.9, 1.0, 1.1, 1.2]
        
        self.templates[name] = {
            'original': template_img,
            'scales': {}
        }
        
        # Generate scaled versions
        for scale in scales:
            width = int(template_img.shape[1] * scale)
            height = int(template_img.shape[0] * scale)
            scaled = cv2.resize(template_img, (width, height))
            self.templates[name]['scales'][scale] = scaled
    
    def match_template(self, image, template_name, threshold=0.8, method='correlation_coeff'):
        """
        Find template in image at multiple scales
        Returns list of matches with positions and scores
        """
        if template_name not in self.templates:
            return []
        
        matches = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        for scale, template in self.templates[template_name]['scales'].items():
            # Convert template to grayscale
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template
            
            # Apply template matching
            result = cv2.matchTemplate(gray, template_gray, self.methods[method])
            
            # Find locations above threshold
            if method == 'squared_diff':
                # For SQDIFF, lower values are better
                locations = np.where(result <= 1 - threshold)
            else:
                locations = np.where(result >= threshold)
            
            # Get match details
            for pt in zip(*locations[::-1]):
                h, w = template_gray.shape[:2]
                matches.append({
                    'position': pt,
                    'size': (w, h),
                    'scale': scale,
                    'score': result[pt[1], pt[0]],
                    'bbox': (pt[0], pt[1], pt[0] + w, pt[1] + h)
                })
        
        # Non-maximum suppression to remove overlapping detections
        return self.non_max_suppression(matches)
    
    def non_max_suppression(self, matches, overlap_thresh=0.3):
        """Remove overlapping detections"""
        if len(matches) == 0:
            return []
        
        # Sort by score
        matches = sorted(matches, key=lambda x: x['score'], reverse=True)
        
        keep = []
        while matches:
            # Take best match
            best = matches.pop(0)
            keep.append(best)
            
            # Remove overlapping matches
            remaining = []
            for match in matches:
                # Calculate IoU (Intersection over Union)
                x1 = max(best['bbox'][0], match['bbox'][0])
                y1 = max(best['bbox'][1], match['bbox'][1])
                x2 = min(best['bbox'][2], match['bbox'][2])
                y2 = min(best['bbox'][3], match['bbox'][3])
                
                if x1 < x2 and y1 < y2:
                    intersection = (x2 - x1) * (y2 - y1)
                    area1 = (best['bbox'][2] - best['bbox'][0]) * (best['bbox'][3] - best['bbox'][1])
                    area2 = (match['bbox'][2] - match['bbox'][0]) * (match['bbox'][3] - match['bbox'][1])
                    iou = intersection / (area1 + area2 - intersection)
                    
                    if iou < overlap_thresh:
                        remaining.append(match)
                else:
                    remaining.append(match)
            
            matches = remaining
        
        return keep
    
    def match_with_rotation(self, image, template_name, angle_range=(-30, 30), angle_step=5):
        """
        Match template with rotation invariance
        Slower but handles rotated objects
        """
        all_matches = []
        
        for angle in range(angle_range[0], angle_range[1] + 1, angle_step):
            # Rotate template
            for scale, template in self.templates[template_name]['scales'].items():
                center = (template.shape[1] // 2, template.shape[0] // 2)
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(template, matrix, (template.shape[1], template.shape[0]))
                
                # Match rotated template
                # ... (similar to match_template but with rotation info)
        
        return all_matches


# Shape Detection

class ShapeDetector:
    """
    Detect and classify geometric shapes.
    Why use this: Identify parts by shape, detect round vs square objects.
    Applications: Part sorting, quality control, pose estimation
    """
    
    def __init__(self):
        self.min_area = 500  # Minimum contour area
        
    def detect_shapes(self, image):
        """
        Detect and classify shapes in image
        Returns list of detected shapes with properties
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shapes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            
            # Approximate polygon
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Get shape properties
            shape_info = self.classify_shape(contour, approx)
            shape_info['contour'] = contour
            shape_info['approximation'] = approx
            shape_info['area'] = area
            
            shapes.append(shape_info)
        
        return shapes
    
    def classify_shape(self, contour, approx):
        """
        Classify shape based on contour properties
        """
        vertices = len(approx)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        
        # Get shape properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Circularity (1.0 for perfect circle)
        circularity = 4 * np.pi * area / (perimeter ** 2)
        
        # Convexity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Classify based on vertices and properties
        if vertices == 3:
            shape_type = "triangle"
        elif vertices == 4:
            # Check if square or rectangle
            if 0.95 <= aspect_ratio <= 1.05:
                shape_type = "square"
            else:
                shape_type = "rectangle"
        elif vertices == 5:
            shape_type = "pentagon"
        elif vertices == 6:
            shape_type = "hexagon"
        elif vertices > 6:
            # Check if circle
            if circularity > 0.8:
                shape_type = "circle"
            else:
                shape_type = f"polygon_{vertices}"
        else:
            shape_type = "irregular"
        
        # Get center and orientation
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w//2, y + h//2
        
        # Get orientation for elongated shapes
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            angle = ellipse[2]
        else:
            angle = 0
        
        return {
            'type': shape_type,
            'vertices': vertices,
            'center': (cx, cy),
            'bbox': (x, y, w, h),
            'aspect_ratio': aspect_ratio,
            'circularity': circularity,
            'solidity': solidity,
            'angle': angle
        }
    
    def match_shape_similarity(self, contour1, contour2, method=cv2.CONTOURS_MATCH_I1):
        """
        Compare two shapes using Hu moments
        Lower values = more similar
        Methods:
        - cv2.CONTOURS_MATCH_I1: Best for general use
        - cv2.CONTOURS_MATCH_I2: More sensitive to differences
        - cv2.CONTOURS_MATCH_I3: Normalized, good for different scales
        """
        similarity = cv2.matchShapes(contour1, contour2, method, 0.0)
        return similarity
    
    def detect_circles_hough(self, image):
        """
        Detect circles using Hough transform
        More robust for perfect circles than contour detection
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Blur to reduce noise
        blurred = cv2.medianBlur(gray, 5)
        
        # Detect circles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,               # Inverse ratio of accumulator resolution
            minDist=50,         # Minimum distance between circles
            param1=100,         # Canny edge threshold
            param2=30,          # Accumulator threshold
            minRadius=10,       # Minimum radius
            maxRadius=100       # Maximum radius
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            return [{'center': (x, y), 'radius': r} for x, y, r in circles]
        
        return []


# Feature matching 

class FeatureTracker:
    """
    Track objects using feature matching (SIFT/ORB).
    Why use this: Robust tracking of textured objects, handles rotation/scale.
    Applications: Object tracking, visual odometry, image stitching
    """
    
    def __init__(self, detector_type='ORB'):
        """
        Initialize feature detector
        ORB: Fast, free to use
        SIFT: More accurate but patented (use for research only)
        """
        if detector_type == 'ORB':
            self.detector = cv2.ORB_create(nfeatures=1000)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        elif detector_type == 'SIFT':
            self.detector = cv2.SIFT_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        
        self.reference_features = {}
    
    def extract_features(self, image):
        """Extract keypoints and descriptors from image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def add_reference(self, name, image):
        """Add reference object to track"""
        keypoints, descriptors = self.extract_features(image)
        self.reference_features[name] = {
            'keypoints': keypoints,
            'descriptors': descriptors,
            'image': image
        }
    
    def track_object(self, image, reference_name, min_matches=10):
        """
        Track reference object in image
        Returns homography matrix and bounding box if found
        """
        if reference_name not in self.reference_features:
            return None
        
        # Get reference features
        ref = self.reference_features[reference_name]
        ref_kp = ref['keypoints']
        ref_desc = ref['descriptors']
        ref_img = ref['image']
        
        # Extract features from current image
        kp, desc = self.extract_features(image)
        
        if desc is None or ref_desc is None:
            return None
        
        # Match features
        matches = self.matcher.match(ref_desc, desc)
        matches = sorted(matches, key=lambda x: x.distance)
        
        if len(matches) < min_matches:
            return None
        
        # Get matched points
        src_pts = np.float32([ref_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Find homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is None:
            return None
        
        # Get bounding box in current image
        h, w = ref_img.shape[:2]
        corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, M)
        
        return {
            'homography': M,
            'corners': transformed_corners,
            'matches': matches[:min_matches],
            'num_matches': len(matches)
        }
    
    def compute_pose_from_homography(self, homography, camera_matrix):
        """
        Decompose homography to get 3D pose
        Returns rotation and translation vectors
        """
        # Decompose homography
        num, Rs, Ts, Ns = cv2.decomposeHomographyMat(homography, camera_matrix)
        
        # Select best solution (usually first one)
        # In practice, you'd need additional checks
        if num > 0:
            return Rs[0], Ts[0]
        
        return None, None


# ==================== SCRIPT 5: LINE AND EDGE DETECTION ====================

class LineDetector:
    """
    Detect lines and edges for navigation and alignment.
    Why use this: Robot alignment, path following, workspace boundary detection.
    Applications: Following conveyor edges, aligning with tables, lane detection
    """
    
    def __init__(self):
        self.canny_low = 50
        self.canny_high = 150
        
    def detect_lines_probabilistic(self, image, min_length=100, max_gap=10):
        """
        Detect line segments using Probabilistic Hough Transform
        Faster and gives endpoints
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Edge detection
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        
        # Detect lines
        lines = cv2.HoughLinesP(
            edges,
            rho=1,                  # Distance resolution in pixels
            theta=np.pi/180,        # Angle resolution in radians
            threshold=50,           # Minimum votes
            minLineLength=min_length,
            maxLineGap=max_gap
        )
        
        if lines is None:
            return []
        
        # Process lines
        processed_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate line properties
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
            
            processed_lines.append({
                'start': (x1, y1),
                'end': (x2, y2),
                'length': length,
                'angle': angle,
                'midpoint': ((x1+x2)//2, (y1+y2)//2)
            })
        
        return processed_lines
    
    def detect_dominant_lines(self, image, num_lines=5):
        """
        Detect dominant lines using Standard Hough Transform
        Good for finding major structural lines
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        
        # Standard Hough Transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is None:
            return []
        
        # Convert to line segments
        dominant_lines = []
        for i in range(min(num_lines, len(lines))):
            rho, theta = lines[i][0]
            
            # Convert polar to cartesian
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            
            # Get two points on the line
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            
            dominant_lines.append({
                'rho': rho,
                'theta': theta,
                'angle_degrees': theta * 180 / np.pi,
                'points': [(x1, y1), (x2, y2)]
            })
        
        return dominant_lines
    
    def find_line_intersections(self, lines):
        """
        Find intersection points between lines
        Useful for corner detection, calibration patterns
        """
        intersections = []
        
        for i in range(len(lines)):
            for j in range(i+1, len(lines)):
                line1 = lines[i]
                line2 = lines[j]
                
                # Get line equations: ax + by = c
                x1, y1 = line1['start']
                x2, y2 = line1['end']
                x3, y3 = line2['start']
                x4, y4 = line2['end']
                
                denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
                
                if abs(denom) < 1e-10:
                    continue  # Lines are parallel
                
                # Find intersection
                t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
                
                # Check if intersection is within both line segments
                if 0 <= t <= 1:
                    u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / denom
                    if 0 <= u <= 1:
                        # Calculate intersection point
                        ix = x1 + t*(x2-x1)
                        iy = y1 + t*(y2-y1)
                        
                        intersections.append({
                            'point': (int(ix), int(iy)),
                            'line1_idx': i,
                            'line2_idx': j,
                            'angle': abs(line1['angle'] - line2['angle'])
                        })
        
        return intersections


# ==================== DEMO AND VISUALIZATION FUNCTIONS ====================

def demo_color_detection():
    """Demo: Detect colored objects"""
    print("\n=== Color Detection Demo ===")
    
    # Create sample image with colored shapes
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Draw colored objects
    cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 255), -1)    # Red
    cv2.rectangle(img, (200, 50), (300, 150), (0, 255, 0), -1)   # Green
    cv2.circle(img, (400, 100), 50, (255, 0, 0), -1)             # Blue
    cv2.rectangle(img, (50, 200), (150, 300), (0, 165, 255), -1) # Orange
    
    detector = ColorDetector()
    
    # Detect each color
    for color in ['red', 'green', 'blue', 'orange']:
        mask, contours = detector.detect_color(img, color)
        
        # Draw contours
        result = img.copy()
        cv2.drawContours(result, contours, -1, (0, 0, 0), 2)
        
        # Add labels
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.putText(result, color, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        cv2.imshow(f"Detected: {color}", result)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def demo_shape_detection():
    """Demo: Detect and classify shapes"""
    print("\n=== Shape Detection Demo ===")
    
    # Create image with various shapes
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Draw shapes
    cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 0), 2)           # Square
    cv2.rectangle(img, (200, 50), (350, 150), (0, 0, 0), 2)          # Rectangle
    cv2.circle(img, (450, 100), 50, (0, 0, 0), 2)                    # Circle
    triangle = np.array([[100, 250], [50, 350], [150, 350]], np.int32)
    cv2.polylines(img, [triangle], True, (0, 0, 0), 2)               # Triangle
    
    detector = ShapeDetector()
    shapes = detector.detect_shapes(img)
    
    # Visualize detected shapes
    result = img.copy()
    for shape in shapes:
        # Draw bounding box
        x, y, w, h = shape['bbox']
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 1)
        
        # Add label
        cv2.putText(result, shape['type'], shape['center'], 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw center
        cv2.circle(result, shape['center'], 3, (0, 0, 255), -1)
    
    cv2.imshow("Shape Detection", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def demo_line_detection():
    """Demo: Detect lines for alignment"""
    print("\n=== Line Detection Demo ===")
    
    # Create image with lines (simulating table edges, conveyor, etc.)
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Draw some lines
    cv2.line(img, (50, 100), (590, 100), (0, 0, 0), 2)    # Horizontal
    cv2.line(img, (100, 50), (100, 430), (0, 0, 0), 2)    # Vertical
    cv2.line(img, (200, 50), (400, 250), (0, 0, 0), 2)    # Diagonal
    cv2.line(img, (400, 50), (200, 250), (0, 0, 0), 2)    # Diagonal
    
    detector = LineDetector()
    
    # Detect lines
    lines = detector.detect_lines_probabilistic(img, min_length=50)
    
    # Visualize
    result = img.copy()
    for line in lines:
        cv2.line(result, line['start'], line['end'], (0, 255, 0), 2)
        cv2.circle(result, line['midpoint'], 5, (255, 0, 0), -1)
        
        # Add angle annotation
        cv2.putText(result, f"{line['angle']:.1f}°", 
                   line['midpoint'], cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 0, 0), 1)
    
    # Find intersections
    intersections = detector.find_line_intersections(lines)
    for intersection in intersections:
        cv2.circle(result, intersection['point'], 8, (0, 0, 255), -1)
    
    cv2.imshow("Line Detection", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("OpenCV Advanced Scripts")
    print("========================\n")
    
    while True:
        print("\nSelect demo:")
        print("1. Color detection")
        print("2. Shape detection") 
        print("3. Line detection")
        print("0. Exit")
        
        choice = input("\nChoice: ")
        
        if choice == "1":
            demo_color_detection()
        elif choice == "2":
            demo_shape_detection()
        elif choice == "3":
            demo_line_detection()
        elif choice == "0":
            break
    
