# UR10e Undergraduate Research Project

Repository for students working with the UR10e robot. This codebase supports Computer Vision (CV) and Robot Control.

## Repository Structure

```
UR10_UGF25_project/
├── src/
│   ├── perception/              # Computer Vision modules (non-ROS)
│   │   ├── azure_kinect.py      # Azure Kinect DK with pyk4a & viewer
│   │   ├── cv_basics.py         # Basic CV with Azure Kinect
│   │   ├── cv_scripts.py        # Advanced CV algorithms
│   │   ├── kinect_scripts.py    # Advanced depth processing
│   │   └── nerf_studio.py       # NeRF Studio integration
│   └── robot_control/           # Direct robot control (non-ROS)
│       ├── ur10_control.py      # Legacy control scripts
│       └── ur10_socket.py       # Low-level socket communication
├── ros2_ws/src/                 # ROS2 workspace
│   └── ur10_control/            # MoveIt2 motion planning package
│       ├── launch/              # ROS2 launch files
│       ├── config/              # Controller configurations
│       └── scripts/             # Motion planning examples
└── examples/                    # Reference code and templates
    ├── urx_examples/            # Legacy URX library examples
    ├── ur_templates/            # UR script templates
    └── nerf_examples/           # NeRF Studio workflow examples
```

## Prerequisites

### Core Dependencies
```bash
# Python dependencies
pip install numpy opencv-python pyk4a open3d scipy scikit-image matplotlib

# NeRF Studio (for 3D reconstruction)
pip install nerfstudio nerfacc

# For ROS2 users (Control group)
sudo apt install ros-humble-ur-robot-driver
sudo apt install ros-humble-moveit
sudo apt install ros-humble-moveit-planners-ompl
```

### Azure Kinect Setup
Install the Azure Kinect SDK:
- Windows: https://github.com/microsoft/Azure-Kinect-Sensor-SDK/releases
- Linux: Follow pyk4a installation guide

## Computer Vision 

The CV portion uses standalone Python scripts without ROS2. All code is in `src/perception/`.

### Azure Kinect Integration

**File:** `src/perception/azure_kinect.py`

Features:
- Real-time viewer (similar to k4aviewer.exe)
- Point cloud generation with Open3D
- Factory camera intrinsics
- Multiple depth modes (NFOV/WFOV, Binned/Unbinned)
- Video recording and snapshot capture

**Basic Usage:**
```python
from src.perception.azure_kinect import AzureKinect

# Initialize and connect
kinect = AzureKinect()
kinect.connect()

# Get factory camera calibration
intrinsics = kinect.get_camera_intrinsics()

# Capture frame
color, depth, ir, aligned_depth = kinect.capture_frame(align_depth_to_color=True)

# Generate point cloud
pcd = kinect.get_point_cloud(color, aligned_depth)
kinect.save_point_cloud("output.ply", pcd)

kinect.disconnect()
```

**Interactive Viewer:**
```python
from src.perception.azure_kinect import KinectViewer

viewer = KinectViewer(device_id=0)
viewer.run()
```

Keyboard controls:
- `1-4`: Switch views (All/Color/Depth/IR)
- `M`: Cycle through depth modes
- `C`: Change depth colormap
- `P`: Toggle point cloud 3D view
- `R`: Toggle RGB/Depth coloring (3D mode)
- `+/-`: Adjust point size (3D) or depth range (2D)
- `V`: Start/stop video recording
- `S`: Save snapshot
- `Q` or `ESC`: Exit

### Basic Computer Vision

**File:** `src/perception/cv_basics.py`

Demo applications for learning CV fundamentals:

```bash
python src/perception/cv_basics.py
```

Available demos:
1. Single camera capture and processing
2. Frame buffer demo (averaging, motion detection)
3. Point cloud generation with visualization
4. Continuous capture (periodic images)
5. Continuous capture (video recording)
6. Capture 10 frames with intervals
7. Custom image processing pipeline
8. Multi-camera synchronization

**Example - Point Cloud:**
```python
from src.perception.cv_basics import KinectCamera

kinect = KinectCamera(device_id=0)
color, depth = kinect.capture_frame()

# Generate point cloud
point_cloud = kinect.get_point_cloud(depth)  # Returns Nx3 array

kinect.close()
```

### Advanced Computer Vision

**File:** `src/perception/cv_scripts.py`

Production-ready CV algorithms with support for both live camera and image folders.

```bash
python src/perception/cv_scripts.py
```

Features:
- Color detection in HSV space
- Shape detection and classification
- Line detection with Hough transform
- Template matching (multi-scale, rotation-invariant)
- Feature tracking (SIFT/ORB)

**Example - Using with Image Folder:**
```bash
python src/perception/cv_scripts.py
# Choose option 2: Load images from folder
# Enter path
# Select demo (color/shape/line detection)
```

**Example - Programmatic Use:**
```python
from src.perception.cv_scripts import ColorDetector, ShapeDetector

# Color detection
detector = ColorDetector()
mask, contours = detector.detect_color(image, 'red')

# Shape detection
shape_detector = ShapeDetector()
shapes = shape_detector.detect_shapes(image)
for shape in shapes:
    print(f"Found {shape['type']} at {shape['center']}")
```

### Advanced Depth Processing

**File:** `src/perception/kinect_scripts.py`

Specialized depth-based algorithms:

```bash
python src/perception/kinect_scripts.py
```

Available demos:
1. Depth-based object segmentation
2. Table plane detection (RANSAC)
3. Motion tracking with optical flow

Features:
- Background subtraction
- Plane fitting with RANSAC
- Object segmentation on tables
- Motion tracking and gesture detection

## Neural Radiance Fields (NeRF)

**File:** `src/perception/nerf_studio.py`

Integration with NeRF Studio for photo-realistic 3D reconstruction from multi-view images.

### Quick Start

**Manual capture:**
```bash
python examples/nerf_examples/basic_nerf_workflow.py
```

**Train NeRF:**
```bash
python examples/nerf_examples/train_nerf_workflow.py data/nerf_basic
```

**View results:**
```bash
ns-viewer --load-config outputs/nerfacto_*/config.yml
```

### Programmatic Usage

**Capture dataset:**
```python
from src.perception.nerf_studio import NeRFDataCapture

capture = NeRFDataCapture(output_dir="data/my_scene")
capture.connect_kinect()

# Capture 100 images with 0.5s interval
capture.capture_dataset(num_images=100, interval=0.5)

# Save camera transforms
capture.save_transforms()
capture.disconnect()
```

**Train model:**
```python
from src.perception.nerf_studio import NeRFTrainer

trainer = NeRFTrainer(data_dir="data/my_scene")
trainer.train(method="nerfacto", max_num_iterations=30000)
```

**Automated capture with robot:**
```python
from src.perception.nerf_studio import RobotNeRFCapture

robot_capture = RobotNeRFCapture(output_dir="data/robot_scene")
robot_capture.connect(robot_ip="192.168.1.101")

# Capture hemisphere around object at (0.4, 0.3, 0.2)
robot_capture.capture_hemisphere(
    center_point=(0.4, 0.3, 0.2),
    radius=0.5,
    num_views=36
)

robot_capture.disconnect()
```

### Available Methods

- `nerfacto` - Default, balanced quality and speed
- `instant-ngp` - Fast training, requires CUDA
- `tensorf` - High quality reconstruction
- `mipnerf` - Better fine details with anti-aliasing

See `examples/nerf_examples/README.md` for detailed workflows and best practices.

## Robot Control Track

### Direct Control (Non-ROS)

**File:** `src/robot_control/ur10_controller.py`

For students not using ROS2:

```python
from src.robot_control.ur10_controller import UR10Controller, RobotConfig

# Configure robot
config = RobotConfig(
    robot_ip="192.168.1.101",
    max_velocity=1.0,
    max_acceleration=1.0
)

robot = UR10Controller(config)
robot.connect()

# Joint space motion
robot.move_joints([0, -1.57, 1.57, -1.57, -1.57, 0])

# Cartesian space motion
robot.move_linear([0.4, 0.3, 0.2, 0, 3.14, 0])

robot.disconnect()
```

### ROS2 Control with MoveIt2

**Build workspace:**
```bash
cd ros2_ws
colcon build --packages-select ur10_control
source install/setup.bash
```

**Launch MoveIt2 demo:**
```bash
ros2 launch ur10_control ur10_moveit_demo.launch.py
```

**Motion planning examples:**
```python
# See: ros2_ws/src/ur10_control/scripts/motion_planning_example.py
```

Available planners:
- OMPL (default): RRT, PRM, EST, KPIECE
- Pilz Industrial Motion: Linear, circular paths
- TrajOpt (optional): Trajectory optimization

## Development Workflow

### Testing Computer Vision

**Quick test with Azure Kinect:**
```bash
# Test basic capture
python src/perception/cv_basics.py

# Test viewer
python -c "from src.perception.azure_kinect import KinectViewer; KinectViewer().run()"

# Test advanced algorithms
python src/perception/cv_scripts.py
```

**Test with saved images:**
```bash
# Capture some test images first
python src/perception/cv_basics.py
# Choose option 4 or 5 to save images

# Run algorithms on saved images
python src/perception/cv_scripts.py
# Choose option 2, enter folder path
```

### Testing Robot Control

**Safety checklist:**
1. Verify robot IP address
2. Ensure emergency stop is accessible
3. Clear workspace of obstacles
4. Start with low velocities (0.1-0.3 m/s)
5. Test in simulation first if available

**Connection test:**
```bash
# Ping robot
ping 192.168.1.101

# Test control script
python -c "from src.robot_control.ur10_controller import UR10Controller; print('Import successful')"
```

## Docker Environment

Docker is provided for Computer Vision dependencies only. ROS2 should be installed natively.

```bash
# Build container
docker-compose up ur10e-dev

# Or build specific image
docker build -t ur10e-dev .

# Run with display support (Linux)
xhost +local:docker
docker run -it --rm \
  --net=host \
  --privileged \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/workspace \
  ur10e-dev
```

## Common Issues

### Azure Kinect

**Issue: "Device is not opened"**
- Solution: Check USB 3.0 connection, install Azure Kinect SDK

**Issue: "No images found in folder"**
- Solution: Ensure images are JPG, PNG, or BMP format (case-insensitive)
- Check folder path is correct and contains images

**Issue: Point cloud appears inverted**
- Solution: This is fixed in current version (Y-axis flip applied)

### Robot Connection

**Issue: Cannot connect to robot**
- Check IP address: `ping 192.168.1.101`
- Verify network connection
- Ensure robot is powered on

**Issue: Motion rejected**
- Check safety limits in RobotConfig
- Verify joint limits not exceeded
- Ensure protective stop is not active

## Camera-Robot Calibration

For hand-eye calibration:

```python
from src.perception.azure_kinect import HandEyeCalibration

# Initialize calibration
calib = HandEyeCalibration(method='eye_to_hand')

# Collect calibration pairs (robot poses + camera images)
robot_poses = [...]  # Get from robot controller
images = [...]        # Capture with Kinect

# Compute transformation
T_camera_to_base = calib.calibrate(robot_poses, images)
```

## Project Organization

### For CV Group
- Work in `src/perception/`
- Use `cv_basics.py` for learning fundamentals
- Use `cv_scripts.py` for project implementation
- Use `kinect_scripts.py` for advanced depth processing
- Use `nerf_studio.py` for 3D reconstruction with NeRF
- Test with both live camera and saved images

### For Control Group
- Work in `ros2_ws/src/ur10_control/`
- Use MoveIt2 for motion planning
- Explore different planners (OMPL, Pilz, TrajOpt)
- Test in RViz before running on hardware

## Resources

### Documentation
- [UR10e Manual](https://www.universal-robots.com/download/)
- [MoveIt2 Tutorials](https://moveit.picknik.ai/main/index.html)
- [Azure Kinect DK Docs](https://docs.microsoft.com/en-us/azure/kinect-dk/)
- [OpenCV Tutorials](https://docs.opencv.org/4.x/d9/df8/tutorial_root.html)
- [Open3D Documentation](http://www.open3d.org/docs/release/)
- [NeRF Studio Documentation](https://docs.nerf.studio/)

### Libraries
- pyk4a: Azure Kinect Python wrapper
- OpenCV: Computer vision algorithms
- Open3D: 3D data processing
- MoveIt2: Robot motion planning
- OMPL: Motion planning library
- NeRF Studio: Neural radiance field training and rendering

## Safety Guidelines

1. Always verify robot IP before connecting
2. Never exceed safety-rated velocities
3. Keep emergency stop accessible
4. Clear workspace before motion
5. Test new code in simulation first (try to)

## Contributing

1. Create feature branch: `git checkout -b feature/your-feature`
2. Make changes and test thoroughly
3. Commit with clear messages
4. Push and create pull request

### Code Style
- Follow PEP 8 for Python
- Use type hints where applicable
- Add docstrings to all functions
- Comment complex algorithms
- Test all changes before committing
