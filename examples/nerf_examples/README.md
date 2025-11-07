# NeRF Studio Examples

This directory contains example scripts for working with NeRF Studio integration in the UR10 project.

## Overview

Neural Radiance Fields (NeRF) enable photo-realistic 3D reconstruction from multi-view images. These examples demonstrate how to capture, process, and train NeRF models using the Azure Kinect camera and optionally the UR10 robot arm.

## Prerequisites

Install NeRF Studio dependencies:

```bash
pip install nerfstudio nerfacc
```

For GPU acceleration (recommended):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Workflow

### 1. Data Capture

Capture multi-view images of an object or scene.

**Manual capture (handheld camera):**

```bash
python examples/nerf_examples/basic_nerf_workflow.py
```

**Quick capture (50 images):**

```bash
python examples/nerf_examples/basic_nerf_workflow.py --quick
```

**Automated capture (with robot):**

```bash
python examples/nerf_examples/robot_nerf_capture.py \
  --robot-ip 192.168.1.101 \
  --object-center 0.4 0.3 0.2 \
  --radius 0.5
```

### 2. Training

Train a NeRF model from captured images.

```bash
python examples/nerf_examples/train_nerf_workflow.py data/nerf_basic
```

**Available methods:**
- `nerfacto` - Default, balanced quality and speed
- `instant-ngp` - Fast training, requires CUDA
- `tensorf` - High quality, slower training
- `mipnerf` - Anti-aliasing, better fine details

**Custom training:**

```bash
python examples/nerf_examples/train_nerf_workflow.py data/nerf_basic \
  --method instant-ngp \
  --iterations 50000
```

### 3. Viewing Results

View the trained model interactively:

```bash
ns-viewer --load-config outputs/nerfacto_*/config.yml
```

Access the viewer at `http://localhost:7007`

### 4. Export

Export to mesh or point cloud:

```bash
ns-export tsdf \
  --load-config outputs/nerfacto_*/config.yml \
  --output-dir exports/mesh
```

```bash
ns-export pointcloud \
  --load-config outputs/nerfacto_*/config.yml \
  --output-dir exports/pointcloud \
  --num-points 1000000
```

## Script Details

### basic_nerf_workflow.py

Manual data capture workflow with Azure Kinect.

**Features:**
- Interactive capture with live preview
- Adjustable number of images and capture interval
- Automatic generation of transforms.json
- Quick capture mode for rapid prototyping

**Usage:**

```bash
python basic_nerf_workflow.py
python basic_nerf_workflow.py --quick --num-images 50
python basic_nerf_workflow.py --output data/my_scene
```

### train_nerf_workflow.py

Train NeRF models from captured data.

**Features:**
- Multiple training methods (nerfacto, instant-ngp, etc.)
- Configurable iteration count
- Automatic output organization

**Usage:**

```bash
python train_nerf_workflow.py data/nerf_basic
python train_nerf_workflow.py data/nerf_basic --method instant-ngp --iterations 50000
```

### robot_nerf_capture.py

Automated capture using the robot arm.

**Features:**
- Hemisphere capture pattern
- Configurable elevation angles
- Automatic viewpoint calculation
- Safety checks and confirmations

**Usage:**

```bash
python robot_nerf_capture.py
python robot_nerf_capture.py --object-center 0.5 0.3 0.25 --radius 0.6
```

## Tips for Best Results

### Capture Guidelines

1. **Coverage**: Capture images from all angles around the object
2. **Overlap**: Ensure 60-80% overlap between consecutive images
3. **Lighting**: Use consistent, diffuse lighting
4. **Distance**: Maintain consistent distance from the object
5. **Quantity**: More images = better quality (100-200 recommended)

### Common Issues

**Blurry reconstruction:**
- Increase number of training iterations
- Ensure images are sharp and well-focused
- Check for motion blur during capture

**Missing geometry:**
- Capture more viewpoints from different angles
- Ensure good coverage of the object
- Check that all views are in transforms.json

**Poor texture quality:**
- Improve lighting during capture
- Use higher resolution images
- Try different NeRF methods (e.g., nerfacto vs instant-ngp)

## Advanced Usage

### Custom Capture Patterns

Modify `robot_nerf_capture.py` to implement custom capture patterns:

```python
from src.perception.nerf_studio import RobotNeRFCapture

robot_capture = RobotNeRFCapture("data/custom")
robot_capture.connect()

# Custom viewpoint loop
for angle in custom_angles:
    position = compute_position(angle)
    transform = compute_transform(position)
    robot_capture.capture.capture_frame(transform_matrix=transform)

robot_capture.capture.save_transforms()
robot_capture.disconnect()
```

### Processing Existing Images

If you have existing images with known camera poses:

```python
from src.perception.nerf_studio import NeRFDataCapture
import numpy as np

capture = NeRFDataCapture("data/existing_images")

# Set camera intrinsics manually
capture.camera_intrinsics = {
    'width': 1920,
    'height': 1080,
    'fx': 1000.0,
    'fy': 1000.0,
    'cx': 960.0,
    'cy': 540.0
}

# Add frames with known transforms
for i, (image_path, transform) in enumerate(image_data):
    capture.frames.append({
        "file_path": image_path,
        "transform_matrix": transform.tolist()
    })

capture.save_transforms()
```

## References

- [NeRF Studio Documentation](https://docs.nerf.studio/)
- [NeRF: Representing Scenes as Neural Radiance Fields](https://www.matthewtancik.com/nerf)
- [Instant Neural Graphics Primitives](https://nvlabs.github.io/instant-ngp/)
