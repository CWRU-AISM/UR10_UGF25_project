# Robotiq 2F-85 Gripper Setup

## Overview

This project controls the Robotiq 2F-85 gripper that is **connected to the UR10 robot** (not directly to the PC). The gripper is controlled by sending URScript commands to the robot on port 30002, using the Robotiq URCap function definitions.

## How It Works

The gripper is connected to the robot, and the Robotiq_Grippers URCap is installed on the robot. When you send commands through Python:

1. The Python code loads the Robotiq function definitions from `grippy.script`
2. These definitions (650+ lines) are sent along with each gripper command
3. The robot executes the complete script, which includes functions like:
   - `rq_activate_all_grippers()` - Activate the gripper
   - `rq_open_and_wait()` - Open gripper and wait for completion
   - `rq_close_and_wait()` - Close gripper and wait for completion
   - `rq_move_and_wait_norm()` - Move to specific position (0-100%)
   - `rq_is_object_detected()` - Check if object is gripped

## Usage

### Basic Usage

```python
from src.robot_control.ur10_control import URRobot, URConfig

config = URConfig(robot_ip="192.168.1.101")
robot = URRobot(config)

# Connect (robot should already be powered on)
robot.connect(activate_gripper=True, power_on_robot=False)

# Use gripper
robot.open_gripper()           # Fully open
robot.close_gripper()          # Fully close
robot.gripper_move_to(50)      # Move to 50% (half open)

# Check if object is gripped
robot.close_gripper()
if robot.rq_check_object_detected():
    print("Object gripped!")

robot.disconnect()
```

### Test Gripper Functionality

Run the gripper test demo:

```bash
python src/robot_control/ur10_control.py
# Select option 9: Gripper test
```

## Gripper Methods

### Main Methods:
- `open_gripper()` - Fully open the gripper
- `close_gripper()` - Fully close the gripper
- `gripper_move_to(position)` - Move to specific position (0-100%)
- `rq_move_gripper(position, speed, force)` - Move with custom speed/force
- `rq_check_object_detected()` - Check if object is gripped

### Position Scale:
- `0` = Fully open
- `100` = Fully closed
- `50` = Half open

## Requirements

1. **Robot Setup:**
   - UR10 robot with Robotiq 2F-85 gripper attached
   - Robotiq_Grippers URCap installed on robot
   - Robot powered on and in remote control mode

2. **Python Requirements:**
   - Python 3.x
   - numpy
   - Standard library modules (socket, time, threading, etc.)
   - No external gripper libraries needed!

3. **Files:**
   - `grippy.script` must be in `src/robot_control/` directory
   - This file contains the Robotiq URCap function definitions

## Troubleshooting

### Gripper Not Moving

1. **Check URCap Installation:**
   - Open PolyScope on the teach pendant
   - Go to "Installation" → "URCaps"
   - Verify "Robotiq_Grippers" is installed

2. **Check grippy.script:**
   - Ensure `grippy.script` exists in `src/robot_control/`
   - Should be ~1170 lines containing Robotiq functions

3. **Manual Activation:**
   - Try activating the gripper from the teach pendant first
   - Use the Robotiq interface in PolyScope
   - Then try Python control

4. **Check Robot Connection:**
   - Ensure robot is in remote control mode
   - Check TCP/IP connection on port 30002
   - Verify robot is not in a protective stop

### Function Definitions Not Found

If you see "Error: Gripper functions not loaded":
- Check that `grippy.script` is in the correct location
- The script should contain the line: `#   Source: Robotiq_Grippers`
- Try regenerating `grippy.script` from a robot program with gripper commands

### Commands Not Executing

If gripper commands seem to be ignored:
- The robot may be in local control mode (switch to remote)
- Check for errors in the robot log
- Ensure no other programs are running on the robot
- Try reducing command frequency (add delays between commands)

## Technical Details

### Port Information:
- **Primary Control**: Port 30002 (URScript commands)
- **Dashboard**: Port 29999 (Power on/off, brake release)
- **RTDE**: Port 30004 (Real-time data exchange)

### URScript Approach:
Unlike direct serial/Modbus communication with the gripper, this approach:
- Sends complete URScript programs including function definitions
- Executes on the robot's controller
- Works with gripper connected to robot tool flange
- Requires Robotiq URCap to be installed

### Why Not pyRobotiqGripper?:
- `pyRobotiqGripper` is for **direct PC-to-gripper serial/USB connection**
- Our gripper is connected to the **robot**, not the PC
- We use URScript with URCap functions instead

## Examples

### Pick and Place with Gripper

```python
robot = URRobot(URConfig(robot_ip="192.168.1.101"))
robot.connect(activate_gripper=True, power_on_robot=False)

# Move to pick position
robot.move_linear([-0.4, 0.3, 0.1, 3.14, 0, 0])

# Open gripper
robot.open_gripper()

# Move down
robot.move_relative([0, 0, -0.05, 0, 0, 0])

# Close gripper
robot.close_gripper()

# Check if object was gripped
if robot.rq_check_object_detected():
    print("Object picked!")
    # Move up
    robot.move_relative([0, 0, 0.05, 0, 0, 0])
else:
    print("Failed to grip object")

robot.disconnect()
```

### Controlled Grip Force

```python
# Move to position with specific force and speed
robot.rq_move_gripper(
    position=80,  # 80% closed
    speed=50,     # 50% speed
    force=30      # 30% force (gentle grip)
)
```

## Notes

- Commands include built-in wait times for motion completion
- The `rq_*_and_wait()` functions block until gripper motion completes
- Gripper position is 0-100% (not 0-255 like some other interfaces)
- Object detection uses force sensors in the gripper fingers
- The gripper communicates with the robot via the tool communication interface

## Advanced Usage

For more complex scenarios, you can call URCap functions directly:

```python
commands = """
    rq_set_force_norm(50, "1")
    rq_set_speed_norm(100, "1")
    rq_move_and_wait_norm(75, "1")
    detected = rq_is_object_detected("1")
    if detected:
        textmsg("Object detected!")
    end
"""
robot._send_gripper_script(commands)
```

##ROS2 Integration

The ROS2 workspace (`ros2_ws/`) currently only handles robot arm motion planning with MoveIt2. To add gripper control:

1. Create a ROS2 node that wraps the Python gripper control
2. Publish gripper state to topics
3. Create action/service interfaces for gripper commands
4. Coordinate with MoveIt2 for synchronized arm + gripper motions

See `ros2_ws/src/ur10_control/` for existing robot control examples.
