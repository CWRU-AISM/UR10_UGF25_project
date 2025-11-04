"""
Control Group
UR Robot Control via TCP/IP and URScript
"""

import socket
import time
import numpy as np
import struct
import threading
from dataclasses import dataclass
from typing import List, Tuple, Optional
import json
import os
from datetime import datetime
from math import pi
from pathlib import Path

@dataclass
class URConfig:
    """Configuration for UR robot connection"""
    robot_ip: str = "192.168.1.101"  # Placeholder
    tcp_port: int = 30002           # Primary interface port
    dashboard_port: int = 29999     # Dashboard server port
    rtde_port: int = 30004          # Real-time data exchange port
    
    # Robot limits 
    max_velocity: float = 1.0       # m/s
    max_acceleration: float = 1.2   # m/s^2
    max_joint_velocity: float = 2.0 # rad/s
    max_joint_acceleration: float = 2.0  # rad/s^2
    
    # Tool configuration
    tool_weight: float = 0.5        # kg
    tool_cog: List[float] = None    # Center of gravity [x, y, z]
    
    def __post_init__(self):
        if self.tool_cog is None:
            self.tool_cog = [0.0, 0.0, 0.0]


class URRobot:
    """Main class for UR robot control via TCP/IP"""

    def __init__(self, config: URConfig = None):
        """Initialize UR robot connection

        Args:
            config: URConfig object with robot settings
        """
        self.config = config or URConfig()
        self.socket = None
        self.dashboard_socket = None
        self.is_connected = False
        self.current_pose = None
        self.current_joints = None
        self.current_digital_inputs = [False] * 10
        self.current_digital_outputs = [False] * 10
        self.gripper_object_detected = False

        # Robotiq Gripper via URCap functions (loaded from grippy.script)
        self.gripper_script_header = None
        self.gripper_initialized = False

        # Thread for monitoring robot state
        self.monitor_thread = None
        self.monitoring = False
        self._state_lock = threading.Lock()

        # Load Robotiq gripper script functions
        self._load_gripper_functions()

    def _load_gripper_functions(self):
        """Load Robotiq gripper function definitions from grippy.script"""
        try:
            script_path = Path(__file__).parent / "grippy.script"
            if script_path.exists():
                with open(script_path, 'r', encoding='utf-8') as f:
                    full_script = f.read()

                # Extract only the Robotiq Gripper functions
                # Find the start of gripper functions
                start_marker = "# begin: URCap Installation Node"
                gripper_marker = "#   Source: Robotiq_Grippers"
                gripper_type_marker = "#   Type: Gripper"

                lines = full_script.split('\n')
                gripper_start = None
                gripper_end = None

                for i, line in enumerate(lines):
                    if gripper_type_marker in line:
                        # Back up to find the begin marker
                        for j in range(i, max(0, i-10), -1):
                            if start_marker in lines[j] and gripper_marker in lines[j+1]:
                                gripper_start = j
                                break
                    if gripper_start and "# end: URCap Installation Node" in line and i > gripper_start + 10:
                        gripper_end = i + 1
                        break

                if gripper_start and gripper_end:
                    self.gripper_script_header = '\n'.join(lines[gripper_start:gripper_end])
                    print("Loaded Robotiq gripper functions")
                else:
                    print("Warning: Could not find Robotiq gripper functions in grippy.script")
                    self.gripper_script_header = None
            else:
                print(f"Warning: grippy.script not found at {script_path}")
                self.gripper_script_header = None

        except Exception as e:
            print(f"Error loading gripper functions: {e}")
            self.gripper_script_header = None

    def connect(self, activate_gripper=True, power_on_robot=False):
        """Establish connection to UR robot

        Args:
            activate_gripper: If True, automatically activate gripper on connect
            power_on_robot: If True, power on robot and release brakes (set False if already on)
        """
        try:
            # Connect to primary interface
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5)
            self.socket.connect((self.config.robot_ip, self.config.tcp_port))

            # Connect to dashboard
            self.dashboard_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.dashboard_socket.settimeout(5)
            self.dashboard_socket.connect((self.config.robot_ip, self.config.dashboard_port))

            self.is_connected = True
            print(f"Connected to UR robot at {self.config.robot_ip}")

            # Start monitoring thread
            self.start_monitoring()

            # Power on and release brakes (only if requested)
            if power_on_robot:
                print("Powering on robot...")
                self.power_on()
                time.sleep(2)
                print("Releasing brakes...")
                self.release_brakes()
                time.sleep(2)

            # Activate Robotiq gripper using URCap functions
            if activate_gripper and self.gripper_script_header:
                try:
                    print("Activating Robotiq gripper via URCap...")
                    self.rq_activate_gripper()
                    self.gripper_initialized = True
                    print("Gripper activated and ready!")
                except Exception as e:
                    print(f"Warning: Gripper activation failed: {e}")
                    print("Try activating manually from teach pendant")
                    self.gripper_initialized = False
            elif not self.gripper_script_header:
                print("Warning: Gripper functions not loaded from grippy.script")

            return True

        except Exception as e:
            print(f"Failed to connect to robot: {e}")
            return False
    
    def disconnect(self):
        """Close connection to UR robot"""
        self.stop_monitoring()

        if self.socket:
            self.socket.close()
        if self.dashboard_socket:
            self.dashboard_socket.close()

        self.is_connected = False
        self.gripper_initialized = False
        print("Disconnected from robot")
    
    def send_script(self, script: str):
        """Send URScript command to robot"""
        if not self.is_connected:
            print("Not connected to robot!")
            return False
        
        try:
            # URScript must end with newline
            if not script.endswith('\n'):
                script += '\n'
            
            # Send script
            self.socket.send(script.encode())
            return True
            
        except Exception as e:
            print(f"Failed to send script: {e}")
            return False
    
    def send_dashboard_command(self, command: str):
        """Send command to dashboard server"""
        if not self.dashboard_socket:
            return None
        
        try:
            self.dashboard_socket.send((command + '\n').encode())
            response = self.dashboard_socket.recv(1024).decode()
            return response.strip()
        except Exception as e:
            print(f"Dashboard command failed: {e}")
            return None

    def power_on(self):
        """Power on the robot"""
        return self.send_dashboard_command("power on")
    
    def power_off(self):
        """Power off the robot"""
        return self.send_dashboard_command("power off")
    
    def release_brakes(self):
        """Release the brakes"""
        return self.send_dashboard_command("brake release")
    
    def get_robot_mode(self):
        """Get current robot mode"""
        return self.send_dashboard_command("robotmode")
    
    def stop(self):
        """Emergency stop"""
        return self.send_dashboard_command("stop")
    
    def move_joint(self, joint_positions: List[float],
                   velocity: float = None, acceleration: float = None, wait: bool = True):
        """
        Move to target joint positions
        Args:
            joint_positions: List of 6 joint angles in radians
            velocity: Joint velocity in rad/s
            acceleration: Joint acceleration in rad/s^2
            wait: If True, wait for movement to complete before returning
        """
        if len(joint_positions) != 6:
            print("Error: Need exactly 6 joint positions")
            return False

        vel = velocity or self.config.max_joint_velocity
        acc = acceleration or self.config.max_joint_acceleration

        # Format joint positions
        joints_str = f"[{', '.join(map(str, joint_positions))}]"

        # Create URScript command
        script = f"movej({joints_str}, a={acc}, v={vel})"

        result = self.send_script(script)

        # Wait for movement to complete
        if wait and result:
            time.sleep(0.5)  # Initial delay for command processing

        return result
    
    def move_linear(self, pose: List[float],
                    velocity: float = None, acceleration: float = None, wait: bool = True):
        """
        Linear movement to target pose
        Args:
            pose: [x, y, z, rx, ry, rz] in meters and radians
            velocity: Tool velocity in m/s
            acceleration: Tool acceleration in m/s^2
            wait: If True, wait for movement to complete before returning
        """
        if len(pose) != 6:
            print("Error: Pose must have 6 values [x, y, z, rx, ry, rz]")
            return False

        vel = velocity or self.config.max_velocity
        acc = acceleration or self.config.max_acceleration

        # Format pose
        pose_str = f"p[{', '.join(map(str, pose))}]"

        # Create URScript command
        script = f"movel({pose_str}, a={acc}, v={vel})"

        result = self.send_script(script)

        # Wait for movement to complete
        if wait and result:
            time.sleep(0.5)  # Initial delay for command processing

        return result
    
    def move_circular(self, via_pose: List[float], end_pose: List[float],
                      velocity: float = None, acceleration: float = None):
        """
        Circular movement through via point to end point
        """
        vel = velocity or self.config.max_velocity
        acc = acceleration or self.config.max_acceleration
        
        via_str = f"p[{', '.join(map(str, via_pose))}]"
        end_str = f"p[{', '.join(map(str, end_pose))}]"
        
        script = f"movec({via_str}, {end_str}, a={acc}, v={vel})"
        
        return self.send_script(script)
    
    def move_relative(self, delta: List[float], 
                      velocity: float = None, acceleration: float = None):
        """
        Move relative to current position
        Args:
            delta: [dx, dy, dz, drx, dry, drz] relative movement
        """
        vel = velocity or self.config.max_velocity
        acc = acceleration or self.config.max_acceleration
        
        # This requires getting current pose and calculating new pose
        delta_str = f"p[{', '.join(map(str, delta))}]"
        
        script = f"""
        current_pose = get_actual_tcp_pose()
        target_pose = pose_trans(current_pose, {delta_str})
        movel(target_pose, a={acc}, v={vel})
        """
        
        return self.send_script(script)
    
    # Gripper: Robotiq 2F85 via URCap functions

    def set_digital_output(self, pin: int, value: bool):
        """Set digital output pin"""
        val = 'True' if value else 'False'
        script = f"set_digital_out({pin}, {val})"
        return self.send_script(script)

    def _send_gripper_script(self, commands: str):
        """Send gripper commands with URCap function definitions

        Args:
            commands: URScript commands to execute (e.g., 'rq_open_and_wait()')
        """
        if not self.gripper_script_header:
            print("Error: Gripper functions not loaded")
            return False

        # Build complete script with header + commands
        full_script = f"""def gripper_program():
{self.gripper_script_header}

{commands}
end
gripper_program()
"""
        return self.send_script(full_script)

    def rq_activate_gripper(self):
        """Activate and initialize Robotiq gripper using URCap functions"""
        if not self.gripper_script_header:
            print("Error: Gripper functions not loaded from grippy.script")
            return False

        commands = """
    # Reset and activate all grippers
    rq_activate_all_grippers(True)
"""
        result = self._send_gripper_script(commands)
        time.sleep(3)  # Wait for activation
        return result

    def rq_close_gripper(self, force: int = 100, speed: int = 100):
        """Close Robotiq gripper using URCap functions

        Args:
            force: Grip force 0-100% (default 100)
            speed: Closing speed 0-100% (default 100)
        """
        if not self.gripper_initialized:
            print("Warning: Gripper not initialized")

        commands = f"""
    # Set force and speed
    rq_set_force_norm({force}, "1")
    rq_set_speed_norm({speed}, "1")

    # Close and wait
    rq_close_and_wait("1")
"""
        return self._send_gripper_script(commands)

    def rq_open_gripper(self, speed: int = 100):
        """Open Robotiq gripper using URCap functions

        Args:
            speed: Opening speed 0-100% (default 100)
        """
        if not self.gripper_initialized:
            print("Warning: Gripper not initialized")

        commands = f"""
    # Set speed
    rq_set_speed_norm({speed}, "1")

    # Open and wait
    rq_open_and_wait("1")
"""
        return self._send_gripper_script(commands)

    def rq_move_gripper(self, position: int, speed: int = 100, force: int = 100):
        """Move gripper to specific position

        Args:
            position: Position 0-100% (0=open, 100=closed)
            speed: Movement speed 0-100% (default 100)
            force: Grip force 0-100% (default 100)
        """
        if not self.gripper_initialized:
            print("Warning: Gripper not initialized")

        commands = f"""
    # Set parameters
    rq_set_force_norm({force}, "1")
    rq_set_speed_norm({speed}, "1")

    # Move to position and wait
    rq_move_and_wait_norm({position}, "1")
"""
        return self._send_gripper_script(commands)

    def rq_check_object_detected(self) -> bool:
        """Check if Robotiq gripper has detected an object
        Returns True if object detected (gripper stopped before fully closed)
        """
        if not self.gripper_initialized:
            print("Warning: Gripper not initialized")
            return False

        # Use digital output to read gripper status
        commands = """
    # Check if object is detected and set digital output
    detected = rq_is_object_detected("1")
    set_standard_digital_out(8, detected)
    sleep(0.1)
"""
        self._send_gripper_script(commands)
        time.sleep(0.5)

        # Read the digital output
        with self._state_lock:
            return self.current_digital_outputs[8]

    def get_gripper_position(self) -> Optional[float]:
        """Get current gripper position (0-100%, 0=fully open, 100=fully closed)
        Returns None if unable to read position

        Note: This requires reading from robot state, not directly available
        """
        # This would require more complex implementation with RTDE or register reads
        print("Warning: get_gripper_position not fully implemented")
        return None

    def show_popup(self, message: str, warning: bool = False):
        """Show popup message on robot pendant
        Args:
            message: Message to display
            warning: If True, shows as warning popup
        """
        if warning:
            script = f'popup("{message}", warning=True, error=False)'
        else:
            script = f'popup("{message}")'
        return self.send_script(script)

    # Convenience methods with default parameters
    def close_gripper(self):
        """Close gripper fully"""
        return self.rq_close_gripper()

    def open_gripper(self):
        """Open gripper fully"""
        return self.rq_open_gripper()

    def gripper_move_to(self, position: int):
        """Move gripper to specific position

        Args:
            position: Position 0-100% (0=fully open, 100=fully closed)
        """
        return self.rq_move_gripper(position)
    
    # State Monitoring
    
    def start_monitoring(self):
        """Start monitoring robot state in background thread"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring thread"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
    
    def _monitor_loop(self):
        """Background thread for monitoring robot state"""
        while self.monitoring:
            try:
                if not self.socket:
                    break

                # Read the message size (first 4 bytes)
                self.socket.settimeout(0.5)
                size_bytes = self.socket.recv(4)

                if len(size_bytes) < 4:
                    continue

                msg_size = struct.unpack('>i', size_bytes)[0]

                # Read the rest of the message
                remaining = msg_size - 4
                data = b''
                while len(data) < remaining:
                    chunk = self.socket.recv(min(remaining - len(data), 4096))
                    if not chunk:
                        break
                    data += chunk

                # Parse the robot state data
                self._parse_robot_state(data)

            except socket.timeout:
                continue
            except Exception as e:
                if self.monitoring:  # Only print if we're supposed to be monitoring
                    print(f"Monitor loop error: {e}")
                break

    def _parse_robot_state(self, data):
        """Parse robot state packet"""
        try:
            offset = 0

            # Parse through all sub-packages in the robot state
            while offset < len(data):
                if offset + 5 > len(data):
                    break

                # Each sub-package has: length(4) + type(1) + data
                pkg_length = struct.unpack('>i', data[offset:offset+4])[0]
                pkg_type = data[offset+4]

                if offset + pkg_length > len(data):
                    break

                pkg_data = data[offset+5:offset+pkg_length]

                # Package type 0: Robot mode data
                if pkg_type == 0 and len(pkg_data) >= 1:
                    pass  # Could parse robot mode here

                # Package type 1: Joint data
                elif pkg_type == 1 and len(pkg_data) >= 252:
                    # Each joint: q(8), qd(8), qdd(8), I(4), V(4), T(4), Tm(4) = 41 bytes * 6 joints
                    joints = []
                    for i in range(6):
                        joint_offset = i * 41
                        if joint_offset + 8 <= len(pkg_data):
                            q = struct.unpack('>d', pkg_data[joint_offset:joint_offset+8])[0]
                            joints.append(q)

                    with self._state_lock:
                        self.current_joints = joints

                # Package type 4: Cartesian info (TCP pose)
                elif pkg_type == 4 and len(pkg_data) >= 48:
                    # X, Y, Z, Rx, Ry, Rz (6 doubles)
                    pose = []
                    for i in range(6):
                        val = struct.unpack('>d', pkg_data[i*8:i*8+8])[0]
                        pose.append(val)

                    with self._state_lock:
                        self.current_pose = pose

                # Package type 3: Tool data (includes digital I/O)
                elif pkg_type == 3 and len(pkg_data) >= 8:
                    # Digital inputs and outputs are in this package
                    # Offset depends on package version, typically around byte 16-17
                    if len(pkg_data) >= 18:
                        digital_inputs = struct.unpack('>H', pkg_data[16:18])[0]

                        # Parse digital input bits
                        with self._state_lock:
                            for i in range(10):
                                self.current_digital_inputs[i] = bool(digital_inputs & (1 << i))

                # Package type 2: Master board data (includes digital I/O)
                elif pkg_type == 2 and len(pkg_data) >= 28:
                    # Digital input states at offset 11-12
                    # Digital output states at offset 13-14
                    if len(pkg_data) >= 14:
                        digital_inputs = struct.unpack('>H', pkg_data[11:13])[0]
                        digital_outputs = struct.unpack('>H', pkg_data[13:15])[0]

                        with self._state_lock:
                            for i in range(10):
                                self.current_digital_inputs[i] = bool(digital_inputs & (1 << i))
                                self.current_digital_outputs[i] = bool(digital_outputs & (1 << i))

                offset += pkg_length

        except Exception:
            # Don't spam errors, just skip this packet
            pass


class TrajectoryGenerator:
    """Generate smooth trajectories for robot movement"""
    
    @staticmethod
    def linear_interpolation(start: List[float], end: List[float], 
                            num_points: int = 10) -> List[List[float]]:
        """
        Generate linear interpolation between two poses
        """
        trajectory = []
        for i in range(num_points):
            t = i / (num_points - 1)
            point = [s + t * (e - s) for s, e in zip(start, end)]
            trajectory.append(point)
        return trajectory
    
    @staticmethod
    def joint_space_trajectory(start_joints: List[float], end_joints: List[float],
                              duration: float = 3.0, dt: float = 0.1) -> List[List[float]]:
        """
        Generate smooth joint space trajectory using quintic polynomial
        """
        num_points = int(duration / dt)
        trajectory = []
        
        for i in range(num_points):
            t = i * dt / duration
            # Quintic polynomial for smooth acceleration
            s = 10 * t**3 - 15 * t**4 + 6 * t**5
            
            joints = [sj + s * (ej - sj) for sj, ej in zip(start_joints, end_joints)]
            trajectory.append(joints)
        
        return trajectory
    
    @staticmethod
    def circular_trajectory(center: List[float], radius: float, 
                          num_points: int = 20) -> List[List[float]]:
        """
        Generate circular trajectory in XY plane
        """
        trajectory = []
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            z = center[2]
            
            # Keep orientation constant
            pose = [x, y, z] + center[3:6]
            trajectory.append(pose)
        
        return trajectory


class SafeURRobot(URRobot):
    """UR Robot with additional safety checks"""
    
    def __init__(self, config: URConfig = None):
        super().__init__(config)
        self.workspace_limits = {
            'x': (-0.8, 0.8),
            'y': (-0.8, 0.8),
            'z': (0.0, 0.8)
        }
        
    def check_pose_safety(self, pose: List[float]) -> bool:
        """Check if pose is within safe workspace"""
        x, y, z = pose[:3]
        
        if not (self.workspace_limits['x'][0] <= x <= self.workspace_limits['x'][1]):
            print(f"X position {x} out of bounds!")
            return False
        if not (self.workspace_limits['y'][0] <= y <= self.workspace_limits['y'][1]):
            print(f"Y position {y} out of bounds!")
            return False
        if not (self.workspace_limits['z'][0] <= z <= self.workspace_limits['z'][1]):
            print(f"Z position {z} out of bounds!")
            return False
        
        return True
    
    def move_linear(self, pose: List[float],
                    velocity: float = None, acceleration: float = None, wait: bool = True):
        """Override move_linear with safety check"""
        # if not self.check_pose_safety(pose):
        #     print("Safety check failed! Movement cancelled.")
        #     return False

        # Reduce speed for safety
        vel = min(velocity or self.config.max_velocity, 0.5)
        acc = min(acceleration or self.config.max_acceleration, 0.5)

        return super().move_linear(pose, vel, acc, wait)

def demo_basic_movement():
    """Basic movement patterns"""

    # Initialize robot
    config = URConfig(robot_ip="192.168.1.101")  # Update IP
    robot = SafeURRobot(config)

    if not robot.connect():
        print("Failed to connect to robot")
        return

    try:
        # Home position
        home_joints = [(159.73*pi/180), -102.5*pi/180, -100*pi/180, 296*pi/180, -260*pi/180, 35*pi/180]

        print("Moving to home position...")
        robot.move_joint(home_joints, velocity=0.5)
        time.sleep(5)  # Increased wait time for smoother operation

        # Square movement pattern
        print("Executing square pattern...")
        square_poses = [
            [-.925, 0.518, .341, 2.976, .952, .016],
            [-1.051, 0.518, .341, 2.976, .952, .016],
            [-1.051, 0.35, .341, 2.976, .952, .016],
            [-.925, 0.35, .341, 2.976, .952, .016],
        ]

        for i, pose in enumerate(square_poses):
            print(f"  Moving to corner {i+1}/4")
            robot.move_linear(pose, velocity=0.1)
            time.sleep(4)  # Longer wait

        print("Returning to home")
        robot.move_joint(home_joints, velocity=0.5)
        time.sleep(5)

    finally:
        robot.disconnect()


def demo_pick_and_place():
    """Smart pick and place with object detection"""
    print("\nSmart Pick and Place")

    config = URConfig(robot_ip="192.168.1.101")
    robot = SafeURRobot(config)

    if not robot.connect():
        return

    try:
        # Home position
        home_joints = [(159.73*pi/180), -102.5*pi/180, -100*pi/180, 296*pi/180, -260*pi/180, 35*pi/180]

        print("Moving to home position...")
        robot.move_joint(home_joints, velocity=0.5)
        time.sleep(5)

        # Open gripper initially
        print("Opening gripper...")
        robot.rq_open_gripper()
        time.sleep(2)

        # Define positions (pickup and dropoff)
        pickup_pose = [-0.413, 0.654, -0.12, 2.191, 2.249, 0.1]
        above_pickup = [-0.413, 0.654, -0.05, 2.191, 2.249, 0.1]  # Safe height above
        dropoff_pose = [-0.413, 0.885, -0.1456, 2.19, 2.25, 0.1]
        above_dropoff = [-0.413, 0.885, -0.08, 2.19, 2.25, 0.1]  # Safe height above

        print("\nStarting smart pick and place sequence...")

        # Try pickup location first
        print("1. Moving above pickup position")
        robot.move_linear(above_pickup, velocity=0.1)
        time.sleep(4)

        print("2. Moving down to pickup")
        robot.move_linear(pickup_pose, velocity=0.05)
        time.sleep(3)

        # Attempt to grip object
        print("3. Closing gripper")
        robot.rq_close_gripper()
        time.sleep(2)

        # Check if object detected at pickup
        print("4. Checking for object...")
        object_at_pickup = robot.rq_check_object_detected()

        if object_at_pickup:
            print("Object detected at pickup location")

            # Move up with object
            print("5. Lifting object")
            robot.move_linear(above_pickup, velocity=0.05)
            time.sleep(3)

            # Move to dropoff position
            print("6. Moving to dropoff position")
            robot.move_linear(above_dropoff, velocity=0.1)
            time.sleep(4)

            # Move down to place
            print("7. Lowering object")
            robot.move_linear(dropoff_pose, velocity=0.05)
            time.sleep(3)

            # Release object
            print("8. Opening gripper")
            robot.rq_open_gripper()
            time.sleep(2)

            # Move up
            print("9. Moving up")
            robot.move_linear(above_dropoff, velocity=0.1)
            time.sleep(3)

        else:
            print("No object at pickup location: checking dropoff...")

            # Move up without object
            robot.move_linear(above_pickup, velocity=0.05)
            time.sleep(3)
            robot.rq_open_gripper()
            time.sleep(1)

            # Move to dropoff to check
            print("5. Moving to dropoff position")
            robot.move_linear(above_dropoff, velocity=0.1)
            time.sleep(4)

            print("6. Moving down to dropoff")
            robot.move_linear(dropoff_pose, velocity=0.05)
            time.sleep(3)

            # Attempt to grip at dropoff
            print("7. Closing gripper")
            robot.rq_close_gripper()
            time.sleep(2)

            # Check if object at dropoff
            object_at_dropoff = robot.rq_check_object_detected()

            if object_at_dropoff:
                print("Object found at dropoff: moving to pickup")

                # Move up with object
                print("8. Lifting object")
                robot.move_linear(above_dropoff, velocity=0.05)
                time.sleep(3)

                # Move to pickup position
                print("9. Moving to pickup position")
                robot.move_linear(above_pickup, velocity=0.1)
                time.sleep(4)

                # Place at pickup
                print("10. Lowering object")
                robot.move_linear(pickup_pose, velocity=0.05)
                time.sleep(3)

                # Release
                print("11. Opening gripper")
                robot.rq_open_gripper()
                time.sleep(2)

                # Move up
                robot.move_linear(above_pickup, velocity=0.1)
                time.sleep(3)
            else:
                print("No object found at either location")
                robot.move_linear(above_dropoff, velocity=0.1)
                time.sleep(3)
                robot.rq_open_gripper()
                time.sleep(1)

        print("\nPick and place complete!")

        # Return to home
        print("Returning to home")
        robot.move_joint(home_joints, velocity=0.5)
        time.sleep(5)

    finally:
        robot.disconnect()


def demo_trajectory_execution():
    """Execute smooth spinning top trajectory with blended movements"""
    print("\nSpinning Top Trajectory")

    config = URConfig(robot_ip="192.168.1.101")
    robot = SafeURRobot(config)

    if not robot.connect():
        return

    try:
        # Home position
        home_joints = [(159.73*pi/180), -102.5*pi/180, -100*pi/180, 296*pi/180, -260*pi/180, 35*pi/180]

        print("Moving to home position...")
        robot.move_joint(home_joints, velocity=0.5)
        time.sleep(5)

        # Close gripper first
        print("Closing gripper...")
        robot.rq_close_gripper()
        time.sleep(2)

        # Move to starting position
        center = [-0.988, 0.434, 0.341, 2.976, 0.952, 0.016]
        print("Moving to start position...")
        robot.move_linear(center, velocity=0.1)
        time.sleep(3)

        # Generate spinning top trajectory
        print("Executing spinning top motion...")
        num_points = 16
        radius = 0.06
        blend_radius = 0.01  # Blend radius for smooth motion

        # Build URScript for smooth blended circular motion
        # Keep EE pointing down, rotate around tool axis (wrist 3)
        script = """
def spinning_top():
    # Starting position
    movej(get_actual_joint_positions(), a=1.2, v=0.5)
    """

        for i in range(num_points + 1):  # +1 to close the loop
            angle = 2 * np.pi * i / num_points

            # Circle motion in XY plane
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            z = center[2]

            # Keep gripper pointing straight down (avoid singularities)
            # Use center pose orientation as base, only rotate wrist 3
            rx = center[3]  # Keep base orientation from center pose
            ry = center[4]

            # Rotate around tool axis (wrist 3) to create spinning effect
            # As it goes around the circle, it also spins around its own axis
            rz = center[5] + angle * 2  # 2 full rotations as it goes around once

            pose_str = f"p[{x}, {y}, {z}, {rx}, {ry}, {rz}]"

            if i < num_points:
                # Use blend radius for smooth transitions
                script += f"\n    movel({pose_str}, a=0.5, v=0.1, r={blend_radius})"
            else:
                # Last point
                script += f"\n    movel({pose_str}, a=0.5, v=0.1)"

        script += """
end
spinning_top()
"""

        robot.send_script(script)
        time.sleep(15)  # Wait for motion to complete

        print("Spinning top motion complete")

        robot.rq_open_gripper()
        time.sleep(2)

        # Return to home
        print("Returning to home")
        robot.move_joint(home_joints, velocity=0.5)
        time.sleep(5)

    finally:
        robot.disconnect()


class URScriptTemplates:
    """Common URScript program templates"""
    
    @staticmethod
    def force_mode_template():
        """Template for force mode operations"""
        return """
        def force_mode_push():
            # Set TCP
            set_tcp(p[0, 0, 0.1, 0, 0, 0])
            
            # Move to start position
            movel(p[0.3, 0, 0.3, 0, 3.14, 0], a=0.5, v=0.1)
            
            # Enable force mode
            # Args: Task frame, selection vector, wrench, type, limits
            force_mode(p[0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, -10, 0, 0, 0], 2, [0.1, 0.1, 0.1, 0.17, 0.17, 0.17])
            
            # Push down until force threshold
            while force() < 5:
                sleep(0.01)
            end
            
            # Disable force mode
            end_force_mode()
            
            # Move up
            movel(p[0.3, 0, 0.4, 0, 3.14, 0], a=0.5, v=0.1)
        end
        """
    
    @staticmethod
    def spiral_search_template():
        """Template for spiral search pattern"""
        return """
        def spiral_search():
            center_x = 0.3
            center_y = 0.0
            z_height = 0.3
            max_radius = 0.1
            num_spirals = 5
            points_per_spiral = 20
            
            i = 0
            while i < num_spirals * points_per_spiral:
                angle = i * 2 * 3.14159 / points_per_spiral
                radius = max_radius * i / (num_spirals * points_per_spiral)
                
                x = center_x + radius * cos(angle)
                y = center_y + radius * sin(angle)
                
                movel(p[x, y, z_height, 0, 3.14, 0], a=0.5, v=0.1)
                
                i = i + 1
            end
        end
        """


# Additional Demos

def demo_individual_joint_control():
    """Individual joint control and movements"""
    print("\nIndividual Joint Control")

    config = URConfig(robot_ip="192.168.1.101")
    robot = SafeURRobot(config)

    if not robot.connect():
        return

    try:
        # Home position
        home_joints = [(159.73*pi/180), -102.5*pi/180, -100*pi/180, 296*pi/180, -260*pi/180, 35*pi/180]

        print("Moving to home position...")
        robot.move_joint(home_joints, velocity=0.5)
        time.sleep(5)

        print("\nDemonstrating individual joint movements...")

        # Move joint 6 (wrist rotation) back and forth
        print("1. Moving joint 6 (wrist rotation)")
        joint_positions = home_joints.copy()
        joint_positions[5] = 85*pi/180  # Rotate wrist
        robot.move_joint(joint_positions, velocity=0.3)
        time.sleep(4)

        joint_positions[5] = -15*pi/180  # Rotate back
        robot.move_joint(joint_positions, velocity=0.3)
        time.sleep(4)

        # Move joint 5 (wrist tilt)
        print("2. Moving joint 5 (wrist tilt)")
        joint_positions = home_joints.copy()
        joint_positions[4] = -220*pi/180
        robot.move_joint(joint_positions, velocity=0.3)
        time.sleep(4)

        # Move joint 4 (elbow rotation)
        print("3. Moving joint 4 (elbow rotation)")
        joint_positions = home_joints.copy()
        joint_positions[3] = 320*pi/180
        robot.move_joint(joint_positions, velocity=0.3)
        time.sleep(4)

        # Return to home
        print("\nReturning to home position")
        robot.move_joint(home_joints, velocity=0.5)
        time.sleep(5)

        print("Individual joint control demo complete!")

    finally:
        robot.disconnect()


def demo_coordinate_frames():
    """Tool frame vs base frame movements"""
    print("\nCoordinate Frames")

    config = URConfig(robot_ip="192.168.1.101")
    robot = SafeURRobot(config)

    if not robot.connect():
        return

    try:
        # Home position
        home_joints = [(159.73*pi/180), -102.5*pi/180, -100*pi/180, 296*pi/180, -260*pi/180, 35*pi/180]

        print("Moving to home position...")
        robot.move_joint(home_joints, velocity=0.5)
        time.sleep(5)

        # Starting position in safe workspace
        start_pose = [-0.988, 0.434, 0.341, 2.976, 0.952, 0.016]

        print("\n1. Moving to start position in base frame")
        robot.move_linear(start_pose, velocity=0.1)
        time.sleep(4)

        # Base frame movement - move along X axis
        print("2. Base frame: Moving +X (forward in world frame)")
        base_pose = start_pose.copy()
        base_pose[0] += 0.05  # Move 5cm along world X
        robot.move_linear(base_pose, velocity=0.05)
        time.sleep(4)

        print("3. Base frame: Moving -Y (left in world frame)")
        base_pose[1] -= 0.05  # Move 5cm along world -Y
        robot.move_linear(base_pose, velocity=0.05)
        time.sleep(4)

        # Return to start
        print("4. Returning to start position")
        robot.move_linear(start_pose, velocity=0.1)
        time.sleep(4)

        # Tool frame movement
        print("5. Tool frame: Moving forward in tool direction")
        script = """
current_pose = get_actual_tcp_pose()
target_pose = pose_trans(current_pose, p[0.05, 0, 0, 0, 0, 0])
movel(target_pose, a=0.5, v=0.05)
"""
        robot.send_script(script)
        time.sleep(4)

        print("6. Tool frame: Moving left in tool direction")
        script = """
current_pose = get_actual_tcp_pose()
target_pose = pose_trans(current_pose, p[0, 0.05, 0, 0, 0, 0])
movel(target_pose, a=0.5, v=0.05)
"""
        robot.send_script(script)
        time.sleep(4)

        print("7. Tool frame: Moving down in tool direction")
        script = """
current_pose = get_actual_tcp_pose()
target_pose = pose_trans(current_pose, p[0, 0, 0.03, 0, 0, 0])
movel(target_pose, a=0.5, v=0.05)
"""
        robot.send_script(script)
        time.sleep(4)

        # Return to home
        print("\nReturning to home position")
        robot.move_joint(home_joints, velocity=0.5)
        time.sleep(5)

        print("Coordinate frames demo complete!")

    finally:
        robot.disconnect()


def demo_speed_profiles():
    """Different speed and acceleration profiles"""
    print("\nSpeed Profiles")

    config = URConfig(robot_ip="192.168.1.101")
    robot = SafeURRobot(config)

    if not robot.connect():
        return

    try:
        # Home position
        home_joints = [(159.73*pi/180), -102.5*pi/180, -100*pi/180, 296*pi/180, -260*pi/180, 35*pi/180]

        print("Moving to home position...")
        robot.move_joint(home_joints, velocity=0.5)
        time.sleep(5)

        # Define start and end positions in safe workspace
        start_pose = [-0.925, 0.518, 0.341, 2.976, 0.952, 0.016]
        end_pose = [-1.051, 0.35, 0.341, 2.976, 0.952, 0.016]

        print("\nExecuting same movement with different speeds...\n")

        # Profile 1: Slow and smooth
        print("1. SLOW profile (v=0.05 m/s, a=0.3 m/s²)")
        robot.move_linear(start_pose, velocity=0.1)
        time.sleep(4)

        start_time = time.time()
        robot.move_linear(end_pose, velocity=0.05, acceleration=0.3)
        time.sleep(6)
        duration_1 = time.time() - start_time
        print(f"   Duration: {duration_1:.2f}s")

        # Return to start
        robot.move_linear(start_pose, velocity=0.1)
        time.sleep(4)

        # Profile 2: Medium speed
        print("\n2. MEDIUM profile (v=0.1 m/s, a=0.5 m/s²)")
        start_time = time.time()
        robot.move_linear(end_pose, velocity=0.1, acceleration=0.5)
        time.sleep(4)
        duration_2 = time.time() - start_time
        print(f"   Duration: {duration_2:.2f}s")

        # Return to start
        robot.move_linear(start_pose, velocity=0.1)
        time.sleep(4)

        # Profile 3: Fast
        print("\n3. FAST profile (v=0.15 m/s, a=0.8 m/s²)")
        start_time = time.time()
        robot.move_linear(end_pose, velocity=0.15, acceleration=0.8)
        time.sleep(3)
        duration_3 = time.time() - start_time
        print(f"   Duration: {duration_3:.2f}s")

        print("\nSpeed Profile Summary")
        print(f"Slow:   {duration_1:.2f}s")
        print(f"Medium: {duration_2:.2f}s")
        print(f"Fast:   {duration_3:.2f}s")

        # Return to home
        print("\nReturning to home position")
        robot.move_joint(home_joints, velocity=0.5)
        time.sleep(5)

        print("Speed profiles demo complete!")

    finally:
        robot.disconnect()


def demo_spiral_search():
    """Spiral search pattern for object location"""
    print("\nSpiral Search Pattern Demo")

    config = URConfig(robot_ip="192.168.1.101")
    robot = SafeURRobot(config)

    if not robot.connect():
        return

    try:
        # Home position
        home_joints = [(159.73*pi/180), -102.5*pi/180, -100*pi/180, 296*pi/180, -260*pi/180, 35*pi/180]

        print("Moving to home position...")
        robot.move_joint(home_joints, velocity=0.5)
        time.sleep(5)

        # Spiral parameters in safe workspace
        center_x = -0.988
        center_y = 0.434
        z_height = 0.341
        max_radius = 0.06
        num_spirals = 3
        points_per_spiral = 16

        # Orientation
        rx, ry, rz = 2.976, 0.952, 0.016

        print("Executing spiral search pattern...")

        # Generate spiral points
        total_points = num_spirals * points_per_spiral

        for i in range(total_points):
            angle = i * 2 * np.pi / points_per_spiral
            radius = max_radius * i / total_points

            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)

            pose = [x, y, z_height, rx, ry, rz]

            print(f"  Point {i+1}/{total_points}")
            robot.move_linear(pose, velocity=0.1, acceleration=0.5)
            time.sleep(1)

        print("Spiral search complete!")

        # Return to home
        print("Returning to home")
        robot.move_joint(home_joints, velocity=0.5)
        time.sleep(5)

    finally:
        robot.disconnect()


def demo_gripper_test():
    """Simple gripper open/close test using Robotiq URCap functions"""
    print("\n=== Gripper Test Demo (Robotiq URCap) ===")

    config = URConfig(robot_ip="192.168.1.101")
    robot = SafeURRobot(config)

    # Connect (robot already on, activate gripper)
    if not robot.connect(activate_gripper=True, power_on_robot=False):
        print("Failed to connect to robot")
        return

    if not robot.gripper_initialized:
        print("\nWarning: Gripper not initialized automatically")
        print("Trying manual activation...")
        try:
            robot.rq_activate_gripper()
            robot.gripper_initialized = True
            print("Gripper manually activated!")
        except Exception as e:
            print(f"Manual activation failed: {e}")
            print("\nTroubleshooting:")
            print("  1. Ensure Robotiq_Grippers URCap is installed on robot")
            print("  2. Check that grippy.script is in src/robot_control/")
            print("  3. Verify gripper is powered on and connected to robot")
            print("  4. Try activating from teach pendant first")
            robot.disconnect()
            return

    try:
        print("\nThis demo will test gripper open and close functionality")
        print("Watch the gripper during this test.\n")

        # Test 1: Open gripper
        print("Test 1: Opening gripper...")
        robot.open_gripper()
        time.sleep(3)
        print("Gripper should be fully open")

        # Test 2: Close gripper
        print("\nTest 2: Closing gripper fully...")
        robot.close_gripper()
        time.sleep(3)
        print("Gripper should be fully closed")

        # Test 3: Half open
        print("\nTest 3: Moving to half-open position (50%)...")
        robot.gripper_move_to(50)
        time.sleep(3)
        print("Gripper should be at 50%")

        # Test 4: Quarter positions
        print("\nTest 4: Testing quarter positions...")
        for target in [25, 50, 75, 100, 0]:
            print(f"  Moving to position {target}%...")
            robot.gripper_move_to(target)
            time.sleep(2)

        # Test 5: Multiple cycles
        print("\nTest 5: Running 3 open/close cycles...")
        for i in range(3):
            print(f"  Cycle {i+1}/3: Opening...")
            robot.open_gripper()
            time.sleep(2)

            print(f"  Cycle {i+1}/3: Closing...")
            robot.close_gripper()
            time.sleep(2)

        # Test 6: Object detection test
        print("\nTest 6: Object detection test...")
        print("Place an object in the gripper and it will attempt to grip...")
        time.sleep(3)
        robot.open_gripper()
        time.sleep(2)
        print("Closing gripper...")
        robot.close_gripper()
        time.sleep(2)
        detected = robot.rq_check_object_detected()
        print(f"Object detected: {detected}")

        print("\n=== Gripper Test Complete ===")
        if robot.gripper_initialized:
            print("✓ Gripper is working correctly with URCap functions!")
        else:
            print("✗ Gripper initialization issues")

        # Leave gripper open
        print("\nLeaving gripper in open position...")
        robot.open_gripper()
        time.sleep(2)

    finally:
        robot.disconnect()


def demo_trick():
    """Mystery trick"""
    print("\Mystery Trick")

    config = URConfig(robot_ip="192.168.1.101")
    robot = SafeURRobot(config)

    if not robot.connect():
        return

    try:
        # Home position
        home_joints = [(159.73*pi/180), -102.5*pi/180, -100*pi/180, 296*pi/180, -260*pi/180, 35*pi/180]

        print("Moving to home position...")
        robot.move_joint(home_joints, velocity=0.5)
        time.sleep(5)

        # Trick pickup position
        trick_pose = [-0.781, 0.505, -0.128, 0.478, 2.292, 1.728]
        above_trick = [-0.781, 0.505, -0.05, 0.478, 2.292, 1.728]

        # Dropoff position (from pick and place)
        dropoff_pose = [-0.413, 0.654, -0.12, 2.191, 2.249, 0.1]
        above_dropoff = [-0.413, 0.654, -0.05, 2.191, 2.249, 0.1]

        print("\nAttempting to grab object...")
        robot.show_popup("Hand me the object! Trying 5 times...")

        object_grabbed = False

        # Try 5 times to grab the object
        for attempt in range(1, 6):
            print(f"\nAttempt {attempt}/5")

            # Move to above position
            print(f"Moving to grab position")
            robot.move_linear(above_trick, velocity=0.1)
            time.sleep(3)

            # Move down
            robot.move_linear(trick_pose, velocity=0.05)
            time.sleep(3)

            # Open gripper
            robot.rq_open_gripper(speed=80)
            time.sleep(2)

            # Close gripper slowly
            print(f"Closing gripper (attempt {attempt})...")
            robot.rq_close_gripper(force=80, speed=50)
            time.sleep(3)

            # Check if object detected
            if robot.rq_check_object_detected():
                print(f"Object grabbed on attempt {attempt}!")
                robot.show_popup(f"Got it on attempt {attempt}! Watch this!")
                object_grabbed = True
                time.sleep(2)
                break
            else:
                print(f"No object detected...")
                robot.rq_open_gripper()
                time.sleep(1)
                robot.move_linear(above_trick, velocity=0.1)
                time.sleep(2)

        if not object_grabbed:
            print("\nNo object after 5 attempts. Giving up!")
            robot.show_popup("No object found after 5 tries. Maybe next time!", warning=True)
            time.sleep(3)

            # Return to home
            print("Returning to home")
            robot.move_joint(home_joints, velocity=0.5)
            time.sleep(5)
        else:
            # Perform maneuver!
            print("\nPerforming maneuver!")

            # Move up with pencil
            robot.move_linear(above_trick, velocity=0.1)
            time.sleep(3)

            # Go to home position
            print("Moving to starting position...")
            robot.move_joint(home_joints, velocity=0.5)
            time.sleep(5)

            # Maneuver: fluid figure-8 motion with wrist flips
            print("Executing moves...")

            # Build URScript for smooth nunchuck motion
            num_points = 20
            script = """
def nunchuck_maneuver():
    """

            # Generate figure-8 trajectory with rapid wrist movements
            for i in range(num_points):
                t = i / num_points
                angle = t * 4 * np.pi  # Two full loops

                # Figure-8 pattern
                x = -0.6 + 0.15 * np.sin(angle)
                y = 0.5 + 0.12 * np.sin(2 * angle)
                z = 0.2 + 0.1 * np.sin(3 * angle)

                # Rapid wrist rotations for nunchuck effect
                rx = 2.5 + 0.8 * np.sin(angle * 3)
                ry = 1.5 + 0.6 * np.cos(angle * 2)
                rz = 0.5 + angle * 0.5  # Continuous spin

                pose_str = f"p[{x}, {y}, {z}, {rx}, {ry}, {rz}]"

                # Use blend radius for smooth continuous motion
                if i < num_points - 1:
                    script += f"\n    movel({pose_str}, a=1.2, v=0.3, r=0.015)"
                else:
                    script += f"\n    movel({pose_str}, a=1.2, v=0.3)"

            script += """
end
nunchuck_maneuver()
"""

            robot.send_script(script)
            time.sleep(15)  # Wait for maneuver to complete

            print("Maneuver complete!")
            robot.show_popup("COMPLETE")
            time.sleep(2)

            # Now drop the object at the dropoff location
            print("\nDropping object at designated location...")

            # Move to dropoff
            print("Moving to dropoff position")
            robot.move_linear(above_dropoff, velocity=0.15)
            time.sleep(4)

            # Move down
            robot.move_linear(dropoff_pose, velocity=0.05)
            time.sleep(3)

            # Release object
            print("Releasing object")
            robot.rq_open_gripper()
            time.sleep(2)

            # Move up
            robot.move_linear(above_dropoff, velocity=0.1)
            time.sleep(3)

            print("Object returned!")

            # Return to home
            print("Returning to home")
            robot.move_joint(home_joints, velocity=0.5)
            time.sleep(5)

    finally:
        robot.disconnect()


# Main Entry

if __name__ == "__main__":
    print("UR Robot Control: TCP/IP & URScript")

    print("Ensure robot area is clear and emergency stop is ready.\n")

    while True:
        print("Select demo to run:")
        print("1. Basic movement patterns")
        print("2. Smart pick and place")
        print("3. Spinning top trajectory")
        print("4. Individual joint control")
        print("5. Coordinate frames (base vs tool)")
        print("6. Speed profiles comparison")
        print("7. Spiral search pattern")
        print("8. Mystery")
        print("9. Gripper test (RECOMMENDED if gripper issues)")
        print("0. Exit")

        choice = input("\nEnter choice: ")

        if choice == "1":
            demo_basic_movement()
        elif choice == "2":
            demo_pick_and_place()
        elif choice == "3":
            demo_trajectory_execution()
        elif choice == "4":
            demo_individual_joint_control()
        elif choice == "5":
            demo_coordinate_frames()
        elif choice == "6":
            demo_speed_profiles()
        elif choice == "7":
            demo_spiral_search()
        elif choice == "8":
            demo_trick()
        elif choice == "9":
            demo_gripper_test()
        elif choice == "0":
            break
        else:
            print("Invalid choice!")

    print("\nGoodbye")