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
        """Initialize UR robot connection"""
        self.config = config or URConfig()
        self.socket = None
        self.dashboard_socket = None
        self.is_connected = False
        self.current_pose = None
        self.current_joints = None
        
        # Thread for monitoring robot state
        self.monitor_thread = None
        self.monitoring = False
        
    def connect(self):
        """Establish connection to UR robot"""
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
            
            # Power on and release brakes
            self.power_on()
            time.sleep(1)
            self.release_brakes()
            
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
                   velocity: float = None, acceleration: float = None):
        """
        Move to target joint positions
        Args:
            joint_positions: List of 6 joint angles in radians
            velocity: Joint velocity in rad/s
            acceleration: Joint acceleration in rad/s^2
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
        
        return self.send_script(script)
    
    def move_linear(self, pose: List[float], 
                    velocity: float = None, acceleration: float = None):
        """
        Linear movement to target pose
        Args:
            pose: [x, y, z, rx, ry, rz] in meters and radians
            velocity: Tool velocity in m/s
            acceleration: Tool acceleration in m/s^2
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
        
        return self.send_script(script)
    
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
    
    # Gripper
    
    def set_digital_output(self, pin: int, value: bool):
        """Set digital output pin"""
        val = 'True' if value else 'False'
        script = f"set_digital_out({pin}, {val})"
        return self.send_script(script)
    
    def close_gripper(self):
        """Close gripper (example using DO 0)"""
        return self.set_digital_output(0, True)
    
    def open_gripper(self):
        """Open gripper (example using DO 0)"""
        return self.set_digital_output(0, False)
    
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
        # Simplified, need parsing
        while self.monitoring:
            try:

                time.sleep(0.1)
            except:
                break


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
                    velocity: float = None, acceleration: float = None):
        """Override move_linear with safety check"""
        # if not self.check_pose_safety(pose):
        #     print("Safety check failed! Movement cancelled.")
        #     return False
        
        # Reduce speed for safety
        vel = min(velocity or self.config.max_velocity, 0.5)
        acc = min(acceleration or self.config.max_acceleration, 0.5)
        
        return super().move_linear(pose, vel, acc)

def demo_basic_movement():
    """Demo: Basic movement patterns"""
    
    # Initialize robot
    config = URConfig(robot_ip="192.168.1.101")  # Update IP
    robot = SafeURRobot(config)
    
    if not robot.connect():
        print("Failed to connect to robot")
        return
    
    try:
        # Home position (update for your robot)
        home_joints = [(159.73*pi/180), -102.5*pi/180, -100*pi/180, 296*pi/180, -260*pi/180, 35*pi/180]  # radians
        
        print("Moving to home position...")
        robot.move_joint(home_joints, velocity=0.5)
        time.sleep(3)
        
        # Square movement pattern
        print("Executing square pattern...")
        base_pose = [-.65, .43, .346, 2.976, 0.952, 0.016]  # [x, y, z, rx, ry, rz]
        
        square_poses = [
            [-.925, 0.518, .341, 2.976, .952, .016],
            [-1.051, 0.518, .341, 2.976, .952, .016],
            [-1.051, 0.35, .341, 2.976, .952, .016],
            [-.925, 0.35, .341, 2.976, .952, .016],
        ]
        
        for i, pose in enumerate(square_poses):
            print(f"  Moving to corner {i+1}/4")
            robot.move_linear(pose, velocity=0.1)
            time.sleep(2)
        
        # print("Returning to home...")
        robot.move_joint(home_joints, velocity=0.5)
        
    finally:
        robot.disconnect()


def demo_pick_and_place():
    """Demo: Simple pick and place operation"""
    print("\n=== Pick and Place Demo ===")
    
    config = URConfig(robot_ip="192.168.1.101")  
    robot = SafeURRobot(config)
    
    if not robot.connect():
        return
    
    try:
        # Define positions
        pick_pose = [0.3, 0.1, 0.2, 0, 3.14, 0]
        above_pick = [0.3, 0.1, 0.3, 0, 3.14, 0]
        place_pose = [0.3, -0.1, 0.2, 0, 3.14, 0]
        above_place = [0.3, -0.1, 0.3, 0, 3.14, 0]
        
        print("Starting pick and place sequence...")
        
        # Move above pick position
        print("1. Moving above pick position")
        robot.move_linear(above_pick, velocity=0.2)
        time.sleep(2)
        
        # Move down to pick
        print("2. Moving down to pick")
        robot.move_linear(pick_pose, velocity=0.05)
        time.sleep(1)
        
        # Close gripper
        print("3. Closing gripper")
        robot.close_gripper()
        time.sleep(1)
        
        # Move up
        print("4. Lifting object")
        robot.move_linear(above_pick, velocity=0.05)
        time.sleep(1)
        
        # Move to place position
        print("5. Moving to place position")
        robot.move_linear(above_place, velocity=0.2)
        time.sleep(2)
        
        # Move down to place
        print("6. Lowering object")
        robot.move_linear(place_pose, velocity=0.05)
        time.sleep(1)
        
        # Open gripper
        print("7. Opening gripper")
        robot.open_gripper()
        time.sleep(1)
        
        # Move up
        print("8. Moving up")
        robot.move_linear(above_place, velocity=0.1)
        
        print("Pick and place complete!")
        
    finally:
        robot.disconnect()


def demo_trajectory_execution():
    """Demo: Execute smooth trajectory"""
    print("\n=== Trajectory Execution Demo ===")
    
    config = URConfig(robot_ip="192.168.1.100")
    robot = SafeURRobot(config)
    
    if not robot.connect():
        return
    
    try:
        # Generate circular trajectory
        center = [0.3, 0.0, 0.3, 0, 3.14, 0]
        radius = 0.1
        
        print("Generating circular trajectory...")
        trajectory = TrajectoryGenerator.circular_trajectory(center, radius, num_points=20)
        
        print("Executing trajectory...")
        for i, pose in enumerate(trajectory):
            print(f"  Point {i+1}/{len(trajectory)}")
            robot.move_linear(pose, velocity=0.1, acceleration=0.1)
            time.sleep(0.5)
        
        print("Trajectory complete!")
        
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


# Implement

def exercise_1_joint_control():
    """Exercise 1: Direct joint control"""
    print("\n=== Exercise 1: Joint Control ===")
    # TODO: Implement moving each joint individually
    # Record joint limits and create safe movement sequence
    pass


def exercise_2_coordinate_frames():
    """Exercise 2: Understanding coordinate frames"""
    print("\n=== Exercise 2: Coordinate Frames ===")
    # TODO: Move robot in tool frame vs base frame
    # Demonstrate the difference
    pass


def exercise_3_speed_profiles():
    """Exercise 3: Speed and acceleration profiles"""
    print("\n=== Exercise 3: Speed Profiles ===")
    # TODO: Execute same movement with different speed/acceleration
    # Log and compare execution times
    pass


# Main Entry

if __name__ == "__main__":
    print("UR Robot Control - TCP/IP & URScript")
    
    print("WARNING: This code controls a real robot!")
    print("Ensure robot area is clear and emergency stop is ready.\n")
    
    while True:
        print("\nSelect demo to run:")
        print("1. Basic movement patterns")
        print("2. Pick and place sequence")
        print("3. Trajectory execution")
        print("4. Exercise 1: Joint control")
        print("5. Exercise 2: Coordinate frames")
        print("6. Exercise 3: Speed profiles")
        print("0. Exit")
        
        choice = input("\nEnter choice: ")
        
        if choice == "1":
            demo_basic_movement()
        elif choice == "2":
            demo_pick_and_place()
        elif choice == "3":
            demo_trajectory_execution()
        elif choice == "4":
            exercise_1_joint_control()
        elif choice == "5":
            exercise_2_coordinate_frames()
        elif choice == "6":
            exercise_3_speed_profiles()
        elif choice == "0":
            break
        else:
            print("Invalid choice!")
    
    print("\nGoodbye!")