#!/usr/bin/env python3
"""
Trajectory Execution Demo for UR10e Robot

This script demonstrates trajectory execution concepts including:
- Direct trajectory execution
- Real-time trajectory monitoring
- Trajectory validation
- Execution error handling
- Integration with real robot vs simulation

Explore trajopt for trajectory optimization beyond basic MoveIt2 planning.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import time
import math
import numpy as np


class TrajectoryExecutionDemo(Node):
    """
    Demonstration of trajectory execution capabilities.

    This shows how to:
    - Create custom trajectories
    - Execute trajectories directly
    - Monitor trajectory execution
    - Handle execution errors
    - Integrate with real robot hardware
    """

    def __init__(self):
        super().__init__('trajectory_execution_demo')

        self.get_logger().info("Initializing Trajectory Execution Demo...")

        # Joint names for UR10e
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]

        # Action client for trajectory execution
        self.trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory'
        )

        # Subscriber for joint states
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Current joint state
        self.current_joint_state = None
        self.joint_state_received = False

        # Wait for action server
        self.get_logger().info("Waiting for trajectory action server...")
        self.trajectory_client.wait_for_server()

        self.get_logger().info("Trajectory Execution Demo initialized!")

    def joint_state_callback(self, msg):
        """
        Callback for joint state updates.

        Args:
            msg (sensor_msgs.msg.JointState): Current joint state
        """
        # Reorder joint states to match our joint names order
        if len(msg.name) >= len(self.joint_names):
            ordered_positions = []
            for joint_name in self.joint_names:
                if joint_name in msg.name:
                    idx = msg.name.index(joint_name)
                    ordered_positions.append(msg.position[idx])
                else:
                    ordered_positions.append(0.0)

            self.current_joint_state = ordered_positions
            self.joint_state_received = True

    def wait_for_joint_states(self, timeout=5.0):
        """
        Wait for joint states to be received.

        Args:
            timeout (float): Maximum time to wait in seconds

        Returns:
            bool: True if joint states received, False if timeout
        """
        start_time = time.time()
        while not self.joint_state_received and (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)

        return self.joint_state_received

    def create_simple_trajectory(self, start_joints, end_joints, duration=5.0, num_points=20):
        """
        Create a simple interpolated trajectory between two joint configurations.

        Args:
            start_joints (list): Starting joint positions
            end_joints (list): Ending joint positions
            duration (float): Trajectory duration in seconds
            num_points (int): Number of waypoints in trajectory

        Returns:
            trajectory_msgs.msg.JointTrajectory: Generated trajectory
        """
        trajectory = JointTrajectory()
        trajectory.header = Header()
        trajectory.header.stamp = self.get_clock().now().to_msg()
        trajectory.joint_names = self.joint_names

        # Generate trajectory points using linear interpolation
        for i in range(num_points):
            point = JointTrajectoryPoint()

            # Linear interpolation between start and end
            alpha = i / (num_points - 1)
            point.positions = [
                start_joints[j] + alpha * (end_joints[j] - start_joints[j])
                for j in range(len(self.joint_names))
            ]

            # Simple velocity profile (zero at start and end, max in middle)
            if i == 0 or i == num_points - 1:
                point.velocities = [0.0] * len(self.joint_names)
            else:
                # Approximate velocities for smooth motion
                point.velocities = [
                    (end_joints[j] - start_joints[j]) / duration
                    for j in range(len(self.joint_names))
                ]

            # Zero accelerations for simplicity
            point.accelerations = [0.0] * len(self.joint_names)

            # Time from start
            point.time_from_start.sec = int(i * duration / (num_points - 1))
            point.time_from_start.nanosec = int(
                ((i * duration / (num_points - 1)) % 1) * 1e9
            )

            trajectory.points.append(point)

        return trajectory

    def create_sinusoidal_trajectory(self, center_joints, amplitude=0.3, frequency=0.5, duration=10.0):
        """
        Create a sinusoidal trajectory around center joint positions.

        Args:
            center_joints (list): Center joint positions
            amplitude (float): Maximum deviation from center (radians)
            frequency (float): Frequency of oscillation (Hz)
            duration (float): Total trajectory duration

        Returns:
            trajectory_msgs.msg.JointTrajectory: Generated sinusoidal trajectory

        This demonstrates smooth, periodic motion useful for testing and calibration.
        """
        trajectory = JointTrajectory()
        trajectory.header = Header()
        trajectory.header.stamp = self.get_clock().now().to_msg()
        trajectory.joint_names = self.joint_names

        num_points = int(duration * 20)  # 20 Hz trajectory
        dt = duration / num_points

        for i in range(num_points):
            point = JointTrajectoryPoint()
            t = i * dt

            # Sinusoidal motion for each joint with different phases
            point.positions = []
            point.velocities = []
            point.accelerations = []

            for j in range(len(self.joint_names)):
                # Phase offset for each joint to create complex motion
                phase = j * math.pi / 3

                # Position: center + amplitude * sin(2πft + phase)
                position = center_joints[j] + amplitude * math.sin(
                    2 * math.pi * frequency * t + phase
                )
                point.positions.append(position)

                # Velocity: amplitude * 2πf * cos(2πft + phase)
                velocity = amplitude * 2 * math.pi * frequency * math.cos(
                    2 * math.pi * frequency * t + phase
                )
                point.velocities.append(velocity)

                # Acceleration: -amplitude * (2πf)² * sin(2πft + phase)
                acceleration = -amplitude * (2 * math.pi * frequency)**2 * math.sin(
                    2 * math.pi * frequency * t + phase
                )
                point.accelerations.append(acceleration)

            # Set time
            point.time_from_start.sec = int(t)
            point.time_from_start.nanosec = int((t % 1) * 1e9)

            trajectory.points.append(point)

        return trajectory

    def execute_trajectory(self, trajectory, blocking=True):
        """
        Execute a trajectory using the action interface.

        Args:
            trajectory (trajectory_msgs.msg.JointTrajectory): Trajectory to execute
            blocking (bool): Whether to wait for completion

        Returns:
            bool: True if execution successful
        """
        self.get_logger().info(f"Executing trajectory with {len(trajectory.points)} points...")

        # Create action goal
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory

        # Send goal
        future = self.trajectory_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Trajectory goal rejected!")
            return False

        self.get_logger().info("Trajectory goal accepted, executing...")

        if blocking:
            # Wait for completion
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)

            result = result_future.result()
            if result.result.error_code == 0:
                self.get_logger().info("Trajectory execution completed successfully!")
                return True
            else:
                self.get_logger().error(f"Trajectory execution failed with error: {result.result.error_code}")
                return False
        else:
            self.get_logger().info("Trajectory execution started (non-blocking)")
            return True

    def demonstrate_trajectory_types(self):
        """
        Demonstrate different types of trajectories

        This shows various trajectory generation techniques
        """
        self.get_logger().info("Starting trajectory demonstration...")

        # Wait for current joint states
        if not self.wait_for_joint_states():
            self.get_logger().error("Failed to receive joint states!")
            return False

        current_joints = self.current_joint_state.copy()
        self.get_logger().info(f"Current joint positions: {current_joints}")

        try:
            # 1. Simple point-to-point trajectory
            self.get_logger().info("Demo 1: Simple point-to-point trajectory")
            target_joints = [0.5, -1.2, 1.0, -1.5, 0.0, 0.0]

            simple_traj = self.create_simple_trajectory(
                current_joints, target_joints, duration=5.0
            )

            if not self.execute_trajectory(simple_traj):
                return False

            time.sleep(2.0)

            # 2. Sinusoidal trajectory around current position
            self.get_logger().info("Demo 2: Sinusoidal trajectory")

            # Use a safer center position
            center_joints = [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]
            sin_traj = self.create_sinusoidal_trajectory(
                center_joints, amplitude=0.2, frequency=0.2, duration=8.0
            )

            if not self.execute_trajectory(sin_traj):
                return False

            time.sleep(2.0)

            # 3. Return to home position
            self.get_logger().info("Demo 3: Return to home position")
            home_joints = [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]

            # Get current position for smooth return
            if self.wait_for_joint_states(timeout=2.0):
                current_joints = self.current_joint_state.copy()

            return_traj = self.create_simple_trajectory(
                current_joints, home_joints, duration=4.0
            )

            if not self.execute_trajectory(return_traj):
                return False

            self.get_logger().info("All trajectory demonstrations completed successfully!")
            return True

        except Exception as e:
            self.get_logger().error(f"Error in trajectory demonstration: {str(e)}")
            return False

    def run_demo(self):
        """Run the complete trajectory execution demonstration."""
        success = self.demonstrate_trajectory_types()
        if success:
            self.get_logger().info("Trajectory execution demo completed successfully!")
        else:
            self.get_logger().error("Trajectory execution demo failed!")


def main(args=None):
    """
    Main function for the trajectory execution demo.

    Usage:
        ros2 run ur10_control trajectory_execution_demo.py

    This demo teaches:
    - Custom trajectory generation
    - Direct trajectory execution
    - Real-time motion control
    - Error handling in trajectory execution

    Advanced topics to explore:
    - trajopt for trajectory optimization
    - Real-time trajectory modification
    - Force-controlled trajectory execution
    """
    rclpy.init(args=args)

    try:
        demo_node = TrajectoryExecutionDemo()
        time.sleep(2.0)  # Allow initialization

        demo_node.run_demo()

        # Keep running for interaction
        demo_node.get_logger().info("Demo completed. Node will continue running...")
        rclpy.spin(demo_node)

    except KeyboardInterrupt:
        print("\nTrajectory execution demo interrupted by user")
    except Exception as e:
        print(f"Error in trajectory demo: {str(e)}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()