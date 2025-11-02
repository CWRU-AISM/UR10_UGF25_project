#!/usr/bin/env python3
"""
Basic Motion Planning Example for UR10e Robot using MoveIt2

This script demonstrates fundamental motion planning concepts using MoveIt2
- Setting up MoveIt2 planning interface
- Planning to joint space goals
- Planning to pose goals
- Executing planned trajectories
- Basic collision avoidance

Advanced topics to explore:
- trajopt for trajectory optimization
- OMPL planners (RRT, PRM, etc.)
- Custom planning pipelines
- Cartesian path planning
"""

import rclpy
from rclpy.node import Node
from moveit_py import MoveItPy
from moveit_py.planning_scene_monitor import PlanningSceneMonitor
from geometry_msgs.msg import PoseStamped, Pose
from std_msgs.msg import Header
import time
import numpy as np


class UR10MotionPlanningExample(Node):
    """
    Example node demonstrating basic motion planning with the UR10e robot.

    This class provides examples of:
    - Joint space planning
    - Cartesian space planning
    - Trajectory execution
    - Planning scene interaction
    """

    def __init__(self):
        super().__init__('ur10_motion_planning_example')

        self.get_logger().info("Initializing UR10e Motion Planning Example...")

        # Initialize MoveIt2
        self.moveit = MoveItPy(node_name="ur10_motion_planning")
        self.ur_manipulator = self.moveit.get_planning_component("ur_manipulator")
        self.planning_scene = self.moveit.get_planning_scene_monitor()

        # Wait for the planning scene to be ready
        self.get_logger().info("Waiting for planning scene...")
        time.sleep(2.0)

        self.get_logger().info("Motion planning setup complete!")

    def plan_to_joint_goal(self, joint_positions):
        """
        Plan to a specific joint configuration.

        Args:
            joint_positions (list): Target joint positions in radians

        Returns:
            bool: True if planning and execution successful
        """
        self.get_logger().info(f"Planning to joint goal: {joint_positions}")

        try:
            # Set the joint goal
            self.ur_manipulator.set_goal_state(
                configuration_name="",
                joint_state=joint_positions
            )

            # Plan the motion
            plan_result = self.ur_manipulator.plan()

            if plan_result:
                self.get_logger().info("Joint space planning successful!")

                # Execute the planned trajectory
                robot_traj = plan_result.trajectory
                success = self.ur_manipulator.execute(robot_traj, blocking=True)

                if success:
                    self.get_logger().info("Joint trajectory execution successful!")
                    return True
                else:
                    self.get_logger().error("Joint trajectory execution failed!")
                    return False
            else:
                self.get_logger().error("Joint space planning failed!")
                return False

        except Exception as e:
            self.get_logger().error(f"Error in joint planning: {str(e)}")
            return False

    def plan_to_pose_goal(self, target_pose):
        """
        Plan to a specific end-effector pose.

        Args:
            target_pose (geometry_msgs.msg.Pose): Target pose for end-effector

        Returns:
            bool: True if planning and execution successful
        """
        self.get_logger().info("Planning to pose goal...")

        try:
            # Create pose stamped message
            pose_stamped = PoseStamped()
            pose_stamped.header = Header()
            pose_stamped.header.frame_id = "base_link"
            pose_stamped.header.stamp = self.get_clock().now().to_msg()
            pose_stamped.pose = target_pose

            # Set the pose goal
            self.ur_manipulator.set_goal_state(pose_stamped_msg=pose_stamped, pose_link="tool0")

            # Plan the motion
            plan_result = self.ur_manipulator.plan()

            if plan_result:
                self.get_logger().info("Pose planning successful!")

                # Execute the planned trajectory
                robot_traj = plan_result.trajectory
                success = self.ur_manipulator.execute(robot_traj, blocking=True)

                if success:
                    self.get_logger().info("Pose trajectory execution successful!")
                    return True
                else:
                    self.get_logger().error("Pose trajectory execution failed!")
                    return False
            else:
                self.get_logger().error("Pose planning failed!")
                return False

        except Exception as e:
            self.get_logger().error(f"Error in pose planning: {str(e)}")
            return False

    def go_to_home_position(self):
        """Move the robot to a safe home position."""
        self.get_logger().info("Moving to home position...")

        # Home position (all joints at 0 except shoulder_lift to avoid singularity)
        home_joints = [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]
        return self.plan_to_joint_goal(home_joints)

    def demonstrate_basic_motions(self):
        """
        Demonstrate basic motion planning capabilities.

        This function shows how to:
        1. Move to home position
        2. Plan to various joint configurations
        3. Plan to different poses
        """
        self.get_logger().info("Starting motion planning demonstration...")

        # 1. Move to home position
        if not self.go_to_home_position():
            self.get_logger().error("Failed to reach home position!")
            return False

        time.sleep(2.0)

        # 2. Move to a joint configuration (shoulder pan rotated)
        self.get_logger().info("Moving to joint configuration 1...")
        joint_config_1 = [1.57, -1.57, 0.0, -1.57, 0.0, 0.0]  # 90 degrees shoulder pan
        if not self.plan_to_joint_goal(joint_config_1):
            self.get_logger().error("Failed to reach joint configuration 1!")
            return False

        time.sleep(2.0)

        # 3. Move to another joint configuration
        self.get_logger().info("Moving to joint configuration 2...")
        joint_config_2 = [0.0, -1.0, 1.57, -2.14, 0.0, 0.0]  # Elbow bent
        if not self.plan_to_joint_goal(joint_config_2):
            self.get_logger().error("Failed to reach joint configuration 2!")
            return False

        time.sleep(2.0)

        # 4. Plan to a specific pose (forward reach)
        self.get_logger().info("Planning to forward reach pose...")
        target_pose = Pose()
        target_pose.position.x = 0.6
        target_pose.position.y = 0.0
        target_pose.position.z = 0.4
        target_pose.orientation.x = 0.0
        target_pose.orientation.y = 0.707
        target_pose.orientation.z = 0.0
        target_pose.orientation.w = 0.707

        if not self.plan_to_pose_goal(target_pose):
            self.get_logger().error("Failed to reach forward pose!")
            return False

        time.sleep(2.0)

        # 5. Return to home
        self.get_logger().info("Returning to home position...")
        if not self.go_to_home_position():
            self.get_logger().error("Failed to return to home position!")
            return False

        self.get_logger().info("Motion planning demonstration completed successfully!")
        return True

    def run_example(self):
        """Run the complete motion planning example."""
        try:
            success = self.demonstrate_basic_motions()
            if success:
                self.get_logger().info("All motion planning examples completed successfully!")
            else:
                self.get_logger().error("Some motion planning examples failed!")
        except Exception as e:
            self.get_logger().error(f"Unexpected error during demonstration: {str(e)}")


def main(args=None):
    """
    Main function to run the motion planning example.

    Usage:
        ros2 run ur10_control motion_planning_example.py
    """
    rclpy.init(args=args)

    try:
        # Create and run the motion planning example
        motion_planning_node = UR10MotionPlanningExample()

        # Wait a moment for everything to initialize
        time.sleep(1.0)

        # Run the demonstration
        motion_planning_node.run_example()

        # Keep the node running for any additional interactions
        motion_planning_node.get_logger().info("Motion planning example completed. Node will continue running...")
        rclpy.spin(motion_planning_node)

    except KeyboardInterrupt:
        print("\nMotion planning example interrupted by user")
    except Exception as e:
        print(f"Error in motion planning example: {str(e)}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()