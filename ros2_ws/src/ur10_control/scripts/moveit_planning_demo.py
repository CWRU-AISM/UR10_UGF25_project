#!/usr/bin/env python3
"""
Advanced MoveIt2 Planning Demo for UR10e Robot

This script demonstrates more advanced MoveIt2 concepts including:
- Different planning algorithms (OMPL planners)
- Cartesian path planning
- Planning with constraints
- Collision object management
- Planning scene manipulation
"""

import rclpy
from rclpy.node import Node
from moveit_py import MoveItPy
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import CollisionObject, PlanningScene
import time
import math


class AdvancedMoveItDemo(Node):
    """
    Advanced MoveIt2 demonstration node.

    This class demonstrates:
    - Different OMPL planning algorithms
    - Cartesian path planning
    - Planning scene management
    - Collision object handling
    - Constrained motion planning
    """

    def __init__(self):
        super().__init__('advanced_moveit_demo')

        self.get_logger().info("Initializing Advanced MoveIt2 Demo...")

        # Initialize MoveIt2
        self.moveit = MoveItPy(node_name="advanced_moveit_demo")
        self.ur_manipulator = self.moveit.get_planning_component("ur_manipulator")
        self.planning_scene_monitor = self.moveit.get_planning_scene_monitor()

        # Publisher for planning scene updates
        self.planning_scene_publisher = self.create_publisher(
            PlanningScene, '/planning_scene', 10
        )

        # Available OMPL planners to explore
        self.ompl_planners = [
            "RRTConnect",  # Default, good balance of speed and optimality
            "RRT",         # Basic rapidly-exploring random tree
            "RRTstar",     # Asymptotically optimal version of RRT
            "PRM",         # Probabilistic roadmap method
            "PRMstar",     # Asymptotically optimal PRM
            "EST",         # Expansive-space trees
            "SBL",         # Single-query bi-directional lazy planner
            "KPIECE",      # Kinodynamic motion planning by interior-exterior cell exploration
            "BiTRRT",      # Bidirectional transition-based RRT
            "LBKPIECE",    # Lazy bi-directional KPIECE
        ]

        self.get_logger().info("Advanced MoveIt2 Demo initialized!")

    def add_collision_objects(self):
        """
        Add collision objects to the planning scene.

        This demonstrates how to add obstacles that the robot must avoid.
        TODO: Modify this to add different objects.
        """
        self.get_logger().info("Adding collision objects to planning scene...")

        # Create a table obstacle
        table = CollisionObject()
        table.header.frame_id = "base_link"
        table.header.stamp = self.get_clock().now().to_msg()
        table.id = "table"

        # Define table as a box
        table_primitive = SolidPrimitive()
        table_primitive.type = SolidPrimitive.BOX
        table_primitive.dimensions = [1.5, 1.0, 0.02]  # length, width, height

        # Position the table
        table_pose = Pose()
        table_pose.position.x = 0.75
        table_pose.position.y = 0.0
        table_pose.position.z = -0.01
        table_pose.orientation.w = 1.0

        table.primitives = [table_primitive]
        table.primitive_poses = [table_pose]
        table.operation = CollisionObject.ADD

        # Create a box obstacle on the table
        box = CollisionObject()
        box.header.frame_id = "base_link"
        box.header.stamp = self.get_clock().now().to_msg()
        box.id = "obstacle_box"

        # Define box dimensions
        box_primitive = SolidPrimitive()
        box_primitive.type = SolidPrimitive.BOX
        box_primitive.dimensions = [0.1, 0.1, 0.2]  # small box

        # Position the box on the table
        box_pose = Pose()
        box_pose.position.x = 0.5
        box_pose.position.y = 0.2
        box_pose.position.z = 0.1
        box_pose.orientation.w = 1.0

        box.primitives = [box_primitive]
        box.primitive_poses = [box_pose]
        box.operation = CollisionObject.ADD

        # Publish the planning scene with collision objects
        planning_scene_msg = PlanningScene()
        planning_scene_msg.world.collision_objects = [table, box]
        planning_scene_msg.is_diff = True

        self.planning_scene_publisher.publish(planning_scene_msg)
        time.sleep(1.0)  # Allow time for the scene to update

        self.get_logger().info("Collision objects added successfully!")

    def plan_with_different_algorithms(self, target_pose):
        """
        Demonstrate planning with different OMPL algorithms.

        Args:
            target_pose (geometry_msgs.msg.Pose): Target pose for comparison
        """
        self.get_logger().info("Comparing different OMPL planning algorithms...")

        results = {}

        for planner in self.ompl_planners[:3]:  # Test first 3 planners
            self.get_logger().info(f"Testing {planner} planner...")

            try:
                # Set the planner
                self.ur_manipulator.set_planning_pipeline_id("ompl")
                self.ur_manipulator.set_planner_id(planner)

                # Create pose stamped message
                pose_stamped = PoseStamped()
                pose_stamped.header.frame_id = "base_link"
                pose_stamped.header.stamp = self.get_clock().now().to_msg()
                pose_stamped.pose = target_pose

                # Set goal
                self.ur_manipulator.set_goal_state(
                    pose_stamped_msg=pose_stamped, pose_link="tool0"
                )

                # Plan
                start_time = time.time()
                plan_result = self.ur_manipulator.plan()
                planning_time = time.time() - start_time

                if plan_result:
                    trajectory = plan_result.trajectory
                    path_length = len(trajectory.joint_trajectory.points)

                    results[planner] = {
                        'success': True,
                        'planning_time': planning_time,
                        'path_length': path_length,
                        'trajectory': trajectory
                    }

                    self.get_logger().info(
                        f"{planner}: Success! Time: {planning_time:.3f}s, "
                        f"Path points: {path_length}"
                    )
                else:
                    results[planner] = {
                        'success': False,
                        'planning_time': planning_time
                    }
                    self.get_logger().warn(f"{planner}: Planning failed!")

            except Exception as e:
                self.get_logger().error(f"Error with {planner}: {str(e)}")
                results[planner] = {'success': False, 'error': str(e)}

            time.sleep(0.5)  # Brief pause between planners

        # Report comparison
        self.get_logger().info("\n=== Planning Algorithm Comparison ===")
        for planner, result in results.items():
            if result['success']:
                self.get_logger().info(
                    f"{planner}: ✓ {result['planning_time']:.3f}s, "
                    f"{result['path_length']} points"
                )
            else:
                self.get_logger().info(f"{planner}: ✗ Failed")

        return results

    def plan_cartesian_path(self, waypoints):
        """
        Plan a Cartesian path through multiple waypoints.

        Args:
            waypoints (list): List of geometry_msgs.msg.Pose waypoints

        This demonstrates straight-line motion planning between poses.
        """
        self.get_logger().info("Planning Cartesian path through waypoints...")

        try:
            # For this example, we'll use the basic path planning
            # In a full implementation, you would use the Cartesian path planner
            success_count = 0

            for i, waypoint in enumerate(waypoints):
                self.get_logger().info(f"Moving to waypoint {i+1}/{len(waypoints)}")

                pose_stamped = PoseStamped()
                pose_stamped.header.frame_id = "base_link"
                pose_stamped.header.stamp = self.get_clock().now().to_msg()
                pose_stamped.pose = waypoint

                self.ur_manipulator.set_goal_state(
                    pose_stamped_msg=pose_stamped, pose_link="tool0"
                )

                plan_result = self.ur_manipulator.plan()

                if plan_result:
                    # Execute the trajectory
                    success = self.ur_manipulator.execute(plan_result.trajectory, blocking=True)
                    if success:
                        success_count += 1
                        self.get_logger().info(f"Waypoint {i+1} reached successfully!")
                    else:
                        self.get_logger().error(f"Failed to execute to waypoint {i+1}")
                else:
                    self.get_logger().error(f"Failed to plan to waypoint {i+1}")

                time.sleep(1.0)

            self.get_logger().info(
                f"Cartesian path planning completed: {success_count}/{len(waypoints)} "
                f"waypoints reached"
            )
            return success_count == len(waypoints)

        except Exception as e:
            self.get_logger().error(f"Error in Cartesian path planning: {str(e)}")
            return False

    def create_circular_trajectory(self, center, radius, num_points=8):
        """
        Create waypoints for a circular trajectory.

        Args:
            center (list): [x, y, z] center of the circle
            radius (float): Radius of the circle
            num_points (int): Number of waypoints in the circle

        Returns:
            list: List of Pose messages forming a circle
        """
        waypoints = []

        for i in range(num_points):
            angle = 2 * math.pi * i / num_points

            pose = Pose()
            pose.position.x = center[0] + radius * math.cos(angle)
            pose.position.y = center[1] + radius * math.sin(angle)
            pose.position.z = center[2]

            # Keep end-effector pointing down
            pose.orientation.x = 0.0
            pose.orientation.y = 0.707
            pose.orientation.z = 0.0
            pose.orientation.w = 0.707

            waypoints.append(pose)

        return waypoints

    def demonstrate_advanced_planning(self):
        """
        Run the complete advanced planning demonstration.

        This shows various MoveIt2 capabilities to explore.
        """
        self.get_logger().info("Starting advanced planning demonstration...")

        try:
            # 1. Add collision objects to the scene
            self.add_collision_objects()
            time.sleep(2.0)

            # 2. Define a target pose for algorithm comparison
            target_pose = Pose()
            target_pose.position.x = 0.4
            target_pose.position.y = 0.3
            target_pose.position.z = 0.5
            target_pose.orientation.x = 0.0
            target_pose.orientation.y = 0.707
            target_pose.orientation.z = 0.0
            target_pose.orientation.w = 0.707

            # 3. Compare different planning algorithms
            results = self.plan_with_different_algorithms(target_pose)
            time.sleep(2.0)

            # 4. Execute the best solution (if any succeeded)
            best_planner = None
            best_time = float('inf')

            for planner, result in results.items():
                if result['success'] and result['planning_time'] < best_time:
                    best_time = result['planning_time']
                    best_planner = planner

            if best_planner:
                self.get_logger().info(f"Executing trajectory from best planner: {best_planner}")
                trajectory = results[best_planner]['trajectory']
                self.ur_manipulator.execute(trajectory, blocking=True)
                time.sleep(2.0)

            # 5. Demonstrate Cartesian path planning with circular trajectory
            self.get_logger().info("Demonstrating circular Cartesian path...")
            circle_center = [0.5, 0.0, 0.4]
            circle_radius = 0.1
            waypoints = self.create_circular_trajectory(circle_center, circle_radius, 6)

            self.plan_cartesian_path(waypoints)

            self.get_logger().info("Advanced planning demonstration completed!")
            return True

        except Exception as e:
            self.get_logger().error(f"Error in advanced planning demo: {str(e)}")
            return False

    def run_demo(self):
        """Run the complete advanced MoveIt2 demonstration."""
        success = self.demonstrate_advanced_planning()
        if success:
            self.get_logger().info("All advanced planning demos completed successfully!")
        else:
            self.get_logger().error("Some advanced planning demos failed!")


def main(args=None):
    """
    Main function for the advanced MoveIt2 demo.

    Usage:
        ros2 run ur10_control moveit_planning_demo.py

    This demo shows advanced MoveIt2 features including:
    - OMPL planner comparison
    - Collision object management
    - Cartesian path planning
    """
    rclpy.init(args=args)

    try:
        demo_node = AdvancedMoveItDemo()
        time.sleep(2.0)  # Allow initialization

        demo_node.run_demo()

        # Keep running for interaction
        demo_node.get_logger().info("Demo completed. Node will continue running...")
        rclpy.spin(demo_node)

    except KeyboardInterrupt:
        print("\nAdvanced MoveIt2 demo interrupted by user")
    except Exception as e:
        print(f"Error in advanced demo: {str(e)}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()