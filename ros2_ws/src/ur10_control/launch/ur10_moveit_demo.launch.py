#!/usr/bin/env python3
"""
MoveIt2 Launch File for UR10e Robot Control

This launch file sets up the complete MoveIt2 environment for the UR10e robot,
including motion planning, execution, and visualization
- Motion planning with MoveIt2
- Trajectory execution
- Robot state visualization in RViz
- Integration with the UR robot driver

Advanced topics to explore:
- trajopt for trajectory optimization
- OMPL for custom planning algorithms
- Custom motion planning pipelines
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition, UnlessCondition
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Generate the launch description for UR10e MoveIt2 demo."""

    # Declare launch arguments
    declared_arguments = []

    declared_arguments.append(
        DeclareLaunchArgument(
            "ur_type",
            default_value="ur10e",
            description="Type/series of used UR robot.",
            choices=["ur3", "ur3e", "ur5", "ur5e", "ur10", "ur10e", "ur16e", "ur20", "ur30"],
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "robot_ip",
            default_value="192.168.1.100",
            description="IP address of the UR robot (change to your robot's IP).",
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "use_fake_hardware",
            default_value="true",
            description="Start robot with fake hardware mirroring command to its states.",
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "fake_sensor_commands",
            default_value="false",
            description="Enable fake command interfaces for sensors used for simple simulations.",
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "launch_rviz",
            default_value="true",
            description="Launch RViz for visualization.",
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Use simulation time.",
        )
    )

    # Initialize Arguments
    ur_type = LaunchConfiguration("ur_type")
    robot_ip = LaunchConfiguration("robot_ip")
    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
    fake_sensor_commands = LaunchConfiguration("fake_sensor_commands")
    launch_rviz = LaunchConfiguration("launch_rviz")
    use_sim_time = LaunchConfiguration("use_sim_time")

    # Get package directories
    ur10_control_path = get_package_share_directory("ur10_control")

    # Robot description and SRDF
    robot_description_content = PathJoinSubstitution([
        FindPackageShare("ur_description"), "urdf", "ur.urdf.xacro"
    ])

    robot_description_semantic_content = PathJoinSubstitution([
        FindPackageShare("ur_moveit_config"), "srdf", "ur.srdf.xacro"
    ])

    # Kinematics configuration
    kinematics_yaml = PathJoinSubstitution([
        FindPackageShare("ur_moveit_config"), "config", "kinematics.yaml"
    ])

    # Planning pipeline configuration
    planning_pipeline_config = {
        "move_group": {
            "planning_plugin": "ompl_interface/OMPLPlanner",
            "request_adapters": """default_planner_request_adapters/AddTimeOptimalParameterization default_planner_request_adapters/FixWorkspaceBounds default_planner_request_adapters/FixStartStateBounds default_planner_request_adapters/FixStartStateCollision default_planner_request_adapters/FixStartStatePathConstraints""",
            "start_state_max_bounds_error": 0.1,
        }
    }

    # Trajectory execution configuration
    trajectory_execution = {
        "moveit_manage_controllers": True,
        "trajectory_execution.allowed_execution_duration_scaling": 1.2,
        "trajectory_execution.allowed_goal_duration_margin": 0.5,
        "trajectory_execution.allowed_start_tolerance": 0.01,
    }

    # Planning scene monitor configuration
    planning_scene_monitor_parameters = {
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
    }

    # MoveGroup node configuration
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            {
                "robot_description": robot_description_content,
                "robot_description_semantic": robot_description_semantic_content,
                "robot_description_kinematics": kinematics_yaml,
                "use_sim_time": use_sim_time,
            },
            planning_pipeline_config,
            trajectory_execution,
            planning_scene_monitor_parameters,
        ],
    )

    # RViz node for visualization
    rviz_config_file = PathJoinSubstitution([
        ur10_control_path, "config", "ur10_moveit.rviz"
    ])

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2_moveit",
        output="log",
        arguments=["-d", rviz_config_file],
        parameters=[
            {
                "robot_description": robot_description_content,
                "robot_description_semantic": robot_description_semantic_content,
                "use_sim_time": use_sim_time,
            }
        ],
        condition=IfCondition(launch_rviz),
    )

    # Robot state publisher
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[
            {
                "robot_description": robot_description_content,
                "use_sim_time": use_sim_time,
            }
        ],
    )

    # Joint state publisher (for fake hardware)
    joint_state_publisher_node = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        condition=IfCondition(use_fake_hardware),
        parameters=[
            {
                "use_sim_time": use_sim_time,
            }
        ],
    )

    # Static transform publisher for the world frame
    static_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "world", "base_link"],
        parameters=[{"use_sim_time": use_sim_time}],
    )

    nodes_to_launch = [
        move_group_node,
        rviz_node,
        robot_state_publisher_node,
        joint_state_publisher_node,
        static_tf_node,
    ]

    return LaunchDescription(declared_arguments + nodes_to_launch)