"""
Control Group
UR Robot Control via TCP/IP and URScript
"""
# Import Required Dependencies
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

# Import classes from control program
from ur10_control import URConfig, URRobot, TrajectoryGenerator, SafeURRobot

# Define Programs
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


# List All Programs Available
programs = [demo_basic_movement,demo_individual_joint_control]

# Main Program Entry
if __name__ == "__main__":
    # Program Introduction
    print("UR Robot Control: TCP/IP & URScript")
    print("Ensure robot area is clear and emergency stop is ready.\n")

    # Enter main loop
    # User will be able to select test program from list
    while True:
        print("Select demo to run:")
        
        # List all available programs
        k = 0
        for program in programs:
            k = k+1
            print(str(k) + ". " + program.__name__)
            
        # Let user select program
        choice = input("\nEnter choice: ")

        # Run selected program
        programs[int(choice)]()

    
    print("\nGoodbye")