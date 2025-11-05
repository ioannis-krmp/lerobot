# !/usr/bin/env python

import time

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    robot_action_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    ForwardKinematicsJointsToEE,
    InverseKinematicsEEToJoints,
)
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.teleoperators.so100_leader.config_so100_leader import SO100LeaderConfig
from lerobot.teleoperators.so100_leader.so100_leader import SO100Leader
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

FPS = 30

# Initialize the robot and teleoperator config
follower_config = SO100FollowerConfig(
    port="/dev/ttyACM0", id="follower_arm_0", use_degrees=True
)
leader_config = SO100LeaderConfig(port="/dev/ttyACM3", id="leader_arm_0")

# Initialize the robot and teleoperator
follower = SO100Follower(follower_config)
leader = SO100Leader(leader_config)

# Connect first to get motor information
follower.connect()
leader.connect()

print(f"Leader motors: {list(leader.bus.motors.keys())}")
print(f"Follower motors: {list(follower.bus.motors.keys())}")

# Separate arm motors from gripper motor
arm_motors = [m for m in follower.bus.motors.keys() if m != "gripper"]
print(f"Arm motors (for kinematics): {arm_motors}")

# NOTE: It is highly recommended to use the urdf in the SO-ARM100 repo: https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
follower_kinematics_solver = RobotKinematics(
    urdf_path="/home/TODO/lerobot/examples/so100_to_so100_EE/SO101/so101_new_calib.urdf",
    target_frame_name="gripper_frame_link",
    joint_names=arm_motors,  # ONLY ARM MOTORS for kinematics
)

leader_kinematics_solver = RobotKinematics(
    urdf_path="/home/TODO/lerobot/examples/so100_to_so100_EE/SO101/so101_new_calib.urdf",
    target_frame_name="gripper_frame_link",
    joint_names=arm_motors,  # ONLY ARM MOTORS for kinematics
)

# Build pipeline to convert teleop joints to EE action
leader_to_ee = RobotProcessorPipeline[RobotAction, RobotAction](
    steps=[
        ForwardKinematicsJointsToEE(
            kinematics=leader_kinematics_solver, motor_names=arm_motors
        ),
    ],
    to_transition=robot_action_to_transition,
    to_output=transition_to_robot_action,
)

# build pipeline to convert EE action to robot joints
ee_to_follower_joints = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
    [
        EEBoundsAndSafety(
            end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
            max_ee_step_m=0.10,
        ),
        InverseKinematicsEEToJoints(
            kinematics=follower_kinematics_solver,
            motor_names=arm_motors,  # ONLY ARM MOTORS
            initial_guess_current_joints=False,
        ),
    ],
    to_transition=robot_action_observation_to_transition,
    to_output=transition_to_robot_action,
)

# Init rerun viewer
init_rerun(session_name="so100_so100_EE_teleop_fixed")

print("\n" + "="*50)
print("ENHANCED TELEOPERATION WITH EXPLICIT GRIPPER")
print("="*50)
print("✓ Arm joints: Processed through kinematic pipeline")
print("✓ Gripper: Direct position copying (bypasses kinematics)")
print("Move the leader arm and gripper!")
print("Press Ctrl+C to stop...")
print("="*50)

loop_count = 0

try:
    while True:
        t0 = time.perf_counter()

        robot_obs = follower.get_observation()
        leader_joints_obs = leader.get_action()
        leader_ee_act = leader_to_ee(leader_joints_obs)
        follower_joints_act = ee_to_follower_joints((leader_ee_act, robot_obs))
        if "gripper.pos" in leader_joints_obs:
            leader_gripper_pos = leader_joints_obs["gripper.pos"]
            follower_joints_act["gripper.pos"] = leader_gripper_pos
        else:
            print("No gripper.pos found in leader action!")

        result = follower.send_action(follower_joints_act)

        log_rerun_data(observation=leader_ee_act, action=follower_joints_act)

        loop_count += 1
        busy_wait(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

except KeyboardInterrupt:
    print("\n" + "="*50)
    print("STOPPING TELEOPERATION")
    print("="*50)

finally:
    try:
        follower.disconnect()
        leader.disconnect()
        print("✓ Robots disconnected successfully")
    except Exception as e:
        print(f"Disconnect error: {e}")

print("Enhanced teleoperation complete!")