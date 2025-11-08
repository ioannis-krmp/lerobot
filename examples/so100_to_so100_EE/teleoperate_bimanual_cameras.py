#!/usr/bin/env python

import time
from threading import Thread, Lock
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
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

@dataclass
class ArmConfig:
    follower_port: str
    leader_port: str
    arm_id: str

# Teleoperation arm configurations
right_arm = ArmConfig(follower_port="/dev/ttyACM1", leader_port="/dev/ttyACM2", arm_id="right_arm")
left_arm = ArmConfig(follower_port="/dev/ttyACM0", leader_port="/dev/ttyACM3", arm_id="left_arm")

# Camera configurations for workspace monitoring
workspace_cameras = {
    "top_view": OpenCVCameraConfig(
        index_or_path="/dev/video2",
        fps=15,
        width=640,
        height=480,
    ),
    "top_left_view": OpenCVCameraConfig(
        index_or_path="/dev/video6", 
        fps=15,
        width=640,
        height=480,
    ),
    "close_view": OpenCVCameraConfig(
        index_or_path="/dev/video4", 
        fps=15,
        width=640,
        height=480,
    ),
}

class BimanualTeleoperation:
    def __init__(self, right_arm_config: ArmConfig, left_arm_config: ArmConfig, camera_configs: Dict[str, OpenCVCameraConfig]):
        self.arms = {}
        self.cameras = {}
        self.data_lock = Lock()
        self.running = False
        
        self._initialize_cameras(camera_configs)
        
        for arm_name, config in [("right", right_arm_config), ("left", left_arm_config)]:
            self._initialize_arm(arm_name, config)
    
    def _initialize_cameras(self, camera_configs: Dict[str, OpenCVCameraConfig]):
        print("Initializing cameras...")
        for camera_name, config in camera_configs.items():
            try:
                camera = OpenCVCamera(config)
                camera.connect()
                self.cameras[camera_name] = camera
                print(f"✓ {camera_name} camera connected: {config.index_or_path}")
            except Exception as e:
                print(f"✗ Failed to connect {camera_name} camera: {e}")
                # Continue without this camera rather than failing completely
    
    def _initialize_arm(self, arm_name: str, config: ArmConfig):
        print(f"Initializing {arm_name} arm...")

        follower_config = SO100FollowerConfig(port=config.follower_port, id=f"follower_{config.arm_id}", use_degrees=True)
        leader_config = SO100LeaderConfig(port=config.leader_port, id=f"leader_{config.arm_id}")

        follower = SO100Follower(follower_config)
        leader = SO100Leader(leader_config)
        
        follower.connect()
        leader.connect()
        
        print(f"{arm_name.capitalize()} arm - Leader motors: {list(leader.bus.motors.keys())}")
        print(f"{arm_name.capitalize()} arm - Follower motors: {list(follower.bus.motors.keys())}")
        
        arm_motors = [m for m in follower.bus.motors.keys() if m != "gripper"]
        print(f"{arm_name.capitalize()} arm motors (for kinematics): {arm_motors}")

        urdf_path = "/home/TODO/lerobot/examples/so100_to_so100_EE/SO101/so101_new_calib.urdf"

        follower_kinematics = RobotKinematics(urdf_path=urdf_path, target_frame_name="gripper_frame_link", joint_names=arm_motors)
        leader_kinematics = RobotKinematics(urdf_path=urdf_path, target_frame_name="gripper_frame_link", joint_names=arm_motors)
        
        # Build processing pipelines
        leader_to_ee = RobotProcessorPipeline[RobotAction, RobotAction](
            steps=[
                ForwardKinematicsJointsToEE(
                    kinematics=leader_kinematics, 
                    motor_names=arm_motors
                ),
            ],
            to_transition=robot_action_to_transition,
            to_output=transition_to_robot_action,
        )
        
        ee_to_follower_joints = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
            [
                EEBoundsAndSafety(
                    end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
                    max_ee_step_m=0.10,
                ),
                InverseKinematicsEEToJoints(
                    kinematics=follower_kinematics,
                    motor_names=arm_motors,
                    initial_guess_current_joints=False,
                ),
            ],
            to_transition=robot_action_observation_to_transition,
            to_output=transition_to_robot_action,
        )
        
        # Store arm data
        self.arms[arm_name] = {
            "follower": follower,
            "leader": leader,
            "arm_motors": arm_motors,
            "leader_to_ee": leader_to_ee,
            "ee_to_follower": ee_to_follower_joints,
            "follower_kinematics": follower_kinematics,
            "leader_kinematics": leader_kinematics,
        }
    
    def _capture_camera_frames(self) -> Dict[str, np.ndarray]:
        frames = {}
        for camera_name, camera in self.cameras.items():
            try:
                frame = camera.read()
                if frame is not None:
                    frames[camera_name] = frame
                else:
                    print(f"Warning: Failed to capture frame from {camera_name}")
            except Exception as e:
                print(f"Error capturing from {camera_name}: {e}")
        return frames
    
    def _control_arm(self, arm_name: str):
        arm_data = self.arms[arm_name]
        follower = arm_data["follower"]
        leader = arm_data["leader"]
        leader_to_ee = arm_data["leader_to_ee"]
        ee_to_follower = arm_data["ee_to_follower"]
        
        print(f"Starting {arm_name} arm control thread...")
        
        while self.running:
            try:
                t0 = time.perf_counter()
                
                robot_obs = follower.get_observation()
                leader_joints_obs = leader.get_action()
                
                # Process through kinematic pipeline
                leader_ee_act = leader_to_ee(leader_joints_obs)
                follower_joints_act = ee_to_follower((leader_ee_act, robot_obs))
                
                # Handle gripper separately
                if "gripper.pos" in leader_joints_obs:
                    leader_gripper_pos = leader_joints_obs["gripper.pos"]
                    follower_joints_act["gripper.pos"] = leader_gripper_pos
                else:
                    print(f"Warning: No gripper.pos found in {arm_name} arm leader action!")
                
                # Send action to follower
                result = follower.send_action(follower_joints_act)
                
                # Capture camera frames (only for right arm to avoid double capture)
                camera_frames = {}
                if arm_name == "right":
                    camera_frames = self._capture_camera_frames()
                
                # Log data with arm prefix
                with self.data_lock:
                    log_data = {
                        **{f"{arm_name}_{k}": v for k, v in leader_ee_act.items()},
                        **{f"{arm_name}_{k}": v for k, v in follower_joints_act.items()},
                        **camera_frames  # Add camera frames to logged data
                    }
                    log_rerun_data(observation=log_data)
                
                # Maintain loop rate
                busy_wait(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
                
            except Exception as e:
                print(f"Error in {arm_name} arm control: {e}")
                break
        
        print(f"{arm_name.capitalize()} arm control thread stopped.")
    
    def start_teleoperation(self):
        print("\n" + "="*60)
        print("BIMANUAL TELEOPERATION WITH CAMERAS & KINEMATIC PROCESSING")
        print("="*60)
        print("Both arms: Processed through kinematic pipeline")
        print("Grippers: Direct position copying (bypasses kinematics)")
        print(f"Cameras: {len(self.cameras)} workspace monitoring cameras")
        print("Move both leader arms and grippers!")
        print("Press Ctrl+C to stop...")
        print("="*60)
        
        # Initialize rerun viewer
        init_rerun(session_name="bimanual_so100_teleop_with_cameras")
        
        self.running = True
        
        # Start control threads for both arms
        threads = []
        for arm_name in self.arms.keys():
            thread = Thread(target=self._control_arm, args=(arm_name,), daemon=True)
            thread.start()
            threads.append(thread)
        
        try:
            # Keep main thread alive
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n" + "="*60)
            print("STOPPING BIMANUAL TELEOPERATION")
            print("="*60)
        finally:
            self.running = False
            
            # Wait for threads to finish
            for thread in threads:
                thread.join(timeout=1.0)
            
            self.disconnect_all()
    
    def disconnect_all(self):
        """Disconnect all robots and cameras"""
        print("Disconnecting all robots and cameras...")
        
        for arm_name, arm_data in self.arms.items():
            try:
                arm_data["follower"].disconnect()
                arm_data["leader"].disconnect()
                print(f"✓ {arm_name.capitalize()} arm disconnected successfully")
            except Exception as e:
                print(f"✗ {arm_name.capitalize()} arm disconnect error: {e}")
        
        for camera_name, camera in self.cameras.items():
            try:
                camera.disconnect()
                print(f"✓ {camera_name} camera disconnected successfully")
            except Exception as e:
                print(f"✗ {camera_name} camera disconnect error: {e}")


def main():
    try:
        bimanual_system = BimanualTeleoperation(right_arm, left_arm, workspace_cameras)
        
        bimanual_system.start_teleoperation()
        
    except Exception as e:
        print(f"Error initializing bimanual system: {e}")
        return
    
    print("Bimanual teleoperation with cameras complete!")


if __name__ == "__main__":
    main()