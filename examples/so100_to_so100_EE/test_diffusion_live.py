#!/usr/bin/env python
"""
Test trained Diffusion policy on bimanual robot.

This script loads a trained Diffusion checkpoint and runs it directly on the robot.

Usage:
    python test_diffusion_live.py --checkpoint 120000
    python test_diffusion_live.py --checkpoint last
"""

import argparse
import time
from pathlib import Path

import torch
import numpy as np

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.bi_so100_follower.config_bi_so100_follower import BiSO100FollowerConfig
from lerobot.robots.bi_so100_follower.bi_so100_follower import BiSO100Follower
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.utils import build_inference_frame, make_robot_action


def main():
    parser = argparse.ArgumentParser(description="Test Diffusion model on robot")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="120000",
        help="Checkpoint to load (e.g., '120000' or 'last')",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="outputs/diffusion_bimanual_70ep",
        help="Directory containing checkpoints",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=500,
        help="Maximum steps per run (safety limit)",
    )
    args = parser.parse_args()

    # Camera and robot ports (same as recording)
    TOP_VIEW_CAMERA = "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_39D25ECF-video-index0"
    TOP_LEFT_VIEW_CAMERA = "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_A0725ECF-video-index0"
    CLOSE_VIEW_CAMERA = "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_92A6D6DF-video-index0"
    LEFT_ARM_FOLLOWER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AB9068616-if00"
    RIGHT_ARM_FOLLOWER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AB9068630-if00"

    # Build checkpoint path
    if args.checkpoint == "last":
        checkpoint_path = Path(args.model_dir) / "checkpoints" / "last" / "pretrained_model"
    else:
        checkpoint_path = Path(args.model_dir) / "checkpoints" / args.checkpoint / "pretrained_model"

    checkpoint_path = checkpoint_path.resolve()

    print("=" * 80)
    print("TESTING DIFFUSION POLICY ON BIMANUAL ROBOT")
    print("=" * 80)
    print(f"Loading checkpoint: {checkpoint_path}")

    if not checkpoint_path.exists():
        print(f"\nError: Checkpoint not found at {checkpoint_path}")
        print(f"\nAvailable checkpoints:")
        checkpoints_dir = Path(args.model_dir) / "checkpoints"
        if checkpoints_dir.exists():
            for ckpt in sorted(checkpoints_dir.iterdir()):
                if ckpt.is_dir() and not ckpt.name.startswith('.'):
                    print(f"  - {ckpt.name}")
        return

    # Load policy and preprocessors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    # Load policy
    policy = DiffusionPolicy.from_pretrained(checkpoint_path)
    policy.eval()
    policy.to(device)

    # Load dataset metadata for stats
    DATASET_REPO_ID = "ioannis-krmp/bimanual_task1_70ep"
    DATASET_PATH = Path.home() / ".cache/huggingface/lerobot/ioannis-krmp/bimanual_tasks_dataset_task_1_20251115_201252_70_episodes"

    print(f"Loading dataset stats from: {DATASET_PATH}")
    dataset_meta = LeRobotDatasetMetadata(DATASET_REPO_ID, root=DATASET_PATH)

    # Load preprocessor and postprocessor (critical for normalization!)
    try:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=checkpoint_path,
            dataset_stats=dataset_meta.stats,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )
        print(f"Loaded preprocessor and postprocessor with dataset stats")
    except Exception as e:
        print(f"Warning: Could not load preprocessor/postprocessor: {e}")
        print(f"   Inference may not work correctly without normalization!")
        import traceback
        traceback.print_exc()
        preprocessor = None
        postprocessor = None

    print(f"Policy loaded successfully!")
    print(f"  Device: {device}")
    print(f"  n_obs_steps: {policy.config.n_obs_steps}")
    print(f"  horizon: {policy.config.horizon}")
    print(f"  n_action_steps: {policy.config.n_action_steps}")
    print("=" * 80)

    # Setup cameras
    cameras_config = {
        "top_view": OpenCVCameraConfig(
            index_or_path=TOP_VIEW_CAMERA,
            fps=20,
            width=640,
            height=480,
        ),
        "top_left_view": OpenCVCameraConfig(
            index_or_path=TOP_LEFT_VIEW_CAMERA,
            fps=20,
            width=640,
            height=480,
        ),
        "close_view": OpenCVCameraConfig(
            index_or_path=CLOSE_VIEW_CAMERA,
            fps=20,
            width=640,
            height=480,
        ),
    }

    # Setup robot
    robot_config = BiSO100FollowerConfig(
        id="follower_bimanual",
        left_arm_port=LEFT_ARM_FOLLOWER_PORT,
        right_arm_port=RIGHT_ARM_FOLLOWER_PORT,
        left_arm_use_degrees=True,
        right_arm_use_degrees=True,
        cameras=cameras_config,
    )

    print("\nConnecting to robot and cameras...")
    robot = BiSO100Follower(robot_config)
    robot.connect()

    print("Connected!")
    print("\n" + "=" * 80)
    print("SAFETY:")
    print("  - Make sure workspace is clear")
    print("  - Robot will move autonomously")
    print("  - Press Ctrl+C to emergency stop")
    print(f"  - Maximum {args.max_steps} steps (25 seconds @ 20Hz)")
    print("=" * 80)
    print("\nPosition the robot at the starting pose")
    print("Press Enter to start...")
    input()

    try:
        print("\nStarting Diffusion policy execution...\n")

        policy.reset()  # Reset policy state
        step = 0

        start_time = time.time()

        while step < args.max_steps:
            step_start = time.time()

            # Get robot observation (includes cameras!)
            observation_dict = robot.get_observation()

            # Build inference frame using LeRobot's official helper function
            obs_frame = build_inference_frame(
                observation=observation_dict,
                ds_features=dataset_meta.features,
                device=device
            )

            # Apply preprocessing (normalization)
            if preprocessor is not None:
                obs_frame = preprocessor(obs_frame)

            # Get action from policy
            with torch.no_grad():
                action_tensor = policy.select_action(obs_frame)

            # Apply postprocessing (denormalization)
            if postprocessor is not None:
                action_tensor = postprocessor(action_tensor)

            # Convert action tensor to robot dict format using LeRobot's official helper
            robot_action = make_robot_action(action_tensor, dataset_meta.features)

            # Debug: print what we're sending
            if step == 0:
                print(f"\nSending action format check:")
                print(f"  Sample action keys: {list(robot_action.keys())[:3]}")
                print(f"  Sample action values: {[robot_action[k] for k in list(robot_action.keys())[:3]]}")
                print()

            # Send action to robot
            robot.send_action(robot_action)

            step += 1

            # Print progress
            if step % 20 == 0:  # Every second
                elapsed = time.time() - start_time
                print(f"  Step {step}/{args.max_steps} | {elapsed:.1f}s elapsed")

            # Debug: print predictions periodically
            if step <= 5 or step % 50 == 0:
                motor_names_short = [
                    "L_pan", "L_lift", "L_elbow", "L_wflex", "L_wroll", "L_grip",
                ]

                # Extract current state from observation (left arm only)
                state_values = [
                    observation_dict["left_shoulder_pan.pos"],
                    observation_dict["left_shoulder_lift.pos"],
                    observation_dict["left_elbow_flex.pos"],
                    observation_dict["left_wrist_flex.pos"],
                    observation_dict["left_wrist_roll.pos"],
                    observation_dict["left_gripper.pos"],
                ]

                # Extract action values from robot_action dict (left arm only)
                action_values = [
                    robot_action["left_shoulder_pan.pos"],
                    robot_action["left_shoulder_lift.pos"],
                    robot_action["left_elbow_flex.pos"],
                    robot_action["left_wrist_flex.pos"],
                    robot_action["left_wrist_roll.pos"],
                    robot_action["left_gripper.pos"],
                ]

                print(f"\n  Step {step}:")
                print(f"    State:  {' '.join([f'{v:7.2f}' for v in state_values])}")
                print(f"    Action: {' '.join([f'{v:7.2f}' for v in action_values])}")
                print(f"    Delta:  {' '.join([f'{action_values[i]-state_values[i]:7.2f}' for i in range(6)])}")

            # Maintain 20Hz loop
            elapsed = time.time() - step_start
            sleep_time = max(0, (1/20) - elapsed)
            time.sleep(sleep_time)

        print(f"\nCompleted {step} steps!")

    except KeyboardInterrupt:
        print("\n\nStopped by user (Ctrl+C)")

    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\nDisconnecting...")
        robot.disconnect()
        print("Disconnected")
        print("\n" + "=" * 80)
        print("TEST COMPLETE")
        print("=" * 80)
        print("\nResults:")
        print("  If robot performed spoon feeding motion correctly -> Success!")
        print("  If robot moved erratically -> Check normalization/preprocessing")
        print("  If robot barely moved -> Check action scaling")


if __name__ == "__main__":
    main()
