#!/usr/bin/env python
"""
Test trained SmolVLA policy on bimanual robot.

Usage:
    python test_smolvla_live.py --checkpoint 030000
    python test_smolvla_live.py --checkpoint last
"""

import argparse
import time
from pathlib import Path

import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.bi_so100_follower.config_bi_so100_follower import BiSO100FollowerConfig
from lerobot.robots.bi_so100_follower.bi_so100_follower import BiSO100Follower
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import build_inference_frame, make_robot_action


def main():
    parser = argparse.ArgumentParser(description="Test SmolVLA model on robot")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="030000",
        help="Checkpoint to load (e.g., '030000' or 'last')",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="outputs/smolvla_task1_70ep",
        help="Directory containing checkpoints",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=500,
        help="Maximum steps per run (safety limit)",
    )
    args = parser.parse_args()

    # Camera and robot ports
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
    print("TESTING SMOLVLA POLICY ON BIMANUAL ROBOT")
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

    # Load policy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    policy = SmolVLAPolicy.from_pretrained(checkpoint_path)
    policy.eval()
    policy.to(device)

    # Load dataset metadata
    DATASET_REPO_ID = "ioannis-krmp/bimanual_task1_70ep"
    DATASET_PATH = Path.home() / ".cache/huggingface/lerobot/ioannis-krmp/bimanual_tasks_dataset_task_1_20251115_201252_70_episodes"

    print(f"Loading dataset stats from: {DATASET_PATH}")
    dataset_meta = LeRobotDatasetMetadata(DATASET_REPO_ID, root=DATASET_PATH)

    # Load preprocessor and postprocessor
    try:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=checkpoint_path,
            dataset_stats=dataset_meta.stats,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )
        print(f"Loaded preprocessor and postprocessor")
    except Exception as e:
        print(f"Warning: Could not load preprocessor/postprocessor: {e}")
        preprocessor = None
        postprocessor = None

    print(f"Policy loaded!")
    print(f"  Device: {device}")
    print(f"  n_obs_steps: {policy.config.n_obs_steps}")
    print(f"  chunk_size: {policy.config.chunk_size}")
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
    print("SAFETY: Press Ctrl+C to emergency stop")
    print(f"Maximum {args.max_steps} steps (25 seconds @ 20Hz)")
    print("=" * 80)
    print("\nPosition robot at starting pose, then press Enter...")
    input()

    try:
        print("\nStarting SmolVLA policy execution...\n")

        policy.reset()
        step = 0
        start_time = time.time()

        while step < args.max_steps:
            step_start = time.time()

            observation_dict = robot.get_observation()

            obs_frame = build_inference_frame(
                observation=observation_dict,
                ds_features=dataset_meta.features,
                device=device
            )

            if preprocessor is not None:
                obs_frame = preprocessor(obs_frame)

            with torch.no_grad():
                action_tensor = policy.select_action(obs_frame)

            if postprocessor is not None:
                action_tensor = postprocessor(action_tensor)

            robot_action = make_robot_action(action_tensor, dataset_meta.features)

            robot.send_action(robot_action)

            step += 1

            if step % 20 == 0:
                elapsed = time.time() - start_time
                print(f"  Step {step}/{args.max_steps} | {elapsed:.1f}s")

            # Maintain 20Hz
            elapsed = time.time() - step_start
            time.sleep(max(0, (1/20) - elapsed))

        print(f"\nCompleted {step} steps!")

    except KeyboardInterrupt:
        print("\n\nStopped by user")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    finally:
        robot.disconnect()
        print("Disconnected")
        print("\n" + "=" * 80)
        print("SMOLVLA TEST COMPLETE")
        print("=" * 80)


if __name__ == "__main__":
    main()
