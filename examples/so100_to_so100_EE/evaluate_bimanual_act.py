#!/usr/bin/env python
"""
Evaluate trained ACT policy on bimanual robot using LeRobot framework.

This script loads a trained ACT checkpoint and runs it on the real bimanual robot
to evaluate performance on the spoon feeding task.

Usage:
    python evaluate_bimanual_act.py --checkpoint last --num_episodes 5
    python evaluate_bimanual_act.py --checkpoint 030000 --num_episodes 3
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.bi_so100_follower.config_bi_so100_follower import BiSO100FollowerConfig
from lerobot.scripts.lerobot_record import RecordConfig, DatasetRecordConfig, record
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


def main():
    parser = argparse.ArgumentParser(description="Evaluate ACT model on bimanual robot")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="last",
        help="Checkpoint to load (e.g., '030000', 'last', or full path)",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="outputs/act_bimanual_quick_test",
        help="Directory containing checkpoints (relative to examples/so100_to_so100_EE/)",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes to run",
    )
    args = parser.parse_args()

    # =======================================================
    # Configuration (same as recording setup)
    # =======================================================

    CONTROL_FPS = 20
    CAMERA_FPS = 20
    EPISODE_TIME_S = 100  # Safety backup
    RESET_TIME_S = 20

    LEFT_ARM_FOLLOWER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AB9068616-if00"
    RIGHT_ARM_FOLLOWER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AB9068630-if00"
    TOP_VIEW_CAMERA = "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_39D25ECF-video-index0"
    TOP_LEFT_VIEW_CAMERA = "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_A0725ECF-video-index0"
    CLOSE_VIEW_CAMERA = "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_92A6D6DF-video-index0"

    # =======================================================
    # Load trained policy
    # =======================================================

    # Build checkpoint path
    if args.checkpoint == "last":
        checkpoint_path = Path(args.model_dir) / "checkpoints" / "last" / "pretrained_model"
    elif "/" in args.checkpoint or Path(args.checkpoint).exists():
        # Full path provided
        checkpoint_path = Path(args.checkpoint)
    else:
        # Checkpoint number provided (e.g., "030000")
        checkpoint_path = Path(args.model_dir) / "checkpoints" / args.checkpoint / "pretrained_model"

    checkpoint_path = checkpoint_path.resolve()

    print("=" * 80)
    print("EVALUATING TRAINED ACT POLICY ON BIMANUAL ROBOT")
    print("=" * 80)
    print(f"Policy checkpoint: {checkpoint_path}")

    if not checkpoint_path.exists():
        print(f"\n❌ Error: Checkpoint not found at {checkpoint_path}")
        print(f"\nAvailable checkpoints in {args.model_dir}/checkpoints/:")
        checkpoints_dir = Path(args.model_dir) / "checkpoints"
        if checkpoints_dir.exists():
            for ckpt in sorted(checkpoints_dir.iterdir()):
                if ckpt.is_dir() and not ckpt.name.startswith('.'):
                    print(f"  - {ckpt.name}")
        return

    # Load policy
    print("Loading ACT policy...")
    policy = ACTPolicy.from_pretrained(checkpoint_path)
    policy.eval()  # Set to evaluation mode

    print(f"✓ Policy loaded successfully")
    print(f"  - Policy type: ACT")
    print(f"  - Chunk size: {policy.config.chunk_size}")
    print(f"  - Model dimension: {policy.config.dim_model}")
    print(f"  - Device: {policy.config.device}")

    # =======================================================
    # Setup cameras and robot
    # =======================================================

    cameras = {
        "top_view": OpenCVCameraConfig(
            index_or_path=TOP_VIEW_CAMERA,
            fps=CAMERA_FPS,
            width=640,
            height=480,
        ),
        "top_left_view": OpenCVCameraConfig(
            index_or_path=TOP_LEFT_VIEW_CAMERA,
            fps=CAMERA_FPS,
            width=640,
            height=480,
        ),
        "close_view": OpenCVCameraConfig(
            index_or_path=CLOSE_VIEW_CAMERA,
            fps=CAMERA_FPS,
            width=640,
            height=480,
        ),
    }

    robot_config = BiSO100FollowerConfig(
        id="follower_bimanual",
        left_arm_port=LEFT_ARM_FOLLOWER_PORT,
        right_arm_port=RIGHT_ARM_FOLLOWER_PORT,
        left_arm_use_degrees=True,
        right_arm_use_degrees=True,
        cameras=cameras,
    )

    # =======================================================
    # Create evaluation dataset
    # =======================================================

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dataset_id = f"ioannis-krmp/eval_bimanual_spoon_feeding_{args.checkpoint}_{timestamp}"

    dataset_config = DatasetRecordConfig(
        repo_id=eval_dataset_id,
        single_task="spoon_feeding_evaluation",
        fps=CONTROL_FPS,
        episode_time_s=EPISODE_TIME_S,
        reset_time_s=RESET_TIME_S,
        num_episodes=args.num_episodes,
        video=True,
        push_to_hub=False,  # Keep locally for now
        private=False,
        num_image_writer_processes=0,
        num_image_writer_threads_per_camera=6,
    )

    # =======================================================
    # Run evaluation with policy
    # =======================================================

    record_config = RecordConfig(
        robot=robot_config,
        teleop=None,  # No teleoperation during evaluation
        dataset=dataset_config,
        policy=policy,  # Use trained policy for control
        display_data=True,
        play_sounds=True,
        resume=False,
    )

    print("=" * 80)
    print(f"EVALUATION SETUP")
    print("=" * 80)
    print(f"Episodes to run: {args.num_episodes}")
    print(f"Control FPS: {CONTROL_FPS}")
    print(f"Max episode time: {EPISODE_TIME_S}s")
    print(f"Evaluation dataset: {eval_dataset_id}")
    print(f"Local path: ~/.cache/huggingface/lerobot/{eval_dataset_id}")
    print("=" * 80)
    print("\n⚠️  SAFETY REMINDERS:")
    print("  - Clear the workspace before starting")
    print("  - Robot will move autonomously based on trained policy")
    print("  - Press ESC to emergency stop if needed")
    print("  - Position robot at start before each episode")
    print("\n🎮 CONTROLS:")
    print("  ➡️  RIGHT ARROW - Finish episode (policy will run until then)")
    print("  ⬅️  LEFT ARROW  - Re-record episode")
    print("  🔴 ESC         - Stop evaluation")
    print("=" * 80)
    print("\nPress Enter to start evaluation...")
    input()

    try:
        print("\n🚀 Starting policy evaluation...")
        dataset = record(record_config)

        print("\n" + "=" * 80)
        print("✅ EVALUATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Episodes recorded: {dataset.num_episodes}")
        print(f"Dataset location: {dataset.repo_id}")
        print(f"Local path: ~/.cache/huggingface/lerobot/{eval_dataset_id}")
        print("\n💡 Next steps:")
        print("  1. Review the recorded episodes and videos")
        print("  2. Check how well the policy performed")
        print("  3. If performance is good, consider running full training (200K steps)")
        print("  4. If performance is poor, collect more training data (aim for 60-80 episodes)")
        print("=" * 80)

    except ValueError as e:
        print(f"\n❌ Configuration error: {e}")
        logging.exception("Configuration error details:")

    except KeyboardInterrupt:
        print("\n⚠️  Evaluation stopped by user")
        print("Partial data for completed episodes has been saved")

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        print("💡 Common issues:")
        print("  - Camera/robot connection problems")
        print("  - Policy checkpoint incompatible with dataset")
        print("  - Insufficient GPU memory")
        logging.exception("Full error details:")

    finally:
        print("\n🏁 Evaluation session finished")


if __name__ == "__main__":
    main()
