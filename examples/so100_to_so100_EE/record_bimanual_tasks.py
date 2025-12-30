#!/usr/bin/env python

"""
Record bimanual dataset for training LeRobot models.
This uses the official LeRobot recording framework with bimanual robots.
"""

import logging
from pathlib import Path

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.bi_so100_follower.config_bi_so100_follower import BiSO100FollowerConfig
from lerobot.scripts.lerobot_record import RecordConfig, DatasetRecordConfig, record
from lerobot.teleoperators.bi_so100_leader.config_bi_so100_leader import BiSO100LeaderConfig
from datetime import datetime


def main():
    """Record bimanual dataset for training LeRobot models"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    DATASET_REPO_ID = "TODO"
    DATASET_TASK = "spoon_feeding"
    #DATASET_TASK = "straw_drinking"
    #DATASET_TASK = "cleaning_with_sponge"
    NUM_EPISODES = 70  # Increase episodes for multiple demonstrations
    

    CONTROL_FPS = 20      # Good trade-off for smooth robot control
    CAMERA_FPS = 20       # Match camera rate to control rate
    EPISODE_TIME_S = 100   # 1 minute max per episode (safety backup)
    RESET_TIME_S = 20      # 15 seconds for manual reset between episodes

    LEFT_ARM_FOLLOWER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AB9068616-if00" 
    RIGHT_ARM_FOLLOWER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AB9068630-if00"
    LEFT_ARM_LEADER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AB9068609-if00"
    RIGHT_ARM_LEADER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AAF220183-if00"
    TOP_VIEW_CAMERA = "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_39D25ECF-video-index0"
    TOP_LEFT_VIEW_CAMERA = "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_A0725ECF-video-index0"
    CLOSE_VIEW_CAMERA = "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_92A6D6DF-video-index0"
    
    # =======================================================
    
    print("="*60)
    print("KEYBOARD-CONTROLLED BIMANUAL DATA RECORDING")
    print("="*60)
    print(f"Dataset: {DATASET_REPO_ID}")
    # Show the full local dataset folder path
    local_cache_path = Path.home() / ".cache/huggingface/lerobot" / DATASET_REPO_ID
    print(f"Local dataset folder: {local_cache_path}")
    print(f"Task: {DATASET_TASK}")
    print(f"Episodes: {NUM_EPISODES}")
    print(f"Max duration per episode: {EPISODE_TIME_S}s (or until you press RIGHT ARROW)")
    print(f"Reset time: {RESET_TIME_S}s")
    print("="*60)
    print("\n🎮 KEYBOARD CONTROLS DURING RECORDING:")
    print("  ➡️  RIGHT ARROW - Finish current episode and move to next")
    print("  ⬅️  LEFT ARROW  - Re-record current episode (if you mess up)")
    print("  🔴 ESC         - Stop all recording and save dataset")
    print("="*60)
    
    # Camera configurations
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
    
    
    # Create robot configuration
    robot_config = BiSO100FollowerConfig(
        id="follower_bimanual",
        left_arm_port=LEFT_ARM_FOLLOWER_PORT,
        right_arm_port=RIGHT_ARM_FOLLOWER_PORT,
        left_arm_use_degrees=True,
        right_arm_use_degrees=True,
        cameras=cameras,
    )
    
    # Create teleoperator configuration
    teleop_config = BiSO100LeaderConfig(
        id="leader_bimanual",
        left_arm_port=LEFT_ARM_LEADER_PORT,
        right_arm_port=RIGHT_ARM_LEADER_PORT,
    )
    
    # Create dataset configuration
    dataset_config = DatasetRecordConfig(
        repo_id=DATASET_REPO_ID,
        single_task=DATASET_TASK,
        fps=CONTROL_FPS,  # Use higher control frequency for smoother robot motion
        episode_time_s=EPISODE_TIME_S,
        reset_time_s=RESET_TIME_S,
        num_episodes=NUM_EPISODES,
        video=True,  # Encode videos for easier analysis
        push_to_hub=False,  # Set to True to upload to Hugging Face Hub
        private=False,
        num_image_writer_processes=0,  # Use threads only
        num_image_writer_threads_per_camera=6,  # Reduce threads to lower system load
    )
    
    # Create complete recording configuration
    record_config = RecordConfig(
        robot=robot_config,
        teleop=teleop_config,
        dataset=dataset_config,
        display_data=True,   # Show camera feeds during recording
        play_sounds=True,    # Audio feedback
        resume=True,       # Resume existing dataset if found
    )
    
    print("\nStarting recording setup...")
    print("Instructions:")
    print("- Perform your task naturally with the leader arms")
    print("- Press RIGHT ARROW when you finish a demonstration")
    print("- Press LEFT ARROW if you want to redo the current episode")
    print("- Press ESC when you're done with all recordings")
    print("- Episode timer is just a safety backup - use arrow keys to control!")
    print("- Camera feeds will be displayed for monitoring")
    print("\nValidating configuration...")
    
    # Validate robot configuration
    print(f"✓ Robot config: {robot_config.id}")
    print(f"✓ Teleop config: {teleop_config.id}")
    print(f"✓ Camera count: {len(cameras)}")
    print(f"✓ Dataset path: {DATASET_REPO_ID}")
    
    print("\nReady to start recording? Press Enter...")
    input()
    
    try:
        print("Initializing recording system...")
        # Start recording using LeRobot's recording system
        dataset = record(record_config)
        
        print("✅ Recording completed successfully!")
        print(f"📊 Dataset saved with {dataset.num_episodes} episodes")
        print(f"📁 Dataset location: {dataset.repo_id}")
        print(f"🎥 Cameras recorded: {list(cameras.keys())}")
        
        if hasattr(dataset, 'features'):
            print(f"📝 Dataset features: {list(dataset.features.keys())}")
            
    except ValueError as e:
        print(f"\n❌ Configuration error: {e}")
        print("💡 This might be due to:")
        print("- Invalid dataset repository ID format")
        print("- Missing or incorrect camera configuration")
        print("- Robot port conflicts")
        logging.exception("Configuration error details:")
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("💡 Check if all required dependencies are installed")
        logging.exception("Import error details:")
        
    except KeyboardInterrupt:
        print("\n⚠️  Recording interrupted by user")
        print("Data for completed episodes has been saved")
        
    except Exception as e:
        print(f"\n❌ Recording failed with error: {e}")
        print("💡 Common causes:")
        print("- Hardware connection issues")
        print("- Camera access problems") 
        print("- Dataset path/permission issues")
        print("- Robot configuration mismatches")
        logging.exception("Full error details:")
        
    finally:
        print("\n🏁 Recording session finished")


if __name__ == "__main__":
    main()