#!/usr/bin/env python
"""
Train SmolVLA on Multiple Tasks with Language Conditioning

This script trains a SINGLE SmolVLA model that can perform multiple tasks
by conditioning on language instructions.

Tasks:
  Task 1 (70 episodes): "scoop food from the bowl and feed the person"
  Task 2 (70 episodes): [Add your task 2 description here]

The model learns to:
- Share visual representations across tasks
- Distinguish tasks via language instructions
- Perform both tasks with one unified policy

Usage:
    # 1. Ensure Task 2 has ~70 episodes (balanced with Task 1)
    # 2. Update TASK_2_INSTRUCTION below
    # 3. Merge datasets (script will guide you)
    # 4. Run training:
    python train_smolvla_multitask.py

Requirements:
    pip install -e ".[smolvla]"
"""

import sys
from pathlib import Path

# Add lerobot to path
lerobot_path = Path(__file__).parent.parent.parent / "src"
if lerobot_path.exists():
    sys.path.insert(0, str(lerobot_path))

from lerobot.scripts.lerobot_train import main as train_main


def check_datasets():
    """Check if both datasets exist and are ready."""
    task1_path = "TODO"
    task2_path = "TODO"

    if not task1_path.exists():
        print(f"ERROR: Task 1 dataset not found at {task1_path}")
        return False

    if not task2_path.exists():
        print(f"ERROR: Task 2 dataset not found at {task2_path}")
        return False

    # Check Task 2 episode count
    import json
    task2_info = task2_path / "meta" / "info.json"
    if task2_info.exists():
        with open(task2_info, 'r') as f:
            info = json.load(f)
            task2_episodes = info.get('total_episodes', 0)
            print(f"Task 2 episodes: {task2_episodes}")

            if task2_episodes < 50:
                print(f"WARNING: Task 2 only has {task2_episodes} episodes.")
                print(f"         Recommend collecting ~70 episodes for balanced training.")
                response = input("Continue anyway? (y/n): ")
                if response.lower() != 'y':
                    return False

    return True


def merge_datasets_info():
    """
    Print instructions for merging datasets.

    For multi-task training, we need to:
    1. Merge both datasets into one
    2. Add language annotations to each episode

    This is currently a manual process in lerobot.
    """
    print("\n" + "=" * 80)
    print("DATASET MERGING REQUIRED")
    print("=" * 80)
    print("To train on multiple tasks, you need to merge datasets with language labels.")
    print()
    print("Method 1: Use lerobot CLI (if available)")
    print("-" * 80)
    print("lerobot-merge-datasets \\")
    print("  --dataset1=/path/to/task1 \\")
    print("  --dataset2=/path/to/task2 \\")
    print("  --task1_instruction='scoop food from the bowl and feed the person' \\")
    print("  --task2_instruction='[your task 2 instruction]' \\")
    print("  --output=bimanual_multitask_140ep")
    print()
    print("Method 2: Manual Script (see merge_datasets_example.py)")
    print("-" * 80)
    print("Create a script that:")
    print("  1. Loads both LeRobotDatasets")
    print("  2. Adds 'language_instruction' field to each episode")
    print("  3. Concatenates episodes")
    print("  4. Saves as new dataset")
    print()
    print("Method 3: Train separately then combine (not recommended)")
    print("=" * 80)
    print()


def main():
    """Train SmolVLA on multiple tasks with language conditioning."""

    # Check datasets first
    if not check_datasets():
        print("\nDataset check failed. Please fix issues above.")
        return

    # Configuration
    # TODO: Update these paths once you've merged the datasets

    MERGED_DATASET_REPO_ID = "TODO"
    MERGED_DATASET_PATH = "TODO"

    # Task instructions (UPDATE TASK 2!)
    TASK_1_INSTRUCTION = "scoop food from the bowl and feed the person"
    TASK_2_INSTRUCTION = "pick up the cup and hand it to the person"  # UPDATE THIS!

    OUTPUT_DIR = "outputs/smolvla_multitask_140ep"

    # Check if merged dataset exists
    if not MERGED_DATASET_PATH.exists():
        print(f"\nERROR: Merged dataset not found at {MERGED_DATASET_PATH}")
        merge_datasets_info()
        print("\nPlease create merged dataset first, then run this script again.")
        return

    original_argv = sys.argv.copy()

    sys.argv = [
        sys.argv[0],

        # SmolVLA Policy Configuration
        "--policy.type=smolvla",

        # Optionally start from pretrained SmolVLA
        # "--policy.path=lerobot/smolvla_base",

        # Model architecture (optimized for 8GB GPU)
        "--policy.n_obs_steps=4",
        "--policy.chunk_size=50",
        "--policy.n_action_steps=50",

        # Multi-task: Language instructions are in dataset
        # The model will read 'language_instruction' field from each episode

        # Dataset (MERGED dataset with both tasks)
        f"--dataset.repo_id={MERGED_DATASET_REPO_ID}",
        f"--dataset.root={MERGED_DATASET_PATH}",

        # Weights & Biases
        "--wandb.enable=true",
        "--wandb.project=lerobot-bimanual-smolvla",
        "--wandb.notes=SmolVLA Multi-Task: Task1 + Task2 (140 episodes)",

        # Image augmentation
        "--dataset.image_transforms.enable=true",
        "--dataset.image_transforms.max_num_transforms=2",

        # Training parameters (more data = more steps)
        "--batch_size=8",           # May need to reduce to 4-6 if OOM
        "--steps=300000",           # More steps for multi-task (140 episodes)
        "--eval_freq=10000",
        "--save_freq=10000",
        "--log_freq=250",
        "--num_workers=8",

        # Optimizer
        "--optimizer.lr=1e-4",
        "--optimizer.weight_decay=1e-6",

        # Output
        f"--output_dir={OUTPUT_DIR}",
        "--seed=1000",
    ]

    print("=" * 80)
    print("SMOLVLA MULTI-TASK TRAINING")
    print("=" * 80)
    print(f"Dataset: {MERGED_DATASET_PATH.name}")
    print("Policy: SmolVLA (Vision-Language-Action)")
    print("Episodes: 140 (70 Task1 + 70 Task2) | FPS: 20")
    print("-" * 80)
    print("Tasks:")
    print(f"  Task 1: '{TASK_1_INSTRUCTION}'")
    print(f"  Task 2: '{TASK_2_INSTRUCTION}'")
    print("-" * 80)
    print("GPU: RTX 4070 8GB")
    print("Batch Size: 8 (may need reduction if OOM)")
    print("Training Steps: 300K | Expected Time: ~20-25 hours")
    print("-" * 80)
    print("MULTI-TASK ADVANTAGES:")
    print("  ✓ One model performs both tasks")
    print("  ✓ Shared visual features (better generalization)")
    print("  ✓ More training data (140 episodes)")
    print("  ✓ Language-conditioned task switching")
    print("=" * 80)
    print("\nStarting multi-task training...\n")

    try:
        train_main()
        print("\n" + "=" * 80)
        print("SMOLVLA MULTI-TASK TRAINING COMPLETED!")
        print(f"Checkpoints: {OUTPUT_DIR}/checkpoints/")
        print("-" * 80)
        print("Model can now perform:")
        print(f"  - Task 1 when given: '{TASK_1_INSTRUCTION}'")
        print(f"  - Task 2 when given: '{TASK_2_INSTRUCTION}'")
        print("=" * 80)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n" + "=" * 80)
            print("GPU OUT OF MEMORY!")
            print("=" * 80)
            print("Solutions:")
            print("  1. Reduce batch_size to 4 or 6")
            print("  2. Reduce n_obs_steps to 2")
            print("  3. Use gradient accumulation")
            print("=" * 80)
        raise
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
