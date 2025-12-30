#!/usr/bin/env python
"""
ACT Training with Post-Training Best Model Selection

Since lerobot's training loop doesn't support custom callbacks, this script:
1. Runs normal training (logs to wandb)
2. After training completes, scans all checkpoints
3. Finds the checkpoint with lowest loss from wandb logs
4. Copies it to 'best_model' directory

Usage:
    python train_act_bimanual_with_best_tracking.py
"""

import sys
import shutil
from pathlib import Path
import json

# Add lerobot to path if needed
lerobot_path = Path(__file__).parent.parent.parent / "src"
if lerobot_path.exists():
    sys.path.insert(0, str(lerobot_path))

from lerobot.scripts.lerobot_train import main as train_main


def find_best_checkpoint_from_wandb(output_dir: Path):
    """
    After training, analyze wandb logs to find best checkpoint.
    Returns the checkpoint directory with lowest training loss.
    """
    try:
        import wandb
        api = wandb.Api()

        # Get the most recent run from this output directory
        runs = api.runs(
            "ioannis-karampinas-ntua/lerobot-bimanual-act",
            filters={"config.output_dir": str(output_dir)}
        )

        if not runs:
            print("Warning: No wandb runs found for this training")
            return None

        run = runs[0]  # Most recent
        history = run.history(keys=["loss", "_step"])

        if history.empty:
            print("Warning: No loss data found in wandb")
            return None

        # Find step with minimum loss
        min_loss_idx = history['loss'].idxmin()
        best_step = int(history.loc[min_loss_idx, '_step'])
        best_loss = history.loc[min_loss_idx, 'loss']

        print(f"\nBest checkpoint found via Wandb:")
        print(f"  Step: {best_step}")
        print(f"  Loss: {best_loss:.6f}")

        # Find corresponding checkpoint directory
        checkpoints_dir = output_dir / "checkpoints"
        for ckpt_dir in checkpoints_dir.glob("*"):
            if ckpt_dir.is_dir() and str(best_step).zfill(6) in ckpt_dir.name:
                return ckpt_dir

        return None

    except Exception as e:
        print(f"Error finding best checkpoint from wandb: {e}")
        return None


def find_best_checkpoint_from_logs(output_dir: Path):
    """
    Fallback: Parse training logs to find checkpoint with lowest loss.
    """
    import re

    log_file = output_dir / "train.log"
    if not log_file.exists():
        return None

    # Parse logs to find loss at each checkpoint step
    checkpoint_losses = {}

    with open(log_file, 'r') as f:
        current_step = None
        for line in f:
            # Look for step information
            step_match = re.search(r'step[:\s]+(\d+)', line, re.IGNORECASE)
            if step_match:
                current_step = int(step_match.group(1))

            # Look for loss
            loss_match = re.search(r'loss[:\s]+([\d.]+)', line, re.IGNORECASE)
            if loss_match and current_step:
                loss = float(loss_match.group(1))
                checkpoint_losses[current_step] = loss

    if not checkpoint_losses:
        return None

    # Find step with min loss
    best_step = min(checkpoint_losses, key=checkpoint_losses.get)
    best_loss = checkpoint_losses[best_step]

    print(f"\nBest checkpoint found from logs:")
    print(f"  Step: {best_step}")
    print(f"  Loss: {best_loss:.6f}")

    # Find checkpoint directory
    checkpoints_dir = output_dir / "checkpoints"
    for ckpt_dir in checkpoints_dir.glob("*"):
        if ckpt_dir.is_dir() and str(best_step).zfill(6) in ckpt_dir.name:
            return ckpt_dir

    return None


def copy_best_model(output_dir: Path):
    """After training, find and copy the best checkpoint."""
    print("\n" + "=" * 80)
    print("FINDING BEST MODEL CHECKPOINT")
    print("=" * 80)

    # Try wandb first (most accurate)
    best_ckpt = find_best_checkpoint_from_wandb(output_dir)

    # Fallback to log parsing
    if best_ckpt is None:
        print("Trying to find best checkpoint from training logs...")
        best_ckpt = find_best_checkpoint_from_logs(output_dir)

    if best_ckpt is None:
        print("✗ Could not determine best checkpoint")
        print("  All checkpoints saved to:", output_dir / "checkpoints")
        return False

    # Copy to best_model directory
    best_model_dir = output_dir / "checkpoints" / "best_model"
    if best_model_dir.exists():
        shutil.rmtree(best_model_dir)

    shutil.copytree(best_ckpt, best_model_dir)

    print("=" * 80)
    print("✓ BEST MODEL SAVED")
    print(f"  Source: {best_ckpt.name}")
    print(f"  Location: {best_model_dir}")
    print("=" * 80)

    return True


def main():
    """Train ACT policy and identify best checkpoint afterwards."""

    # Configuration
    # Note: Using short alias to avoid wandb tag length limit (64 chars)
    DATASET_REPO_ID = "TODO"
    DATASET_ACTUAL_PATH = "TODO"
    OUTPUT_DIR = "TODO"

    # Override sys.argv
    original_argv = sys.argv.copy()

    sys.argv = [
        sys.argv[0],

        # Policy configuration (Optimized for RTX 4070 8GB)
        "--policy.type=act",
        "--policy.vision_backbone=resnet18",
        "--policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1",
        "--policy.n_obs_steps=1",
        "--policy.chunk_size=50",
        "--policy.n_action_steps=50",
        "--policy.dim_model=128",
        "--policy.n_heads=4",
        "--policy.n_encoder_layers=4",
        "--policy.kl_weight=10.0",
        "--policy.push_to_hub=false",

        # Dataset (using actual path to bypass repo_id, but set short repo_id for wandb)
        f"--dataset.repo_id={DATASET_REPO_ID}",
        f"--dataset.root={DATASET_ACTUAL_PATH}",

        # Weights & Biases logging
        "--wandb.enable=true",
        "--wandb.project=lerobot-bimanual-act",
        "--wandb.notes=ACT 70ep best-tracking",

        # Image augmentation
        "--dataset.image_transforms.enable=true",
        "--dataset.image_transforms.max_num_transforms=2",

        # Training parameters
        "--batch_size=8",
        "--steps=300000",
        "--eval_freq=10000",
        "--save_freq=10000",
        "--log_freq=250",
        "--num_workers=8",  # Increased for 22-core CPU

        # Output directory
        f"--output_dir={OUTPUT_DIR}",

        # Seed
        "--seed=1000",
    ]

    print("=" * 80)
    print("ACT TRAINING WITH BEST MODEL TRACKING")
    print("=" * 80)
    print(f"Dataset: {DATASET_ACTUAL_PATH.name} (aliased as {DATASET_REPO_ID})")
    print("Policy: ACT (Action Chunking Transformer)")
    print("Episodes: 70 | FPS: 20")
    print("Cameras: top_view, top_left_view, close_view @ 640x480")
    print("Image Augmentation: ENABLED")
    print("-" * 80)
    print("GPU: RTX 4070 8GB - Memory Optimized")
    print("Batch Size: 8 | Model Dim: 128 | Heads: 4 | Chunk: 50")
    print("Training Steps: 300K | Expected Time: ~12-15 hours")
    print("-" * 80)
    print("FEATURES:")
    print("  ✓ Weights & Biases tracking")
    print("  ✓ Post-training best model selection")
    print("  ✓ All checkpoints saved every 10K steps")
    print("  ✓ Best model will be identified after training")
    print("=" * 80)
    print("\nStarting training...\n")

    try:
        # Run training
        train_main()

        print("\n" + "=" * 80)
        print("TRAINING COMPLETED!")
        print("=" * 80)

        # Find and copy best model
        copy_best_model(Path(OUTPUT_DIR))

        print("\nTRAINING SUMMARY:")
        print(f"  All checkpoints: {OUTPUT_DIR}/checkpoints/")
        print(f"  Best model: {OUTPUT_DIR}/checkpoints/best_model/")
        print(f"  Wandb logs: https://wandb.ai/ioannis-karampinas-ntua/lerobot-bimanual-act")
        print("=" * 80)

    finally:
        # Restore original argv
        sys.argv = original_argv


if __name__ == "__main__":
    #main()
    copy_best_model(Path("outputs/act_bimanual_70ep"))
