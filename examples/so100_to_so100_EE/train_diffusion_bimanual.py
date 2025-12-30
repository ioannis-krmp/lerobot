#!/usr/bin/env python
"""
Train Diffusion Policy on bimanual spoon feeding task.

Diffusion policies are good at:
- Multimodal action distributions
- Complex, dexterous manipulation
- Long-horizon tasks

Trade-offs vs ACT:
- Slower inference (~10 Hz vs ACT's 20 Hz)
- More training time needed
- Better at handling multiple solution paths

Usage:
    python train_diffusion_bimanual.py
"""

import sys
from pathlib import Path

# Add lerobot to path
lerobot_path = Path(__file__).parent.parent.parent / "src"
if lerobot_path.exists():
    sys.path.insert(0, str(lerobot_path))

from lerobot.scripts.lerobot_train import main as train_main


def main():
    """Train Diffusion policy for bimanual manipulation."""

    # Configuration
    DATASET_REPO_ID = "TODO"
    DATASET_ACTUAL_PATH = "TODO"
    OUTPUT_DIR = "TODO"

    original_argv = sys.argv.copy()

    sys.argv = [
        sys.argv[0],

        # Diffusion Policy Configuration
        "--policy.type=diffusion",
        "--policy.vision_backbone=resnet18",
        "--policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1",

        # Diffusion specific params
        "--policy.n_obs_steps=4",              # Use 4 frames history (0.20s @ 20Hz - captures velocity & acceleration)
        "--policy.horizon=16",                  # Predict 16 future actions
        "--policy.n_action_steps=8",           # Execute 8 actions per prediction
        "--policy.num_inference_steps=10",     # Diffusion denoising steps (10=fast, 100=slow but better)

        # Model architecture (optimized for 8GB GPU + 3 cameras)
        "--policy.down_dims=[256,512,1024]",   # Smaller than default to fit in 8GB
        "--policy.diffusion_step_embed_dim=128",
        "--policy.use_group_norm=false",       # Disable GroupNorm to use pretrained weights

        "--policy.push_to_hub=false",

        # Dataset
        f"--dataset.repo_id={DATASET_REPO_ID}",
        f"--dataset.root={DATASET_ACTUAL_PATH}",

        # Weights & Biases
        "--wandb.enable=true",
        "--wandb.project=lerobot-bimanual-diffusion",
        "--wandb.notes=Diffusion policy 70ep comparison to ACT",

        # Image augmentation
        "--dataset.image_transforms.enable=true",
        "--dataset.image_transforms.max_num_transforms=2",

        # Training parameters (Diffusion needs more steps than ACT)
        "--batch_size=8",          # Same as ACT
        "--steps=350000",          # 350K steps (~15-18 hours)
        "--eval_freq=10000",
        "--save_freq=10000",
        "--log_freq=250",
        "--num_workers=8",

        # Output
        f"--output_dir={OUTPUT_DIR}",
        "--seed=1000",
    ]

    print("=" * 80)
    print("DIFFUSION POLICY TRAINING - BIMANUAL MANIPULATION")
    print("=" * 80)
    print(f"Dataset: {DATASET_ACTUAL_PATH.name}")
    print("Policy: Diffusion (DDPM)")
    print("Episodes: 70 | FPS: 20")
    print("-" * 80)
    print("GPU: RTX 4070 8GB")
    print("Batch Size: 8 | Horizon: 16 | Inference Steps: 10")
    print("Training Steps: 400K | Expected Time: ~18-24 hours")
    print("-" * 80)
    print("NOTE: Diffusion is slower to train but may handle")
    print("      multimodal behaviors better than ACT")
    print("=" * 80)
    print("\nStarting training...\n")

    try:
        train_main()
        print("\n" + "=" * 80)
        print("DIFFUSION TRAINING COMPLETED!")
        print(f"Checkpoints: {OUTPUT_DIR}/checkpoints/")
        print("=" * 80)
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
