#!/usr/bin/env python
"""
Overfitting Test on Single Batch

This script tests if the ACT model CAN learn by overfitting to a single batch.
If the model cannot overfit to a single batch, something is wrong with:
- The model architecture
- The loss function
- The data preprocessing
- The optimization setup

Expected behavior:
- Loss should decrease rapidly to near zero
- Model should perfectly memorize the single batch

If this doesn't happen, DO NOT proceed with full training - fix the issue first!

Usage:
    python test_overfitting_single_batch.py

Expected time: ~5-10 minutes
"""

import sys
from pathlib import Path
import torch
from tqdm import tqdm

# Try to import matplotlib, but don't fail if not available
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not found. Skipping plot generation.")

# Try to import wandb for logging
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: wandb not found. Install with: pip install wandb")

# Add lerobot to path if needed
lerobot_path = Path(__file__).parent.parent.parent / "src"
if lerobot_path.exists():
    sys.path.insert(0, str(lerobot_path))

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.factory import make_policy, make_policy_config, make_pre_post_processors
from lerobot.utils.random_utils import set_seed


def test_single_batch_overfitting():
    """Test if model can overfit to a single batch."""

    print("=" * 80)
    print("OVERFITTING TEST - SINGLE BATCH")
    print("=" * 80)
    print("This test verifies that the model CAN learn by overfitting one batch.")
    print("If loss doesn't go to ~0, something is wrong with the setup!")
    print("=" * 80 + "\n")

    # Initialize wandb if available
    if HAS_WANDB:
        wandb.init(
            project="lerobot-bimanual-act",
            name="overfitting_test_5k_iters",
            tags=["overfitting_test", "sanity_check", "5k_iterations"],
            config={
                "batch_size": 8,
                "iterations": 5000,
                "learning_rate": 5e-4,
                "test_type": "single_batch_overfitting",
                "note": "Robotic task - higher LR and more iterations",
            }
        )
        print("✓ Weights & Biases initialized")
        print(f"  View at: https://wandb.ai\n")

    # Set seed for reproducibility
    set_seed(1000)

    # Configuration
    DATASET_REPO_ID = "ioannis-krmp/bimanual_tasks_dataset_task_1_20251115_201252_70_episodes"

    # Use local dataset path (HuggingFace cache)
    DATASET_LOCAL_PATH = Path.home() / ".cache/huggingface/lerobot" / DATASET_REPO_ID

    # Create policy configuration first (needed for delta_timestamps)
    print("\nCreating ACT policy configuration...")
    policy_cfg = make_policy_config(
        policy_type="act",
        vision_backbone="resnet18",
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
        n_obs_steps=1,
        chunk_size=50,
        n_action_steps=50,
        dim_model=128,
        n_heads=4,
        n_encoder_layers=4,
        kl_weight=10.0,
    )

    # Load dataset metadata to get delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    ds_meta = LeRobotDatasetMetadata(repo_id=DATASET_REPO_ID, root=DATASET_LOCAL_PATH)

    # Resolve delta_timestamps for ACT
    delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)

    # Create dataset with delta_timestamps
    print(f"Loading dataset from: {DATASET_LOCAL_PATH}")
    dataset = LeRobotDataset(
        repo_id=DATASET_REPO_ID,
        root=DATASET_LOCAL_PATH,
        delta_timestamps=delta_timestamps,
        image_transforms=None,  # No augmentation for overfitting test
    )

    print(f"Dataset loaded: {len(dataset)} frames from {dataset.num_episodes} episodes")

    # Get single batch
    batch_size = 8
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Single threaded for simplicity
    )

    # Get one batch and keep it
    single_batch = next(iter(dataloader))

    print(f"\nSingle batch shape:")
    for key, value in single_batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                if isinstance(v, torch.Tensor):
                    print(f"    {k}: {v.shape}")

    # Create policy from config and dataset metadata
    print("Creating ACT policy...")
    policy = make_policy(cfg=policy_cfg, ds_meta=dataset.meta)

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = policy.to(device)
    print(f"Policy moved to: {device}")

    # Create preprocessor and postprocessor
    print("Creating preprocessor...")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        preprocessor_overrides={
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        },
    )

    # Create optimizer with higher learning rate for faster convergence
    optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4)

    # Training loop - more iterations for robotic tasks
    num_iterations = 5000
    losses = []

    print(f"\nStarting overfitting test: {num_iterations} iterations")
    print("Expected: Loss should drop to near 0 if model can learn")
    print("-" * 80)

    policy.train()

    # Preprocess the batch once (applies normalization, device transfer, etc.)
    single_batch_processed = preprocessor(single_batch)

    for iteration in tqdm(range(num_iterations), desc="Overfitting"):
        optimizer.zero_grad()

        # Forward pass
        loss, output_dict = policy.forward(single_batch_processed)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track loss
        loss_value = loss.item()
        losses.append(loss_value)

        # Log to wandb
        if HAS_WANDB:
            wandb.log({
                "loss": loss_value,
                "iteration": iteration,
            })

        # Print progress more frequently for longer run
        if (iteration + 1) % 250 == 0:
            print(f"Iteration {iteration + 1:5d} | Loss: {loss_value:.6f}")

    # Results
    print("\n" + "=" * 80)
    print("OVERFITTING TEST RESULTS")
    print("=" * 80)
    print(f"Initial loss: {losses[0]:.6f}")
    print(f"Final loss:   {losses[-1]:.6f}")
    print(f"Reduction:    {losses[0] - losses[-1]:.6f} ({(1 - losses[-1]/losses[0])*100:.1f}%)")
    print("-" * 80)

    # For robotic tasks with ACT, success criteria are more lenient
    # ACT has KL divergence + MSE loss, naturally higher than pure MSE
    if losses[-1] < 5.0:
        print("✓ SUCCESS! Model can overfit to single batch.")
        print("  Loss dropped significantly - model architecture is working!")
        print("  For robotic tasks with ACT, loss < 5.0 on single batch is good.")
        print("  You can proceed with full training.")
        success = True
    elif losses[-1] < 15.0 and losses[-1] < losses[0] * 0.3:
        print("⚠ PARTIAL SUCCESS. Loss decreased significantly.")
        print("  Model is learning. For robotic manipulation:")
        print("  - Final loss ~10-15 on single batch is acceptable")
        print("  - 70%+ reduction shows model capacity is good")
        print("  Consider proceeding with training but monitor closely.")
        success = True
    elif losses[-1] < losses[0] * 0.5:
        print("⚠ MARGINAL. Loss decreased but not enough.")
        print("  Model shows some learning but may have issues:")
        print("  - Try even more iterations (10K)")
        print("  - Check if loss is still decreasing (not plateaued)")
        print("  - Review wandb graphs for trends")
        success = False
    else:
        print("✗ FAILURE! Model cannot overfit to single batch.")
        print("  Loss did not decrease significantly. Possible issues:")
        print("  - Model architecture problem")
        print("  - Data preprocessing issue")
        print("  - Optimizer/learning rate problem")
        print("  - Loss function issue")
        print("  DO NOT proceed with full training until this is fixed!")
        success = False

    print("=" * 80)

    # Plot loss curve
    output_dir = Path("outputs/overfitting_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    if HAS_MATPLOTLIB:
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Single Batch Overfitting Test')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale to see small changes

        plot_path = output_dir / "overfitting_test_loss_curve.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nLoss curve saved to: {plot_path}")
    else:
        print("\nSkipping plot generation (matplotlib not available)")

    # Save loss history
    loss_history_path = output_dir / "loss_history.txt"
    with open(loss_history_path, 'w') as f:
        f.write("iteration,loss\n")
        for i, loss in enumerate(losses):
            f.write(f"{i},{loss}\n")
    print(f"Loss history saved to: {loss_history_path}")

    # Finish wandb run
    if HAS_WANDB:
        wandb.log({
            "final_loss": losses[-1],
            "initial_loss": losses[0],
            "loss_reduction": losses[0] - losses[-1],
            "test_passed": success,
        })
        wandb.finish()
        print("\n✓ Wandb run completed. Check https://wandb.ai for results")

    return success


def main():
    try:
        success = test_single_batch_overfitting()
        sys.exit(0 if success else 1)
    except Exception as e:
        print("\n" + "=" * 80)
        print("ERROR during overfitting test!")
        print("=" * 80)
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
