#!/usr/bin/env python3
"""
Main training script for artistic style period classification.
"""

import os
import sys
import argparse
import torch
import random
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config.config import ProjectConfig, DataConfig, ModelConfig, TrainingConfig
from data.dataset import ArtPeriodDataset, DatasetSplitter, create_transforms
from models.efficientnet_classifier import EfficientNetClassifier
from training.trainer import Trainer
from torch.utils.data import DataLoader


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train artistic style period classifier")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Path to the dataset directory")
    parser.add_argument("--output_dir", type=str, default="./experiments/run_001",
                       help="Output directory for experiments")
    
    # Model arguments
    parser.add_argument("--backbone", type=str, default="efficientnet_b0",
                       choices=["efficientnet_b0", "efficientnet_b1"],
                       help="EfficientNet backbone to use")
    parser.add_argument("--dropout_rate", type=float, default=0.3,
                       help="Dropout rate before classifier")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Training batch size")
    parser.add_argument("--phase1_epochs", type=int, default=10,
                       help="Number of epochs for phase 1 training")
    parser.add_argument("--phase2_epochs", type=int, default=20,
                       help="Number of epochs for phase 2 training")
    parser.add_argument("--phase1_lr", type=float, default=1e-3,
                       help="Learning rate for phase 1")
    parser.add_argument("--phase2_lr", type=float, default=1e-4,
                       help="Learning rate for phase 2")
    parser.add_argument("--weight_decay", type=float, default=1e-2,
                       help="Weight decay for optimizer")
    
    # Regularization arguments
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                       help="Label smoothing factor")
    parser.add_argument("--use_mixup", action="store_true",
                       help="Use MixUp augmentation")
    parser.add_argument("--mixup_alpha", type=float, default=0.2,
                       help="MixUp alpha parameter")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of workers for data loading")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (cuda/cpu/auto)")
    
    return parser.parse_args()


def create_data_loaders(config: ProjectConfig) -> tuple:
    """Create training, validation, and test data loaders."""
    
    # Load and split data
    dataset = ArtPeriodDataset(
        data_dir=config.data.data_dir,
        art_periods=config.data.art_periods
    )
    
    splitter = DatasetSplitter()
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = \
        splitter.stratified_split(
            dataset.image_paths,
            dataset.labels,
            config.data.train_split,
            config.data.val_split,
            config.data.test_split,
            config.training.seed
        )
    
    # Create transforms
    train_transform = create_transforms(config.data.image_size, is_training=True)
    val_transform = create_transforms(config.data.image_size, is_training=False)
    
    # Create datasets
    train_dataset = ArtPeriodDataset(
        data_dir=config.data.data_dir,
        art_periods=config.data.art_periods,
        split="train",
        transform=train_transform,
        image_paths=train_paths,
        labels=train_labels
    )
    
    val_dataset = ArtPeriodDataset(
        data_dir=config.data.data_dir,
        art_periods=config.data.art_periods,
        split="val",
        transform=val_transform,
        image_paths=val_paths,
        labels=val_labels
    )
    
    test_dataset = ArtPeriodDataset(
        data_dir=config.data.data_dir,
        art_periods=config.data.art_periods,
        split="test",
        transform=val_transform,
        image_paths=test_paths,
        labels=test_labels
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create configuration
    config = ProjectConfig(
        data=DataConfig(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        ),
        model=ModelConfig(
            backbone=args.backbone,
            dropout_rate=args.dropout_rate,
            phase1_epochs=args.phase1_epochs,
            phase2_epochs=args.phase2_epochs,
            phase1_lr=args.phase1_lr,
            phase2_lr=args.phase2_lr,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            use_mixup=args.use_mixup,
            mixup_alpha=args.mixup_alpha
        ),
        training=TrainingConfig(
            device=str(device),
            seed=args.seed,
            output_dir=args.output_dir
        )
    )
    
    # Create output directories
    os.makedirs(config.training.output_dir, exist_ok=True)
    os.makedirs(os.path.join(config.training.output_dir, "checkpoints"), exist_ok=True)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(config)
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print(f"Creating {config.model.backbone} model...")
    model = EfficientNetClassifier(
        backbone=config.model.backbone,
        num_classes=config.model.num_classes,
        pretrained=True,
        dropout_rate=config.model.dropout_rate
    )
    
    # Print model info
    param_count = model.get_parameter_count()
    print(f"Model parameters: {param_count['total_parameters']:,}")
    print(f"Trainable parameters: {param_count['trainable_parameters']:,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=config.training.output_dir
    )
    
    # Training Phase 1: Head-only
    print("\n" + "="*50)
    print("PHASE 1: HEAD-ONLY TRAINING")
    print("="*50)
    
    phase1_history = trainer.train_phase1(
        num_epochs=config.model.phase1_epochs,
        learning_rate=config.model.phase1_lr,
        weight_decay=config.model.weight_decay,
        label_smoothing=config.model.label_smoothing
    )
    
    # Training Phase 2: Unfreeze last block
    print("\n" + "="*50)
    print("PHASE 2: FINE-TUNING WITH UNFROZEN LAYERS")
    print("="*50)
    
    phase2_history = trainer.train_phase2(
        num_epochs=config.model.phase2_epochs,
        learning_rate=config.model.phase2_lr,
        weight_decay=config.model.weight_decay,
        label_smoothing=config.model.label_smoothing,
        unfreeze_last_n_blocks=1
    )
    
    # Save training plots
    print("\nGenerating training plots...")
    plots_path = os.path.join(config.training.output_dir, "training_history.png")
    trainer.plot_training_history(save_path=plots_path)
    
    # Print training summary
    summary = trainer.get_training_summary()
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Best validation accuracy: {summary['best_val_accuracy']:.4f}")
    print(f"Best epoch: {summary['best_epoch']}")
    print(f"Final validation ECE: {summary['final_val_ece']:.4f}")
    print(f"Total epochs: {summary['total_epochs']}")
    print(f"Final trainable parameters: {summary['parameter_count']['trainable_parameters']:,}")
    
    print(f"\nTraining completed! Best model saved to:")
    print(f"{os.path.join(config.training.output_dir, 'checkpoints', 'best_model.pth')}")


if __name__ == "__main__":
    main()
