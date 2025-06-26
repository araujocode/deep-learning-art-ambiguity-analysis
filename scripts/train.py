#!/usr/bin/env python3
"""
Main training script for artistic style period classification.
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F 
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
from torchvision import transforms
from sklearn.utils.class_weight import compute_class_weight
from torchvision.transforms import ToPILImage
import pandas as pd
import logging

here = Path(__file__).resolve().parent
# Add project root to path for absolute imports
sys.path.append(str(here.parent))

from src.config.config import ProjectConfig, DataConfig, ModelConfig, TrainingConfig, AmbiguityConfig 
from src.data.dataset import ArtPeriodDataset, DatasetSplitter, create_transforms, create_tta_transforms, safe_collate_fn
from src.models.efficientnet_classifier import EfficientNetClassifier
from src.training.trainer import Trainer
from torch.utils.data import DataLoader
from src.visualization.gradcam import GradCAMVisualizer, preprocess_image_for_gradcam
from analyze_training import analyze_training_dynamics 
from analyze_ambiguity import run_ambiguity_analysis 

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
                   choices=["efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3", "efficientnet_b4"],
                   help="EfficientNet backbone to use")
    parser.add_argument("--dropout_rate", type=float, default=0.3,
                       help="Dropout rate before classifier")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Training batch size")
    parser.add_argument("--phase1_epochs", type=int, default=10,
                       help="Number of epochs for phase 1 training")
    parser.add_argument("--phase2_epochs", type=int, default=40, # Changed from 20 to 40
                       help="Number of epochs for phase 2 training")
    parser.add_argument("--phase1_lr", type=float, default=1e-3,
                       help="Learning rate for phase 1")
    parser.add_argument("--phase2_lr", type=float, default=1e-4,
                       help="Learning rate for phase 2")
    parser.add_argument("--weight_decay", type=float, default=1e-2,
                       help="Weight decay for optimizer")
    parser.add_argument("--early_stopping_patience_phase1", type=int, default=5,
                       help="Early stopping patience for phase 1 (head training)")
    parser.add_argument("--early_stopping_patience_phase2", type=int, default=16,
                       help="Early stopping patience for phase 2 (fine-tuning)")
    
    # Regularization arguments
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                       help="Label smoothing factor")
    parser.add_argument("--use_mixup", action="store_true",
                       help="Use MixUp augmentation")
    parser.add_argument("--mixup_alpha", type=float, default=0.2,
                       help="MixUp alpha parameter")
    parser.add_argument("--use_cutmix", action="store_true",
                       help="Use CutMix augmentation")
    parser.add_argument("--cutmix_alpha", type=float, default=1.0,
                       help="CutMix alpha parameter")
    parser.add_argument("--use_focal_loss", action="store_true",
                       help="Use Focal Loss instead of CrossEntropyLoss")
    parser.add_argument("--lr_scheduler", type=str, default="onecycle", choices=["onecycle", "cosine", "plateau"],
                       help="Learning rate scheduler type")
    parser.add_argument("--ambiguity_scores_path", type=str, default=None,
                       help="Path to ambiguity scores file (npy, pt, or csv)")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of workers for data loading")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (cuda/cpu/auto)")
    parser.add_argument("--image_size", type=int, default=224, help="Input image size (e.g., 224, 384)")
    
    # Ambiguity Analysis arguments
    parser.add_argument("--run_ambiguity_analysis", action="store_true",
                       help="Run ambiguity analysis after training.")
    parser.add_argument("--softmax_pmax_threshold", type=float, default=0.6,
                       help="Ambiguity: Softmax p_max threshold.")
    parser.add_argument("--softmax_gap_threshold", type=float, default=0.1,
                       help="Ambiguity: Softmax probability gap threshold.")
    parser.add_argument("--entropy_percentile_threshold", type=float, default=80.0,
                       help="Ambiguity: Entropy percentile threshold for calibration.")
    
    # Augmentation arguments
    parser.add_argument("--use_randaugment", action="store_true", help="Use RandAugment for training augmentations")
    parser.add_argument("--use_autoaugment", action="store_true", help="Use AutoAugment for training augmentations")
    parser.add_argument("--use_random_erasing", action="store_true", help="Use Random Erasing for training augmentations")
    
    # New arguments
    parser.add_argument("--progressive_unfreezing", action="store_true", help="Use progressive unfreezing schedule in phase 2 (recommended)")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic training (slower, for debugging)")
    parser.add_argument("--unfreeze_last_n_blocks", type=int, default=2, help="Number of last blocks to unfreeze in phase 2 (non-progressive)")
    parser.add_argument("--gradcam_layer", type=str, default="blocks[-1][-1].conv_dw",
                       help="Target layer for Grad-CAM visualization (e.g., 'blocks[-1][-1].conv_dw', 'backbone.blocks[-1]', etc.)")
    
    return parser.parse_args()


def setup_project_logger(output_dir: str) -> logging.Logger:
    """Setup the unique project logger, robust to duplicate handlers."""
    logger = logging.getLogger('my_project.trainer')
    logger.setLevel(logging.INFO)
    # Check for both file and console handlers
    has_file_handler = any(isinstance(h, logging.FileHandler) for h in logger.handlers)
    has_console_handler = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    if not has_file_handler:
        log_file = os.path.join(output_dir, 'training.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    if not has_console_handler:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger

def create_data_loaders(config: ProjectConfig, train_transform, val_transform) -> tuple:
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
        pin_memory=True,
        collate_fn=safe_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
        collate_fn=safe_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
        collate_fn=safe_collate_fn
    )
    
    return train_loader, val_loader, test_loader


def evaluate_model_after_training(config: ProjectConfig, model_path: str, test_loader: DataLoader, device: torch.device, art_periods: list):
    """Evaluate the trained model on test data and save metrics."""
    print("\n" + "="*50)
    print("POST-TRAINING EVALUATION ON TEST SET")
    print("="*50)

    # Load trained model
    print(f"Loading best model from: {model_path}")
    model = EfficientNetClassifier(
        backbone=config.model.backbone,
        num_classes=len(art_periods),
        dropout_rate=config.model.dropout_rate
    )
    checkpoint = torch.load(model_path, map_location=device)
    # Unified checkpoint loading: always expect 'state_dict' (no legacy fallback)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise RuntimeError("Checkpoint missing model weights (expected 'state_dict')")
    model.to(device)
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'N/A')}")
    metrics = checkpoint.get('metrics', {})
    if 'best_val_acc' in metrics:
        print(f"Original best validation accuracy during training: {metrics['best_val_acc']:.4f}")
    elif 'best_val_acc' in checkpoint: # For older checkpoints
        print(f"Original best validation accuracy during training: {checkpoint['best_val_acc']:.4f}")


    all_predictions = []
    all_labels = []
    all_confidences = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating on Test Set"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            confidences = torch.max(probabilities, dim=1)[0]
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())

    test_accuracy = accuracy_score(all_labels, all_predictions)
    print(f"\n🎯 TEST SET RESULTS:")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Average Confidence: {np.mean(all_confidences):.4f}")

    # Classification report
    report_text = classification_report(
        all_labels, all_predictions, 
        target_names=art_periods, 
        digits=4
    )
    print("\n📊 DETAILED CLASSIFICATION REPORT (Test Set):")
    print("--------------------------------------------------")
    print(report_text)
    
    report_path = os.path.join(config.training.output_dir, "test_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)\n")
        f.write(f"Average Confidence: {np.mean(all_confidences):.4f}\n\n")
        f.write(report_text)
    print(f"Classification report saved to: {report_path}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(12, 10)) # Adjusted figure size
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=art_periods, yticklabels=art_periods)
    plt.title(f'Test Set Confusion Matrix - Accuracy: {test_accuracy:.2%}', fontsize=14) # Added fontsize
    plt.xlabel('Predicted Label', fontsize=12) # Added fontsize
    plt.ylabel('True Label', fontsize=12) # Added fontsize
    plt.xticks(rotation=45, ha='right', fontsize=10) # Added fontsize
    plt.yticks(rotation=0, fontsize=10) # Added fontsize
    plt.tight_layout(pad=2.0) # Adjusted padding

    cm_path = os.path.join(config.training.output_dir, "test_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {cm_path}")
    # plt.show() # Optionally show, but usually not needed in automated scripts
    plt.close()

    return model # Return the loaded model for potential reuse


def generate_gradcam_examples(config: ProjectConfig, model: torch.nn.Module, test_loader: DataLoader, device: torch.device, art_periods: list, num_examples_per_class: int = 1, gradcam_layer: str = None):
    """Generate Grad-CAM visualizations for a few examples from the test set.
    Accepts a loaded model directly.
    """
    print("\n" + "="*50)
    print("GENERATING GRAD-CAM EXAMPLES")
    print("="*50)

    model.eval() # Ensure the passed model is in eval mode

    # Unified: use CLI arg if set, else config.ambiguity.target_layer_name
    target_layer_name = gradcam_layer or getattr(config.ambiguity, 'target_layer_name', None)
    if not target_layer_name:
        target_layer_name = "blocks[-1][-1].conv_dw"  # Final fallback
    try:
        gradcam_vis = GradCAMVisualizer(model, target_layer_name=target_layer_name)
    except Exception as e:
        print(f"Error initializing GradCAMVisualizer with layer '{target_layer_name}': {e}. Skipping Grad-CAM generation.")
        return

    print(f"Grad-CAM initialized with target layer: {gradcam_vis.target_layer_name}")

    # Get a few images from the test_loader
    # We need original image paths to load them without normalization for visualization
    # The test_loader.dataset should be an ArtPeriodDataset or a Subset of it.
    
    # Get the underlying dataset if test_loader.dataset is a Subset
    actual_dataset = test_loader.dataset
    if isinstance(actual_dataset, torch.utils.data.Subset):
        actual_dataset = actual_dataset.dataset

    # Ensure the actual_dataset has image_paths and labels attributes
    if not hasattr(actual_dataset, 'image_paths') or not hasattr(actual_dataset, 'labels'):
        print("Error: Test dataset does not have 'image_paths' or 'labels' attributes. Skipping Grad-CAM.")
        return

    # Use config.data.image_size for Grad-CAM transform
    gradcam_transform = create_transforms(config.data.image_size, is_training=False)

    output_gradcam_dir = os.path.join(config.training.output_dir, "gradcam_examples")
    os.makedirs(output_gradcam_dir, exist_ok=True)

    images_processed_per_class = {i: 0 for i in range(len(art_periods))}
    images_shown_count = 0

    for i in range(len(actual_dataset)):
        if images_shown_count >= len(art_periods) * num_examples_per_class:
            break # Stop if we have enough examples overall

        image_path = actual_dataset.image_paths[i]
        label_idx = actual_dataset.labels[i]

        if images_processed_per_class[label_idx] < num_examples_per_class:
            try:
                print(f"Processing Grad-CAM for: {image_path} (Class: {art_periods[label_idx]})")
                # Preprocess image for Grad-CAM (gets tensor for model, and original for display)
                input_tensor, original_image_np = preprocess_image_for_gradcam(image_path, gradcam_transform)
                input_tensor = input_tensor.to(device)

                with torch.no_grad():
                    logits = model(input_tensor)
                
                # Sanitize filename
                base_filename = os.path.basename(image_path)
                safe_filename = "".join(c if c.isalnum() or c in ('.', '_') else '_' for c in base_filename)
                save_path = os.path.join(output_gradcam_dir, f"{art_periods[label_idx]}_{safe_filename}_gradcam.png")

                gradcam_vis.visualize_predictions(
                    input_tensor=input_tensor,
                    original_image=original_image_np,
                    logits=logits,
                    class_names=art_periods,
                    top_k=3,
                    save_path=save_path
                )
                print(f"Grad-CAM saved to {save_path}")
                images_processed_per_class[label_idx] += 1
                images_shown_count += 1
            except Exception as e:
                print(f"Error generating Grad-CAM for {image_path}: {e}")
        
    print(f"Grad-CAM example generation complete. Images saved in {output_gradcam_dir}")


def tta_evaluate(model, dataset, device, art_periods, n_tta=5, batch_size=32):
    """
    Run Test-Time Augmentation (TTA) evaluation on a dataset (vectorized, fast).
    Args:
        model: Trained model.
        dataset: ArtPeriodDataset (test set).
        device: torch.device.
        art_periods: List of class names.
        n_tta: Number of TTA augmentations per image.
        batch_size: Batch size for evaluation.
    Returns:
        TTA accuracy (float)
    """
    model.eval()
    # Use dataset.image_size if available, else default to 224
    image_size = getattr(dataset, 'image_size', 224)
    if hasattr(dataset, 'transform') and hasattr(dataset.transform, 'transforms'):
        for t in dataset.transform.transforms:
            if hasattr(t, 'size'):
                image_size = t.size if isinstance(t.size, int) else t.size[0]
                break
    tta_transforms = create_tta_transforms(image_size, n=n_tta)
    all_labels = []
    all_probs = []
    to_pil = ToPILImage()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=safe_collate_fn)
    with torch.no_grad():
        for imgs, labels, _ in tqdm(loader, desc="TTA Evaluation"):  # imgs: (B, C, H, W)
            if imgs.numel() == 0:
                continue  # Skip empty batches
            B = imgs.size(0)
            # Vectorized: for each TTA, apply to all images in batch, then stack
            augmented_batches = []
            for t in tta_transforms:
                batch_aug = torch.stack([t(to_pil(img.cpu())) for img in imgs])
                augmented_batches.append(batch_aug)
            aug_imgs = torch.cat(augmented_batches, dim=0).to(device)
            outputs = model(aug_imgs)
            num_classes = outputs.size(1)
            outputs = outputs.view(n_tta, B, num_classes).permute(1, 0, 2)
            probs = torch.softmax(outputs, dim=2)
            avg_probs = probs.mean(dim=1)
            all_probs.append(avg_probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    all_probs = np.concatenate(all_probs, axis=0)
    preds = np.argmax(all_probs, axis=1)
    acc = (preds == np.array(all_labels)).mean()
    print(f"\n🎯 TTA TEST SET ACCURACY: {acc:.4f} ({acc*100:.2f}%)")
    return acc


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
    if not args.deterministic:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    
    # Setup logger
    logger = setup_project_logger(args.output_dir)
    
    # Create configuration
    config = ProjectConfig(
        data=DataConfig(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,  # Pass CLI image_size
            # Optionally add other fields
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
            mixup_alpha=args.mixup_alpha,
            use_cutmix=args.use_cutmix,
            cutmix_alpha=args.cutmix_alpha,
            use_focal_loss=args.use_focal_loss,
            unfreeze_last_n_blocks=args.unfreeze_last_n_blocks,
            lr_scheduler=args.lr_scheduler
        ),
        training=TrainingConfig(
            device=str(device),
            seed=args.seed,
            output_dir=args.output_dir
        ),
        ambiguity=AmbiguityConfig(
            run_analysis=args.run_ambiguity_analysis,
            softmax_pmax_threshold=args.softmax_pmax_threshold,
            softmax_gap_threshold=args.softmax_gap_threshold,
            entropy_percentile_threshold=args.entropy_percentile_threshold
        )
    )
    
    # Create transforms with CLI/config toggles
    train_transform = create_transforms(
        image_size=config.data.image_size,  # Use config.data.image_size
        is_training=True,
        use_randaugment=args.use_randaugment,
        use_autoaugment=args.use_autoaugment,
        use_random_erasing=args.use_random_erasing
    )
    val_transform = create_transforms(
        image_size=config.data.image_size,  # Use config.data.image_size
        is_training=False
    )
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(config, train_transform, val_transform)
    
    # Compute class weights from training labels
    train_labels = []
    for _, labels, _ in train_loader:
        train_labels.extend(labels.cpu().numpy())
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print(f"Creating {config.model.backbone} model...")
    model = EfficientNetClassifier(
        backbone=args.backbone,
        num_classes=len(config.data.art_periods),
        dropout_rate=args.dropout_rate
    )
    
    # Print model info
    param_count = model.get_parameter_count()
    print(f"Model parameters: {param_count['total_parameters']:,}")
    print(f"Trainable parameters: {param_count['trainable_parameters']:,}")
    
    # Load ambiguity scores if provided
    ambiguity_weights = None
    if args.ambiguity_scores_path is not None:
        
        path = args.ambiguity_scores_path
        if path.endswith('.npy'):
            ambiguity_weights = torch.from_numpy(np.load(path)).float()
        elif path.endswith('.pt'):
            ambiguity_weights = torch.load(path)
        elif path.endswith('.csv'):
            df = pd.read_csv(path)
            # Assume a column 'ambiguity_score' or use the last column
            if 'ambiguity_score' in df.columns:
                ambiguity_weights = torch.from_numpy(df['ambiguity_score'].values).float()
            else:
                ambiguity_weights = torch.from_numpy(df.iloc[:, -1].values).float()
        else:
            print(f"Unknown ambiguity score file format: {path}")
            ambiguity_weights = None
        if ambiguity_weights is not None:
            print(f"Loaded ambiguity weights from {path} (shape: {ambiguity_weights.shape})")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=args.output_dir,
        logger=logger,
        use_mixup=args.use_mixup,
        mixup_alpha=args.mixup_alpha,
        use_cutmix=args.use_cutmix,
        cutmix_alpha=args.cutmix_alpha,
        use_focal_loss=args.use_focal_loss,
        ambiguity_weights=ambiguity_weights
    )
    # Set class weights in trainer
    trainer.set_class_weights(class_weights)
    
    # Training Phase 1: Head-only
    print("\n" + "="*50)
    print("PHASE 1: HEAD-ONLY TRAINING")
    print("="*50)
    
    phase1_history = trainer.train_phase1(
        num_epochs=config.model.phase1_epochs,
        learning_rate=config.model.phase1_lr,
        weight_decay=config.model.weight_decay,
        label_smoothing=config.model.label_smoothing,
        lr_scheduler=config.model.lr_scheduler,
        early_stopping_patience=args.early_stopping_patience_phase1
    )
    
    # Training Phase 2: Unfreeze last block or progressive unfreezing
    print("\n" + "="*50)
    print("PHASE 2: FINE-TUNING WITH UNFROZEN LAYERS")
    print("="*50)

    if args.progressive_unfreezing:
        phase2_history = trainer.train_phase2_staged(
            total_epochs=config.model.stage1_epochs + config.model.stage2_epochs + config.model.stage3_epochs,
            stage1_epochs=config.model.stage1_epochs,
            stage2_epochs=config.model.stage2_epochs,
            stage3_epochs=config.model.stage3_epochs,
            learning_rate=config.model.phase2_lr,
            weight_decay=config.model.weight_decay,
            label_smoothing=config.model.label_smoothing,
            early_stopping_patience=args.early_stopping_patience_phase2,
            lr_scheduler=config.model.lr_scheduler
        )
    else:
        phase2_history = trainer.train_phase2(
            num_epochs=config.model.phase2_epochs,
            learning_rate=config.model.phase2_lr,
            weight_decay=config.model.weight_decay,
            label_smoothing=config.model.label_smoothing,
            unfreeze_last_n_blocks=config.model.unfreeze_last_n_blocks,
            early_stopping_patience=args.early_stopping_patience_phase2,
            lr_scheduler=config.model.lr_scheduler
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
    
    print("\nTraining completed! Best model saved to:")
    best_model_path = os.path.join(config.training.output_dir, 'checkpoints', 'best_model.pth')
    print(best_model_path)

    # Evaluate model on test set after training
    # The model is returned by evaluate_model_after_training
    evaluated_model = evaluate_model_after_training(config, best_model_path, test_loader, device, config.data.art_periods)

    # TTA evaluation on test set
    if evaluated_model:
        print("\n" + "="*50)
        print("RUNNING TEST-TIME AUGMENTATION (TTA) ON TEST SET")
        print("="*50)
        tta_acc = tta_evaluate(evaluated_model, test_loader.dataset, device, config.data.art_periods, n_tta=5, batch_size=config.data.batch_size)
    else:
        print("Skipping TTA evaluation as model could not be loaded/evaluated.")

    # Generate Grad-CAM examples using the already loaded model
    if evaluated_model: # Ensure model was loaded successfully
        generate_gradcam_examples(config, evaluated_model, test_loader, device, config.data.art_periods, num_examples_per_class=1, gradcam_layer=args.gradcam_layer)
    else:
        print("Skipping Grad-CAM generation as model could not be loaded/evaluated.")

    # Analyze training dynamics
    analyze_training_dynamics(experiment_dir=config.training.output_dir, logger=logger)

    # Run Ambiguity Analysis if flagged
    if config.ambiguity.run_analysis:
        if evaluated_model: # Reuse the loaded model
            print("\n" + "="*50)
            print("RUNNING AMBIGUITY ANALYSIS")
            print("="*50)
            ambiguity_output_dir = Path(config.training.output_dir) / "ambiguity_analysis"
            run_ambiguity_analysis(
                config=config, # Pass the full config
                model=evaluated_model,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                ambiguity_output_dir=ambiguity_output_dir,
                logger=logger
            )
        else:
            print("Skipping ambiguity analysis as the model was not available from evaluation.")


if __name__ == "__main__":
    main()
