#!/usr/bin/env python3
"""
Comprehensive ambiguity analysis of the trained model.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Remove sys.path.append hacks; use absolute imports
from src.models.efficientnet_classifier import EfficientNetClassifier
from src.data.dataset import ArtPeriodDataset, DatasetSplitter, create_transforms, safe_collate_fn
from src.config.config import ProjectConfig
from src.ambiguity.detector import AmbiguityDetector


def run_ambiguity_analysis(
    config: ProjectConfig,
    model: torch.nn.Module, # Pass the loaded model
    val_loader: DataLoader,   # Pass the validation loader
    test_loader: DataLoader,  # Pass the test loader
    device: torch.device,
    ambiguity_output_dir: Path,
    logger=None
):
    """Comprehensive analysis of model ambiguity detection. Uses logger if provided."""
    
    art_periods = config.data.art_periods
    ambiguity_output_dir.mkdir(exist_ok=True, parents=True)
    
    def logprint(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    logprint("\n🔍 Comprehensive Ambiguity Analysis")
    logprint("=" * 50)
    
    model.eval() # Ensure model is in eval mode
    
    logprint(f"  Device for ambiguity analysis: {device}")
    
    # Initialize ambiguity detector using config values if available, else defaults
    detector = AmbiguityDetector(
        softmax_pmax_threshold=getattr(config.ambiguity, 'softmax_pmax_threshold', 0.6),
        softmax_gap_threshold=getattr(config.ambiguity, 'softmax_gap_threshold', 0.1),
        entropy_percentile_threshold=getattr(config.ambiguity, 'entropy_percentile_threshold', 80.0),
        logger=logger
    )
    
    # Get validation data for threshold computation
    val_logits_list = []
    # val_labels_list = [] # Not strictly needed for val_probs for entropy threshold
    
    logprint("\n🔄 Processing validation set for ambiguity threshold calibration...")
    with torch.no_grad():
        for images, _ in val_loader: # labels not needed here
            images = images.to(device)
            logits = model(images)
            val_logits_list.append(logits.cpu())
            # val_labels_list.append(labels.cpu())
    
    val_all_logits = torch.cat(val_logits_list, dim=0)
    # val_all_labels = torch.cat(val_labels_list, dim=0) # Not used
    val_probs = torch.softmax(val_all_logits, dim=1)
    
    # Get test data predictions and labels
    test_logits_list = []
    test_labels_list = []
    
    logprint("🔄 Processing test set for ambiguity analysis...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            test_logits_list.append(logits.cpu())
            test_labels_list.append(labels.cpu())
    
    test_all_logits = torch.cat(test_logits_list, dim=0)
    test_all_labels = torch.cat(test_labels_list, dim=0)
    
    # Comprehensive ambiguity analysis
    logprint("\n📈 Computing ambiguity statistics...")
    # Pass val_probs for entropy threshold calibration
    stats = detector.analyze_ambiguity_statistics(test_all_logits, test_all_labels, val_probs)
    
    # Print statistics
    logprint(f"\n📊 AMBIGUITY DETECTION RESULTS (Test Set):")
    logprint(f"{'='*50}")
    logprint(f"Overall Test Accuracy (from ambiguity module): {stats['overall_accuracy']:.4f}")
    logprint(f"Average Max Probability: {stats['average_max_prob']:.4f}")
    logprint(f"Average Probability Gap: {stats['average_prob_gap']:.4f}")
    logprint(f"Average Entropy: {stats['average_entropy']:.4f}")
    logprint("")
    
    logprint(f"📉 SOFTMAX-BASED DETECTION:")
    logprint(f"  Ambiguous samples: {stats['softmax_ambiguous_count']}/{len(test_all_labels)} ({stats['softmax_ambiguous_percentage']:.2f}%)")
    logprint(f"  Accuracy on ambiguous: {stats['softmax_ambiguous_accuracy']:.4f}")
    logprint("")
    
    logprint(f"📈 ENTROPY-BASED DETECTION (Threshold: {detector.entropy_threshold_value:.4f if detector.entropy_threshold_value else 'N/A'}):")
    logprint(f"  Ambiguous samples: {stats['entropy_ambiguous_count']}/{len(test_all_labels)} ({stats['entropy_ambiguous_percentage']:.2f}%)")
    logprint(f"  Accuracy on ambiguous: {stats['entropy_ambiguous_accuracy']:.4f}")
    logprint("")
    
    logprint(f"🔄 COMBINED DETECTION (Union):")
    logprint(f"  Ambiguous samples: {stats['combined_ambiguous_count']}/{len(test_all_labels)} ({stats['combined_ambiguous_percentage']:.2f}%)")
    logprint(f"  Accuracy on ambiguous: {stats['combined_ambiguous_accuracy']:.4f}")
    logprint("")
    
    logprint(f"🔗 OVERLAP ANALYSIS:")
    logprint(f"  Softmax ∩ Entropy: {stats['softmax_entropy_overlap']}/{len(test_all_labels)} ({stats['softmax_entropy_overlap_percentage']:.2f}%)")
    
    # Get detailed ambiguity masks for visualization
    # Re-run with val_probs to ensure entropy threshold is set if not already
    _, softmax_metrics = detector.get_softmax_ambiguity(test_all_logits)
    _, entropy_metrics = detector.get_entropy_ambiguity(test_all_logits, val_probs) # Ensure threshold is computed
    
    # Create visualizations
    logprint(f"\n🎨 Creating ambiguity visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14)) # Increased size slightly
    
    probs = torch.softmax(test_all_logits, dim=1)
    max_probs, _ = torch.max(probs, dim=1)
    entropy = entropy_metrics['entropy'] # Use calculated entropy
    predictions = torch.argmax(test_all_logits, dim=1)
    correct = (predictions == test_all_labels)
    
    # Scatter plot: Confidence vs Entropy
    ax = axes[0, 0]
    # Define colors based on correctness for the scatter plot
    scatter_colors = ['green' if c else 'red' for c in correct]
    ax.scatter(max_probs.numpy(), entropy.numpy(), c=scatter_colors, alpha=0.5, s=15) # smaller points
    ax.set_xlabel('Max Probability (Confidence)')
    ax.set_ylabel('Entropy')
    ax.set_title('Confidence vs Entropy (Test Set)\\n(Red=Incorrect, Green=Correct)')
    ax.axvline(x=detector.softmax_pmax_threshold, color='blue', linestyle='--', alpha=0.7, label=f'PMax Thr: {detector.softmax_pmax_threshold}')
    if detector.entropy_threshold_value is not None:
        ax.axhline(y=detector.entropy_threshold_value, color='orange', linestyle='--', alpha=0.7, label=f'Entropy Thr: {detector.entropy_threshold_value:.2f}')
    ax.legend(fontsize='small')
    ax.grid(True, alpha=0.3)
    
    # Histogram: Confidence distribution
    ax = axes[0, 1]
    ax.hist(max_probs[correct].numpy(), alpha=0.7, label='Correct', bins=30, color='green', density=True)
    ax.hist(max_probs[~correct].numpy(), alpha=0.7, label='Incorrect', bins=30, color='red', density=True)
    ax.axvline(x=detector.softmax_pmax_threshold, color='blue', linestyle='--', alpha=0.7, label=f'PMax Thr: {detector.softmax_pmax_threshold}')
    ax.set_xlabel('Max Probability (Confidence)')
    ax.set_ylabel('Density')
    ax.set_title('Confidence Distribution (Test Set)')
    ax.legend(fontsize='small')
    ax.grid(True, alpha=0.3)
    
    # Histogram: Entropy distribution
    ax = axes[1, 0]
    ax.hist(entropy[correct].numpy(), alpha=0.7, label='Correct', bins=30, color='green', density=True)
    ax.hist(entropy[~correct].numpy(), alpha=0.7, label='Incorrect', bins=30, color='red', density=True)
    if detector.entropy_threshold_value is not None:
        ax.axvline(x=detector.entropy_threshold_value, color='orange', linestyle='--', alpha=0.7, label=f'Entropy Thr: {detector.entropy_threshold_value:.2f}')
    ax.set_xlabel('Entropy')
    ax.set_ylabel('Density')
    ax.set_title('Entropy Distribution (Test Set)')
    ax.legend(fontsize='small')
    ax.grid(True, alpha=0.3)
    
    # Ambiguity detection comparison
    ax = axes[1, 1]
    detection_counts = [
        stats['softmax_ambiguous_count'],
        stats['entropy_ambiguous_count'], 
        stats['combined_ambiguous_count'],
        stats['softmax_entropy_overlap']
    ]
    detection_labels = ['Softmax', 'Entropy', 'Combined (Union)', 'Overlap (S ∩ E)']
    bars = ax.bar(detection_labels, detection_counts, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    ax.set_ylabel('Number of Ambiguous Samples')
    ax.set_title('Ambiguity Detection Comparison (Test Set)')
    ax.grid(True, alpha=0.2, axis='y')
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right") # Rotate labels slightly
    
    for bar_obj in bars: # Corrected variable name from bar to bar_obj
        height = bar_obj.get_height()
        ax.text(bar_obj.get_x() + bar_obj.get_width()/2., height,
                f'{height}', ha='center', va='bottom', fontsize='small')
    
    plt.tight_layout(pad=2.0)
    plot_save_path = ambiguity_output_dir / 'ambiguity_analysis_plots.png'
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    logprint(f"Ambiguity analysis plots saved to: {plot_save_path}")
    plt.close()
    
    # Per-class ambiguity analysis
    logprint("\\n📋 Computing per-class ambiguity statistics...")
    per_class_stats_dict = {} # Renamed from per_class_stats to avoid conflict
    for class_idx, class_name in enumerate(art_periods):
        class_mask = test_all_labels == class_idx
        if class_mask.sum() > 0:
            class_logits_subset = test_all_logits[class_mask]
            class_labels_subset = test_all_labels[class_mask]
            
            # For per-class, we don't re-calibrate entropy threshold, use the global one
            class_softmax_mask, _ = detector.get_softmax_ambiguity(class_logits_subset)
            class_entropy_mask, _ = detector.get_entropy_ambiguity(class_logits_subset, validation_probs=None) # Use existing threshold
            class_combined_mask, _ = detector.get_combined_ambiguity(class_logits_subset, validation_probs=None, combine_mode='union')
            
            class_probs_subset = torch.softmax(class_logits_subset, dim=1)
            class_max_probs_subset, _ = torch.max(class_probs_subset, dim=1)
            class_entropy_subset = -(class_probs_subset * torch.log(class_probs_subset + 1e-9)).sum(dim=1)
            class_predictions_subset = torch.argmax(class_logits_subset, dim=1)
            class_correct_subset = (class_predictions_subset == class_labels_subset)
            
            per_class_stats_dict[class_name] = {
                'total_samples': class_mask.sum().item(),
                'accuracy': class_correct_subset.float().mean().item() if class_correct_subset.numel() > 0 else 0.0,
                'avg_confidence': class_max_probs_subset.mean().item() if class_max_probs_subset.numel() > 0 else 0.0,
                'avg_entropy': class_entropy_subset.mean().item() if class_entropy_subset.numel() > 0 else 0.0,
                'softmax_ambiguous': class_softmax_mask.sum().item(),
                'entropy_ambiguous': class_entropy_mask.sum().item(),
                'combined_ambiguous': class_combined_mask.sum().item(),
            }
        else:
            per_class_stats_dict[class_name] = {
                'total_samples': 0, 'accuracy': 0, 'avg_confidence': 0, 'avg_entropy': 0,
                'softmax_ambiguous': 0, 'entropy_ambiguous': 0, 'combined_ambiguous': 0,
            }

    df_per_class = pd.DataFrame(per_class_stats_dict).T
    # Calculate percentages safely
    df_per_class['softmax_ambiguous_pct'] = df_per_class.apply(
        lambda row: (row['softmax_ambiguous'] / row['total_samples'] * 100) if row['total_samples'] > 0 else 0, axis=1
    ).round(1)
    df_per_class['entropy_ambiguous_pct'] = df_per_class.apply(
        lambda row: (row['entropy_ambiguous'] / row['total_samples'] * 100) if row['total_samples'] > 0 else 0, axis=1
    ).round(1)
    df_per_class['combined_ambiguous_pct'] = df_per_class.apply(
        lambda row: (row['combined_ambiguous'] / row['total_samples'] * 100) if row['total_samples'] > 0 else 0, axis=1
    ).round(1)
    
    logprint(f"\\n📋 PER-CLASS AMBIGUITY ANALYSIS (Test Set):")
    logprint(f"{'='*80}")
    # Select columns for printing to make it more readable
    cols_to_print = ['total_samples', 'accuracy', 'avg_confidence', 'avg_entropy', 
                     'softmax_ambiguous_pct', 'entropy_ambiguous_pct', 'combined_ambiguous_pct']
    logprint(df_per_class[cols_to_print].round(3).to_string())
    
    # Save results
    per_class_csv_path = ambiguity_output_dir / 'per_class_ambiguity_stats.csv'
    df_per_class.to_csv(per_class_csv_path)
    logprint(f"Per-class ambiguity stats saved to: {per_class_csv_path}")
    
    overall_stats_csv_path = ambiguity_output_dir / 'overall_ambiguity_stats.csv'
    stats_df = pd.DataFrame([stats]) # stats is the overall dictionary
    stats_df.to_csv(overall_stats_csv_path, index=False)
    logprint(f"Overall ambiguity stats saved to: {overall_stats_csv_path}")
    
    logprint(f"\\n✅ Ambiguity analysis complete! Results saved to: {ambiguity_output_dir}")
    
    return stats, df_per_class


# if __name__ == "__main__":
#     # This part is for standalone testing and needs to be adapted or removed
#     # For now, we'll comment it out as it's meant to be called from train.py
#     # print("Running analyze_ambiguity.py as a standalone script (for testing purposes).")
#     # # Minimal config for testing - replace with actual loading if needed for standalone
#     class MockDataConfig:
#         art_periods = ['Renaissance', 'Baroque', 'Romanticism', 'Impressionism', 
#                        'Post-Impressionism', 'Modernism', 'Surrealism', 'Contemporary']
#         # Add other necessary fields if your functions use them
#         data_dir = './data/wikiart_real_processed' # Example
#         image_size = 224
#         batch_size = 32
#         num_workers = 0 # Simpler for local test
#         train_split = 0.01 # Tiny for fast test
#         val_split = 0.01
#         test_split = 0.01


#     class MockModelConfig:
#         backbone = "efficientnet_b0"
#         num_classes = len(MockDataConfig.art_periods)
#         dropout_rate = 0.3
#         pretrained = True

#     class MockTrainingConfig:
#         seed = 42
#         # ... other params if needed by create_data_loaders or model
    
#     class MockAmbiguityConfig:
#         softmax_pmax_threshold = 0.6
#         softmax_gap_threshold = 0.1
#         entropy_percentile_threshold = 80.0

#     class MockProjectConfig:
#         def __init__(self):
#             self.data = MockDataConfig()
#             self.model = MockModelConfig()
#             self.training = MockTrainingConfig()
#             self.ambiguity = MockAmbiguityConfig()
#             self.experiment_name = "test_ambiguity_standalone"
#             self.output_dir = Path("./experiments/test_ambiguity_standalone")
#             self.device = "cpu"


#     mock_config = MockProjectConfig()
#     mock_config.output_dir.mkdir(exist_ok=True, parents=True)
#     amb_output_dir = mock_config.output_dir / "ambiguity_analysis"

#     # Create dummy data loaders and model for testing
#     # This requires more setup (dataset, splits etc.)
#     # For a quick test, one might need to load an actual model and data
#     # Or, ensure the function is robust to minimal inputs if that's a valid use case.
#     print("Standalone test execution would require loading a model and data.")
#     print("Please test by integrating with train.py or by setting up a full test environment here.")

#     # Example of how it might be called (needs actual model and loaders)
#     # from data.dataset import ArtPeriodDataset, DatasetSplitter, create_transforms
#     # from torch.utils.data import DataLoader, Subset
#     # from models.efficientnet_classifier import EfficientNetClassifier

#     # # Setup device
#     # device = torch.device(mock_config.device)
#     # set_seed(mock_config.training.seed) # If you have a set_seed function

#     # # Create DataLoaders (simplified)
#     # # This is a placeholder - you'd need your actual data loading logic
#     # # For example, using create_data_loaders from train.py if it's importable
#     # print("Creating dummy data loaders for standalone test...")
#     # try:
#     #     # This is a rough sketch and might need adjustments based on your create_data_loaders
#     #     # Ensure ArtPeriodDataset can be initialized with minimal paths for a test
#     #     full_dataset = ArtPeriodDataset(data_dir=mock_config.data.data_dir, art_periods=mock_config.data.art_periods)
#     #     splitter = DatasetSplitter()
#     #     (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = splitter.stratified_split(
#     #         full_dataset.image_paths, full_dataset.labels, 0.01, 0.01, 0.98 # Minimal train/val
#     #     ) # Using most for test
        
#     #     val_transform = create_transforms(mock_config.data.image_size, is_training=False)
#     #     test_transform = create_transforms(mock_config.data.image_size, is_training=False)

#     #     val_dataset = ArtPeriodDataset(data_dir=mock_config.data.data_dir, art_periods=mock_config.data.art_periods, split="val", transform=val_transform, image_paths=val_paths, labels=val_labels)
#     #     test_dataset = ArtPeriodDataset(data_dir=mock_config.data.data_dir, art_periods=mock_config.data.art_periods, split="test", transform=test_transform, image_paths=test_paths, labels=test_labels)
        
#     #     val_loader = DataLoader(val_dataset, batch_size=mock_config.data.batch_size, num_workers=0, collate_fn=safe_collate_fn)
#     #     test_loader = DataLoader(
#     #         test_dataset, batch_size=mock_config.data.batch_size, num_workers=0, collate_fn=safe_collate_fn)
#     # except Exception as e:
#     #     print(f"Could not create dummy data loaders: {e}")
#     #     val_loader, test_loader = None, None


#     # # Load or create a model
#     # print("Creating dummy model for standalone test...")
#     # test_model = EfficientNetClassifier(backbone=mock_config.model.backbone, num_classes=mock_config.model.num_classes).to(device)
#     # # In a real test, you'd load a checkpoint:
#     # # model_path = "./experiments/your_run/checkpoints/best_model.pth"
#     # # checkpoint = torch.load(model_path, map_location=device)
#     # # test_model.load_state_dict(checkpoint['model_state_dict'])
#     # test_model.eval()

#     # if test_model and val_loader and test_loader:
#     #     print("Running ambiguity analysis with dummy data...")
#     #     run_ambiguity_analysis(mock_config, test_model, val_loader, test_loader, device, amb_output_dir)
#     # else:
#     #     print("Skipping run_ambiguity_analysis due to missing model or data loaders.")
# pass # End of if __name__ == "__main__"
