#!/usr/bin/env python3
"""
Comprehensive ambiguity analysis of the trained model.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import torch.nn.functional as F
from models.efficientnet_classifier import EfficientNetClassifier
from data.dataset import ArtPeriodDataset, DatasetSplitter, create_transforms
from ambiguity.detector import AmbiguityDetector
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

def analyze_model_ambiguity():
    """Comprehensive analysis of model ambiguity detection."""
    
    # Configuration
    art_periods = ['Renaissance', 'Baroque', 'Romanticism', 'Impressionism', 
                   'Post-Impressionism', 'Modernism', 'Surrealism', 'Contemporary']
    data_dir = './data/wikiart_real_processed'  # Updated path
    model_path = './experiments/run_001/checkpoints/best_model.pth'
    output_dir = Path('./experiments/run_001/ambiguity_analysis')
    output_dir.mkdir(exist_ok=True)
    
    print("🔍 Comprehensive Ambiguity Analysis")
    print("=" * 50)
    
    # Load dataset
    dataset = ArtPeriodDataset(data_dir, art_periods)
    
    # Create train/val/test splits
    splitter = DatasetSplitter()
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = \
        splitter.stratified_split(
            dataset.image_paths, dataset.labels, 
            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )
    
    # Create datasets
    transform = create_transforms(is_training=False)
    
    val_indices = [i for i, path in enumerate(dataset.image_paths) if path in val_paths]
    test_indices = [i for i, path in enumerate(dataset.image_paths) if path in test_paths]
    
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    val_dataset.dataset.transform = transform
    test_dataset.dataset.transform = transform
    
    # Load trained model
    model = EfficientNetClassifier(
        backbone="efficientnet_b0",
        num_classes=len(art_periods)
    )
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"📊 Dataset Statistics:")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Device: {device}")
    
    # Initialize ambiguity detector
    detector = AmbiguityDetector(
        softmax_pmax_threshold=0.6,
        softmax_gap_threshold=0.1,
        entropy_percentile_threshold=80.0
    )
    
    # Get validation data for threshold computation
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    val_logits = []
    val_labels = []
    
    print("\n🔄 Processing validation set...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            val_logits.append(logits.cpu())
            val_labels.append(labels)
    
    val_logits = torch.cat(val_logits, dim=0)
    val_labels = torch.cat(val_labels, dim=0)
    val_probs = torch.softmax(val_logits, dim=1)
    
    # Get test data
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    test_logits = []
    test_labels = []
    
    print("🔄 Processing test set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            test_logits.append(logits.cpu())
            test_labels.append(labels)
    
    test_logits = torch.cat(test_logits, dim=0)
    test_labels = torch.cat(test_labels, dim=0)
    
    # Comprehensive ambiguity analysis
    print("\n📈 Computing ambiguity statistics...")
    stats = detector.analyze_ambiguity_statistics(test_logits, test_labels, val_probs)
    
    # Print statistics
    print(f"\n📊 AMBIGUITY DETECTION RESULTS:")
    print(f"{'='*50}")
    print(f"Overall Test Accuracy: {stats['overall_accuracy']:.3f}")
    print(f"Average Max Probability: {stats['average_max_prob']:.3f}")
    print(f"Average Probability Gap: {stats['average_prob_gap']:.3f}")
    print(f"Average Entropy: {stats['average_entropy']:.3f}")
    print()
    
    print(f"📉 SOFTMAX-BASED DETECTION:")
    print(f"  Ambiguous samples: {stats['softmax_ambiguous_count']}/{len(test_labels)} ({stats['softmax_ambiguous_percentage']:.1f}%)")
    print(f"  Accuracy on ambiguous: {stats['softmax_ambiguous_accuracy']:.3f}")
    print()
    
    print(f"📈 ENTROPY-BASED DETECTION:")
    print(f"  Ambiguous samples: {stats['entropy_ambiguous_count']}/{len(test_labels)} ({stats['entropy_ambiguous_percentage']:.1f}%)")
    print(f"  Accuracy on ambiguous: {stats['entropy_ambiguous_accuracy']:.3f}")
    print()
    
    print(f"🔄 COMBINED DETECTION:")
    print(f"  Ambiguous samples: {stats['combined_ambiguous_count']}/{len(test_labels)} ({stats['combined_ambiguous_percentage']:.1f}%)")
    print(f"  Accuracy on ambiguous: {stats['combined_ambiguous_accuracy']:.3f}")
    print()
    
    print(f"🔗 OVERLAP ANALYSIS:")
    print(f"  Softmax ∩ Entropy: {stats['softmax_entropy_overlap']}/{len(test_labels)} ({stats['softmax_entropy_overlap_percentage']:.1f}%)")
    
    # Get detailed ambiguity masks for visualization
    softmax_mask, softmax_metrics = detector.get_softmax_ambiguity(test_logits)
    entropy_mask, entropy_metrics = detector.get_entropy_ambiguity(test_logits, val_probs)
    combined_mask, _ = detector.get_combined_ambiguity(test_logits, val_probs)
    
    # Create visualizations
    print(f"\n🎨 Creating visualizations...")
    
    # 1. Confidence vs Entropy scatter plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    probs = torch.softmax(test_logits, dim=1)
    max_probs, _ = torch.max(probs, dim=1)
    entropy = entropy_metrics['entropy']
    predictions = torch.argmax(test_logits, dim=1)
    correct = (predictions == test_labels)
    
    # Scatter plot: Confidence vs Entropy
    ax = axes[0, 0]
    colors = ['red' if not c else 'green' for c in correct]
    scatter = ax.scatter(max_probs.numpy(), entropy.numpy(), c=colors, alpha=0.6)
    ax.set_xlabel('Max Probability (Confidence)')
    ax.set_ylabel('Entropy')
    ax.set_title('Confidence vs Entropy\n(Red=Incorrect, Green=Correct)')
    ax.axvline(x=0.6, color='blue', linestyle='--', alpha=0.7, label='Softmax threshold')
    ax.axhline(y=entropy_metrics['entropy_threshold'], color='orange', linestyle='--', alpha=0.7, label='Entropy threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Histogram: Confidence distribution
    ax = axes[0, 1]
    ax.hist(max_probs[correct].numpy(), alpha=0.7, label='Correct', bins=20, color='green')
    ax.hist(max_probs[~correct].numpy(), alpha=0.7, label='Incorrect', bins=20, color='red')
    ax.axvline(x=0.6, color='blue', linestyle='--', alpha=0.7, label='Threshold')
    ax.set_xlabel('Max Probability (Confidence)')
    ax.set_ylabel('Count')
    ax.set_title('Confidence Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Histogram: Entropy distribution
    ax = axes[1, 0]
    ax.hist(entropy[correct].numpy(), alpha=0.7, label='Correct', bins=20, color='green')
    ax.hist(entropy[~correct].numpy(), alpha=0.7, label='Incorrect', bins=20, color='red')
    ax.axvline(x=entropy_metrics['entropy_threshold'], color='orange', linestyle='--', alpha=0.7, label='Threshold')
    ax.set_xlabel('Entropy')
    ax.set_ylabel('Count')
    ax.set_title('Entropy Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Ambiguity detection comparison
    ax = axes[1, 1]
    detection_counts = [
        stats['softmax_ambiguous_count'],
        stats['entropy_ambiguous_count'], 
        stats['combined_ambiguous_count'],
        stats['softmax_entropy_overlap']
    ]
    detection_labels = ['Softmax', 'Entropy', 'Combined', 'Overlap']
    bars = ax.bar(detection_labels, detection_counts, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    ax.set_ylabel('Number of Ambiguous Samples')
    ax.set_title('Ambiguity Detection Comparison')
    ax.grid(True, alpha=0.3)
    
    # Add count labels on bars
    for bar, count in zip(bars, detection_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ambiguity_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Per-class ambiguity analysis
    per_class_stats = {}
    for class_idx, class_name in enumerate(art_periods):
        class_mask = test_labels == class_idx
        if class_mask.sum() > 0:
            class_logits = test_logits[class_mask]
            class_labels = test_labels[class_mask]
            
            class_softmax_mask, _ = detector.get_softmax_ambiguity(class_logits)
            class_entropy_mask, _ = detector.get_entropy_ambiguity(class_logits, val_probs)
            class_combined_mask, _ = detector.get_combined_ambiguity(class_logits, val_probs)
            
            class_probs = torch.softmax(class_logits, dim=1)
            class_max_probs, _ = torch.max(class_probs, dim=1)
            class_entropy = -(class_probs * torch.log(class_probs + 1e-9)).sum(dim=1)
            class_predictions = torch.argmax(class_logits, dim=1)
            class_correct = (class_predictions == class_labels)
            
            per_class_stats[class_name] = {
                'total_samples': class_mask.sum().item(),
                'accuracy': class_correct.float().mean().item(),
                'avg_confidence': class_max_probs.mean().item(),
                'avg_entropy': class_entropy.mean().item(),
                'softmax_ambiguous': class_softmax_mask.sum().item(),
                'entropy_ambiguous': class_entropy_mask.sum().item(),
                'combined_ambiguous': class_combined_mask.sum().item(),
            }
    
    # Create per-class analysis table
    df = pd.DataFrame(per_class_stats).T
    df['softmax_ambiguous_pct'] = (df['softmax_ambiguous'] / df['total_samples'] * 100).round(1)
    df['entropy_ambiguous_pct'] = (df['entropy_ambiguous'] / df['total_samples'] * 100).round(1)
    df['combined_ambiguous_pct'] = (df['combined_ambiguous'] / df['total_samples'] * 100).round(1)
    
    print(f"\n📋 PER-CLASS AMBIGUITY ANALYSIS:")
    print(f"{'='*80}")
    print(df.round(3).to_string())
    
    # Save results
    df.to_csv(output_dir / 'per_class_ambiguity_stats.csv')
    
    # Save overall statistics
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(output_dir / 'overall_ambiguity_stats.csv', index=False)
    
    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")
    print(f"   - ambiguity_analysis.png: Visualization plots")
    print(f"   - per_class_ambiguity_stats.csv: Per-class statistics")
    print(f"   - overall_ambiguity_stats.csv: Overall statistics")
    
    return stats, per_class_stats

if __name__ == "__main__":
    analyze_model_ambiguity()
