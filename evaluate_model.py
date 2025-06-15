#!/usr/bin/env python3
"""
Evaluate the trained model on test data.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import torch.nn.functional as F
from models.efficientnet_classifier import EfficientNetClassifier
from data.dataset import ArtPeriodDataset, DatasetSplitter, create_transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model():
    """Evaluate the trained model on test data."""
    
    # Configuration
    art_periods = ['Renaissance', 'Baroque', 'Romanticism', 'Impressionism', 
                   'Post-Impressionism', 'Modernism', 'Surrealism', 'Contemporary']
    data_dir = './data/wikiart_real_processed'  # Updated path
    model_path = './experiments/run_001/checkpoints/best_model.pth'
    
    print("🎨 Art Period Classification - Model Evaluation")
    print("=" * 60)
    
    # Load dataset
    print("Loading dataset...")
    dataset = ArtPeriodDataset(data_dir, art_periods)
    
    # Create test split (same as training)
    splitter = DatasetSplitter()
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = \
        splitter.stratified_split(
            dataset.image_paths, dataset.labels, 
            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )
    
    # Create test dataset
    test_transform = create_transforms(is_training=False)
    test_indices = [i for i, path in enumerate(dataset.image_paths) if path in test_paths]
    test_dataset = Subset(dataset, test_indices)
    test_dataset.dataset.transform = test_transform
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Load trained model
    print("Loading trained model...")
    model = EfficientNetClassifier(
        backbone="efficientnet_b0",
        num_classes=len(art_periods)
    )
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    print(f"Best validation accuracy: {checkpoint['metrics']['best_val_acc']:.4f}")
    print(f"Best epoch: {checkpoint['metrics']['best_epoch']}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    all_predictions = []
    all_labels = []
    all_confidences = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            confidences = torch.max(probabilities, dim=1)[0]
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
    
    # Calculate metrics
    test_accuracy = accuracy_score(all_labels, all_predictions)
    
    print(f"\n🎯 TEST RESULTS:")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Average Confidence: {np.mean(all_confidences):.4f}")
    
    # Classification report
    print(f"\n📊 DETAILED CLASSIFICATION REPORT:")
    print("-" * 60)
    report = classification_report(
        all_labels, all_predictions, 
        target_names=art_periods, 
        digits=4
    )
    print(report)
    
    # Confusion matrix
    print(f"\n🔄 CONFUSION MATRIX:")
    cm = confusion_matrix(all_labels, all_predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=art_periods, yticklabels=art_periods)
    plt.title(f'Confusion Matrix - Test Accuracy: {test_accuracy:.2%}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save confusion matrix
    cm_path = './experiments/run_001/confusion_matrix.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {cm_path}")
    plt.show()
    
    # Analyze per-class performance
    print(f"\n📈 PER-CLASS ANALYSIS:")
    print("-" * 60)
    for i, period in enumerate(art_periods):
        class_mask = np.array(all_labels) == i
        if np.sum(class_mask) > 0:
            class_predictions = np.array(all_predictions)[class_mask]
            class_accuracy = np.mean(class_predictions == i)
            class_confidence = np.mean(np.array(all_confidences)[class_mask])
            print(f"{period:20s}: Accuracy={class_accuracy:.3f}, Avg Confidence={class_confidence:.3f}")
    
    return test_accuracy, all_predictions, all_labels, all_confidences

if __name__ == "__main__":
    evaluate_model()
