#!/usr/bin/env python3
"""
Test ambiguity detection on the trained model.
"""

from src.models.efficientnet_classifier import EfficientNetClassifier
from src.data.dataset import ArtPeriodDataset, DatasetSplitter, create_transforms, safe_collate_fn
from src.ambiguity.detector import AmbiguityDetector
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def test_ambiguity_detection():
    """Test ambiguity detection on some sample images."""
    
    # Configuration
    art_periods = ['Renaissance', 'Baroque', 'Romanticism', 'Impressionism', 
                   'Post-Impressionism', 'Modernism', 'Surrealism', 'Contemporary']
    data_dir = './data/wikiart_real_processed' # Updated path
    model_path = './experiments/run_001/checkpoints/best_model.pth'
    
    print("🔍 Ambiguity Detection Test")
    print("=" * 40)
    
    # Load dataset
    dataset = ArtPeriodDataset(data_dir, art_periods)
    
    # Create test split
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
    
    # Load trained model
    model = EfficientNetClassifier(
        backbone="efficientnet_b0",
        num_classes=len(art_periods)
    )
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    
    # Initialize ambiguity detector
    detector = AmbiguityDetector(
        softmax_pmax_threshold=0.6,  # Lower threshold to detect more ambiguity
        softmax_gap_threshold=0.1,
        entropy_percentile_threshold=80.0
    )
    
    print(f"Testing ambiguity detection on {len(test_dataset)} test images...")
    
    # Test ambiguity detection
    ambiguous_results = []
    certain_results = []
    
    # Set model to evaluation mode
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False, collate_fn=safe_collate_fn)
    
    # Process all test images in batch
    all_logits = []
    all_labels = []
    all_images = []
    
    # First, get validation data to compute entropy threshold
    val_indices = [i for i, path in enumerate(dataset.image_paths) if path in val_paths]
    val_dataset = Subset(dataset, val_indices)
    val_dataset.dataset.transform = test_transform
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=safe_collate_fn)
    
    val_logits = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            val_logits.append(logits.cpu())
    
    val_logits = torch.cat(val_logits, dim=0) if val_logits else torch.empty(0, len(art_periods))
    val_probs = torch.softmax(val_logits, dim=1)
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
            all_images.extend(images.cpu())
            break  # Only process first batch (20 images)
    
    if all_logits:
        batch_logits = torch.cat(all_logits, dim=0)
        batch_labels = torch.cat(all_labels, dim=0)
        
        # Get ambiguity detections (now with validation data for entropy threshold)
        softmax_mask, softmax_metrics = detector.get_softmax_ambiguity(batch_logits)
        entropy_mask, entropy_metrics = detector.get_entropy_ambiguity(batch_logits, val_probs)
        combined_mask, _ = detector.get_combined_ambiguity(batch_logits, val_probs)
        
        # Get predictions and probabilities
        probs = torch.softmax(batch_logits, dim=1)
        predictions = torch.argmax(batch_logits, dim=1)
        max_probs, _ = torch.max(probs, dim=1)
        entropy = entropy_metrics['entropy']
        
        # Process each image
        for i in range(len(batch_labels)):
            true_class = art_periods[batch_labels[i].item()]
            pred_class = art_periods[predictions[i].item()]
            confidence = max_probs[i].item()
            entropy_val = entropy[i].item()
            is_ambiguous = combined_mask[i].item()
            
            print(f"\nImage {i+1}:")
            print(f"  True class: {true_class}")
            print(f"  Predicted: {pred_class}")
            print(f"  Confidence: {confidence:.3f}")
            print(f"  Entropy: {entropy_val:.3f}")
            print(f"  Ambiguous (softmax): {'Yes' if softmax_mask[i] else 'No'}")
            print(f"  Ambiguous (entropy): {'Yes' if entropy_mask[i] else 'No'}")
            print(f"  Ambiguous (combined): {'Yes' if is_ambiguous else 'No'}")
            
            # Get top predictions
            sorted_probs, sorted_indices = torch.sort(probs[i], descending=True)
            top_predictions = [(art_periods[idx.item()], prob.item()) 
                             for idx, prob in zip(sorted_indices[:3], sorted_probs[:3])]
            
            print(f"  Top predictions:")
            for j, (class_name, prob) in enumerate(top_predictions):
                print(f"    {j+1}. {class_name}: {prob:.3f}")
            
            result = {
                'true_class': true_class,
                'predicted_class': pred_class,
                'confidence': confidence,
                'entropy': entropy_val,
                'is_ambiguous': is_ambiguous,
                'top_predictions': top_predictions
            }
            
            if is_ambiguous:
                ambiguous_results.append(result)
            else:
                certain_results.append(result)
    
    print(f"\n📊 AMBIGUITY ANALYSIS:")
    print(f"Ambiguous predictions: {len(ambiguous_results)}/20 ({len(ambiguous_results)/20*100:.1f}%)")
    print(f"Certain predictions: {len(certain_results)}/20 ({len(certain_results)/20*100:.1f}%)")
    
    if ambiguous_results:
        avg_ambiguous_conf = np.mean([r['confidence'] for r in ambiguous_results])
        avg_ambiguous_entropy = np.mean([r['entropy'] for r in ambiguous_results])
        print(f"Average confidence for ambiguous: {avg_ambiguous_conf:.3f}")
        print(f"Average entropy for ambiguous: {avg_ambiguous_entropy:.3f}")
    
    if certain_results:
        avg_certain_conf = np.mean([r['confidence'] for r in certain_results])
        avg_certain_entropy = np.mean([r['entropy'] for r in certain_results])
        print(f"Average confidence for certain: {avg_certain_conf:.3f}")
        print(f"Average entropy for certain: {avg_certain_entropy:.3f}")
    
    return ambiguous_results, certain_results

if __name__ == "__main__":
    test_ambiguity_detection()
