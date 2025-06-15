import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torchmetrics.classification import (
    MulticlassAccuracy, 
    MulticlassCalibrationError,
    MulticlassConfusionMatrix
)
import logging


class ModelEvaluator:
    """Comprehensive model evaluation class."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        class_names: List[str],
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model to evaluate
            device: Device for computation
            class_names: List of class names
            logger: Logger instance
        """
        self.model = model
        self.device = device
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize metrics
        self.accuracy_metric = MulticlassAccuracy(num_classes=self.num_classes).to(device)
        self.top2_accuracy_metric = MulticlassAccuracy(num_classes=self.num_classes, top_k=2).to(device)
        self.calibration_metric = MulticlassCalibrationError(
            num_classes=self.num_classes,
            n_bins=15,
            norm='l1'
        ).to(device)
        self.confusion_metric = MulticlassConfusionMatrix(num_classes=self.num_classes).to(device)
    
    def evaluate_dataset(
        self,
        data_loader,
        compute_features: bool = False
    ) -> Dict:
        """
        Evaluate model on a dataset.
        
        Args:
            data_loader: DataLoader for the dataset
            compute_features: Whether to extract and return features
            
        Returns:
            Dictionary containing evaluation results
        """
        self.model.eval()
        
        all_logits = []
        all_labels = []
        all_features = [] if compute_features else None
        total_loss = 0.0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                loss = criterion(output, target)
                
                # Collect outputs
                all_logits.append(output.cpu())
                all_labels.append(target.cpu())
                total_loss += loss.item()
                
                # Extract features if requested
                if compute_features:
                    features = self.model.extract_features(data)
                    all_features.append(features.cpu())
                
                # Update metrics
                probs = torch.softmax(output, dim=1)
                self.accuracy_metric.update(output, target)
                self.top2_accuracy_metric.update(output, target)
                self.calibration_metric.update(probs, target)
                self.confusion_metric.update(output, target)
        
        # Concatenate all results
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_probs = torch.softmax(all_logits, dim=1)
        
        if compute_features:
            all_features = torch.cat(all_features, dim=0)
        
        # Compute final metrics
        avg_loss = total_loss / len(data_loader)
        top1_accuracy = self.accuracy_metric.compute().item()
        top2_accuracy = self.top2_accuracy_metric.compute().item()
        ece = self.calibration_metric.compute().item()
        confusion_mat = self.confusion_metric.compute().cpu().numpy()
        
        # Reset metrics
        self.accuracy_metric.reset()
        self.top2_accuracy_metric.reset()
        self.calibration_metric.reset()
        self.confusion_metric.reset()
        
        # Compute additional metrics
        predictions = torch.argmax(all_logits, dim=1)
        per_class_accuracy = self._compute_per_class_accuracy(all_labels, predictions)
        
        results = {
            'loss': avg_loss,
            'top1_accuracy': top1_accuracy,
            'top2_accuracy': top2_accuracy,
            'ece': ece,
            'confusion_matrix': confusion_mat,
            'per_class_accuracy': per_class_accuracy,
            'logits': all_logits,
            'labels': all_labels,
            'probabilities': all_probs,
            'predictions': predictions
        }
        
        if compute_features:
            results['features'] = all_features
        
        return results
    
    def _compute_per_class_accuracy(
        self,
        labels: torch.Tensor,
        predictions: torch.Tensor
    ) -> Dict[str, float]:
        """Compute per-class accuracy."""
        per_class_acc = {}
        
        for i, class_name in enumerate(self.class_names):
            class_mask = (labels == i)
            if class_mask.sum() > 0:
                class_predictions = predictions[class_mask]
                class_labels = labels[class_mask]
                accuracy = (class_predictions == class_labels).float().mean().item()
                per_class_acc[class_name] = accuracy
            else:
                per_class_acc[class_name] = 0.0
        
        return per_class_acc
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        save_path: Optional[str] = None,
        normalize: bool = True
    ) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            confusion_matrix: Confusion matrix array
            save_path: Path to save the plot
            normalize: Whether to normalize the matrix
            
        Returns:
            Matplotlib figure
        """
        if normalize:
            cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            cm = confusion_matrix
            fmt = 'd'
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax
        )
        
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_calibration_curve(
        self,
        probabilities: torch.Tensor,
        labels: torch.Tensor,
        n_bins: int = 15,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot reliability diagram for model calibration.
        
        Args:
            probabilities: Model probability outputs
            labels: True labels
            n_bins: Number of bins for calibration
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Get maximum probabilities and predicted classes
        max_probs, predicted = torch.max(probabilities, dim=1)
        correct = (predicted == labels).float()
        
        # Create bins
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        accuracies = []
        confidences = []
        bin_sizes = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (max_probs > bin_lower) & (max_probs <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = correct[in_bin].mean()
                avg_confidence_in_bin = max_probs[in_bin].mean()
                
                accuracies.append(accuracy_in_bin.item())
                confidences.append(avg_confidence_in_bin.item())
                bin_sizes.append(in_bin.sum().item())
            else:
                accuracies.append(0)
                confidences.append(0)
                bin_sizes.append(0)
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        
        # Calibration bars
        ax.bar(confidences, accuracies, width=0.05, alpha=0.7, 
               label='Model Calibration', edgecolor='black')
        
        # Add bin size annotations
        for i, (conf, acc, size) in enumerate(zip(confidences, accuracies, bin_sizes)):
            if size > 0:
                ax.annotate(f'{size}', (conf, acc), 
                           textcoords="offset points", xytext=(0,5), ha='center')
        
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_title('Reliability Diagram')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_classification_report(
        self,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate detailed classification report.
        
        Args:
            labels: True labels
            predictions: Predicted labels
            save_path: Path to save the report
            
        Returns:
            Classification report string
        """
        report = classification_report(
            labels.numpy(),
            predictions.numpy(),
            target_names=self.class_names,
            digits=4
        )
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report
    
    def create_evaluation_summary(
        self,
        results: Dict,
        dataset_name: str = "Test"
    ) -> Dict[str, float]:
        """
        Create summary of evaluation results.
        
        Args:
            results: Results dictionary from evaluate_dataset
            dataset_name: Name of the evaluated dataset
            
        Returns:
            Summary dictionary
        """
        summary = {
            f'{dataset_name.lower()}_loss': results['loss'],
            f'{dataset_name.lower()}_top1_accuracy': results['top1_accuracy'],
            f'{dataset_name.lower()}_top2_accuracy': results['top2_accuracy'],
            f'{dataset_name.lower()}_ece': results['ece'],
        }
        
        # Add per-class accuracies
        for class_name, acc in results['per_class_accuracy'].items():
            summary[f'{dataset_name.lower()}_{class_name.lower()}_accuracy'] = acc
        
        return summary
