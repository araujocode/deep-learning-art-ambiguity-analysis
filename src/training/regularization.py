"""
Regularization techniques for training.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class MixUp:
    """MixUp data augmentation for regularization."""
    
    def __init__(self, alpha: float = 0.2):
        """
        Initialize MixUp.
        
        Args:
            alpha: Beta distribution parameter for lambda sampling
        """
        self.alpha = alpha
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply MixUp to batch.
        
        Args:
            x: Input batch tensor
            y: Label tensor
            
        Returns:
            Tuple of (mixed_x, y_a, y_b, lambda)
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
    """
    Compute MixUp loss.
    
    Args:
        criterion: Loss function
        pred: Model predictions
        y_a: First set of labels
        y_b: Second set of labels
        lam: MixUp lambda parameter
        
    Returns:
        Mixed loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class LabelSmoothing:
    """Label smoothing regularization."""
    
    def __init__(self, smoothing: float = 0.1):
        """
        Initialize label smoothing.
        
        Args:
            smoothing: Smoothing factor
        """
        self.smoothing = smoothing
    
    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Apply label smoothing loss.
        
        Args:
            predictions: Model predictions (logits)
            targets: Target labels
            
        Returns:
            Smoothed loss
        """
        num_classes = predictions.size(-1)
        with torch.no_grad():
            # Create smoothed targets
            true_dist = torch.zeros_like(predictions)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.data.unsqueeze(1), 1.0 - self.smoothing)
        
        return torch.mean(torch.sum(-true_dist * F.log_softmax(predictions, dim=-1), dim=-1))
