import torch
import numpy as np
from typing import Dict, Tuple, List, Optional
import logging


class AmbiguityDetector:
    """Class for detecting stylistically ambiguous artworks."""
    
    def __init__(
        self,
        softmax_pmax_threshold: float = 0.5,
        softmax_gap_threshold: float = 0.1,
        entropy_percentile_threshold: float = 90.0,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize ambiguity detector.
        
        Args:
            softmax_pmax_threshold: Threshold for maximum probability
            softmax_gap_threshold: Threshold for probability gap between top-1 and top-2
            entropy_percentile_threshold: Percentile threshold for entropy-based detection
            logger: Logger instance
        """
        self.softmax_pmax_threshold = softmax_pmax_threshold
        self.softmax_gap_threshold = softmax_gap_threshold
        self.entropy_percentile_threshold = entropy_percentile_threshold
        self.logger = logger or logging.getLogger('my_project.trainer')
        
        # Thresholds computed from validation set
        self.entropy_threshold_value = None
    
    def get_softmax_ambiguity(
        self,
        logits: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Detect ambiguity using softmax-based rules.
        
        Args:
            logits: Model logits of shape (batch_size, num_classes)
            
        Returns:
            Tuple of (ambiguous_mask, metrics_dict)
        """
        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
            p_max, top1_indices = probs.max(dim=1)
            
            # Get top-2 probabilities
            sorted_probs, sorted_indices = torch.sort(probs, dim=1, descending=True)
            
            # Calculate gap between top-1 and top-2
            if sorted_probs.size(1) > 1:
                gap = sorted_probs[:, 0] - sorted_probs[:, 1]
            else:
                gap = torch.ones_like(p_max) * float('inf')
            
            # Apply ambiguity rules
            low_confidence_mask = p_max < self.softmax_pmax_threshold
            small_gap_mask = gap < self.softmax_gap_threshold
            ambiguous_mask = low_confidence_mask | small_gap_mask
            
            metrics = {
                'probabilities': probs,
                'p_max': p_max,
                'gap': gap,
                'top1_indices': top1_indices,
                'top2_indices': sorted_indices[:, 1] if sorted_indices.size(1) > 1 else None,
                'low_confidence_mask': low_confidence_mask,
                'small_gap_mask': small_gap_mask
            }
            
            return ambiguous_mask, metrics
    
    def get_entropy_ambiguity(
        self,
        logits: torch.Tensor,
        validation_probs: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Detect ambiguity using entropy-based metric.
        
        Args:
            logits: Model logits of shape (batch_size, num_classes)
            validation_probs: Validation set probabilities for threshold computation
            
        Returns:
            Tuple of (ambiguous_mask, metrics_dict)
        """
        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
            
            # Compute entropy
            entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=1)
            
            # Compute threshold if validation probabilities provided
            if validation_probs is not None:
                val_entropy = -(validation_probs * torch.log(validation_probs + 1e-9)).sum(dim=1)
                self.entropy_threshold_value = torch.quantile(
                    val_entropy, 
                    self.entropy_percentile_threshold / 100.0
                ).item()
                self.logger.info(f"Computed entropy threshold: {self.entropy_threshold_value:.4f}")
            
            # Apply ambiguity rule
            if self.entropy_threshold_value is not None:
                ambiguous_mask = entropy >= self.entropy_threshold_value
            else:
                self.logger.warning("Entropy threshold not set, returning all False")
                ambiguous_mask = torch.zeros_like(entropy, dtype=torch.bool)
            
            metrics = {
                'probabilities': probs,
                'entropy': entropy,
                'entropy_threshold': self.entropy_threshold_value
            }
            
            return ambiguous_mask, metrics
    
    def get_combined_ambiguity(
        self,
        logits: torch.Tensor,
        validation_probs: Optional[torch.Tensor] = None,
        combine_mode: str = 'union'
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Combine softmax and entropy-based ambiguity detection.
        
        Args:
            logits: Model logits of shape (batch_size, num_classes)
            validation_probs: Validation set probabilities for entropy threshold
            combine_mode: How to combine rules ('union', 'intersection')
            
        Returns:
            Tuple of (ambiguous_mask, combined_metrics_dict)
        """
        # Get individual ambiguity detections
        softmax_mask, softmax_metrics = self.get_softmax_ambiguity(logits)
        entropy_mask, entropy_metrics = self.get_entropy_ambiguity(logits, validation_probs)
        
        # Combine masks
        if combine_mode == 'union':
            combined_mask = softmax_mask | entropy_mask
        elif combine_mode == 'intersection':
            combined_mask = softmax_mask & entropy_mask
        else:
            raise ValueError(f"Unknown combine_mode: {combine_mode}")
        
        # Combine metrics
        combined_metrics = {
            **softmax_metrics,
            **entropy_metrics,
            'softmax_ambiguous': softmax_mask,
            'entropy_ambiguous': entropy_mask,
            'combine_mode': combine_mode
        }
        
        return combined_mask, combined_metrics
    
    def analyze_ambiguity_statistics(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        validation_probs: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Analyze ambiguity detection statistics.
        
        Args:
            logits: Model logits
            labels: True labels
            validation_probs: Validation probabilities for entropy threshold
            
        Returns:
            Dictionary of statistics
        """
        with torch.no_grad():
            # Get ambiguity detections
            softmax_mask, softmax_metrics = self.get_softmax_ambiguity(logits)
            entropy_mask, entropy_metrics = self.get_entropy_ambiguity(logits, validation_probs)
            combined_mask, _ = self.get_combined_ambiguity(logits, validation_probs, 'union')
            
            # Get predictions
            predictions = torch.argmax(logits, dim=1)
            correct = (predictions == labels)
            
            total_samples = len(logits)
            
            # Calculate statistics
            stats = {
                # Softmax-based statistics
                'softmax_ambiguous_count': softmax_mask.sum().item(),
                'softmax_ambiguous_percentage': (softmax_mask.sum().item() / total_samples) * 100,
                'softmax_ambiguous_accuracy': correct[softmax_mask].float().mean().item() if softmax_mask.any() else 0.0,
                
                # Entropy-based statistics
                'entropy_ambiguous_count': entropy_mask.sum().item(),
                'entropy_ambiguous_percentage': (entropy_mask.sum().item() / total_samples) * 100,
                'entropy_ambiguous_accuracy': correct[entropy_mask].float().mean().item() if entropy_mask.any() else 0.0,
                
                # Combined statistics
                'combined_ambiguous_count': combined_mask.sum().item(),
                'combined_ambiguous_percentage': (combined_mask.sum().item() / total_samples) * 100,
                'combined_ambiguous_accuracy': correct[combined_mask].float().mean().item() if combined_mask.any() else 0.0,
                
                # Overlap statistics
                'softmax_entropy_overlap': (softmax_mask & entropy_mask).sum().item(),
                'softmax_entropy_overlap_percentage': ((softmax_mask & entropy_mask).sum().item() / total_samples) * 100,
                
                # General statistics
                'overall_accuracy': correct.float().mean().item(),
                'average_entropy': entropy_metrics['entropy'].mean().item(),
                'average_max_prob': softmax_metrics['p_max'].mean().item(),
                'average_prob_gap': softmax_metrics['gap'].mean().item()
            }
            
            return stats
    
    def get_ambiguous_samples(
        self,
        logits: torch.Tensor,
        image_indices: List[int],
        detection_method: str = 'combined',
        validation_probs: Optional[torch.Tensor] = None
    ) -> Dict[str, List[int]]:
        """
        Get indices of ambiguous samples for manual inspection.
        
        Args:
            logits: Model logits
            image_indices: Original image indices
            detection_method: Which detection method to use
            validation_probs: Validation probabilities for entropy threshold
            
        Returns:
            Dictionary mapping method names to lists of ambiguous indices
        """
        results = {}
        
        if detection_method in ['softmax', 'combined']:
            softmax_mask, _ = self.get_softmax_ambiguity(logits)
            softmax_indices = [image_indices[i] for i in range(len(image_indices)) if softmax_mask[i]]
            results['softmax'] = softmax_indices
        
        if detection_method in ['entropy', 'combined']:
            entropy_mask, _ = self.get_entropy_ambiguity(logits, validation_probs)
            entropy_indices = [image_indices[i] for i in range(len(image_indices)) if entropy_mask[i]]
            results['entropy'] = entropy_indices
        
        if detection_method == 'combined':
            combined_mask, _ = self.get_combined_ambiguity(logits, validation_probs, 'union')
            combined_indices = [image_indices[i] for i in range(len(image_indices)) if combined_mask[i]]
            results['combined'] = combined_indices
        
        return results
    
    def update_thresholds(
        self,
        softmax_pmax_threshold: Optional[float] = None,
        softmax_gap_threshold: Optional[float] = None,
        entropy_percentile_threshold: Optional[float] = None
    ):
        """
        Update detection thresholds.
        
        Args:
            softmax_pmax_threshold: New max probability threshold
            softmax_gap_threshold: New probability gap threshold
            entropy_percentile_threshold: New entropy percentile threshold
        """
        if softmax_pmax_threshold is not None:
            self.softmax_pmax_threshold = softmax_pmax_threshold
            self.logger.info(f"Updated softmax p_max threshold to {softmax_pmax_threshold}")
        
        if softmax_gap_threshold is not None:
            self.softmax_gap_threshold = softmax_gap_threshold
            self.logger.info(f"Updated softmax gap threshold to {softmax_gap_threshold}")
        
        if entropy_percentile_threshold is not None:
            self.entropy_percentile_threshold = entropy_percentile_threshold
            self.entropy_threshold_value = None  # Reset computed threshold
            self.logger.info(f"Updated entropy percentile threshold to {entropy_percentile_threshold}")
