import torch
import torch.nn as nn
import timm
from typing import Optional, List, Dict, Any
import numpy as np


class EfficientNetClassifier(nn.Module):
    """EfficientNet-based classifier for art period classification.
    Supports efficientnet_b0, b1, b2, b3, b4 backbones (timm required).
    """
    
    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        num_classes: int = 8,
        pretrained: bool = True,
        dropout_rate: float = 0.3
    ):
        """
        Initialize the classifier.
        
        Args:
            backbone: EfficientNet variant to use
            num_classes: Number of art period classes
            pretrained: Whether to use ImageNet pretrained weights
            dropout_rate: Dropout rate before final classifier
        """
        super(EfficientNetClassifier, self).__init__()
        
        self.backbone_name = backbone
        self.num_classes = num_classes
        
        # Load pretrained EfficientNet
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove the default classifier
            global_pool=""  # Remove global pooling to add our own
        )
        
        # Get the number of features from the backbone
        self.feature_dim = self.backbone.num_features
        
        # Add custom classifier head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        
        # Initialize classifier weights
        self._init_classifier_weights()
    
    def _init_classifier_weights(self):
        """Initialize classifier weights."""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
            
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        # Extract features using backbone
        features = self.backbone(x)
        
        # Global average pooling
        features = self.global_pool(features)
        features = torch.flatten(features, 1)
        
        # Apply dropout and classifier
        features = self.dropout(features)
        logits = self.classifier(features)
        
        return logits
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before the classifier.
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
            
        Returns:
            Feature tensor of shape (batch_size, feature_dim)
        """
        with torch.no_grad():
            features = self.backbone(x)
            features = self.global_pool(features)
            features = torch.flatten(features, 1)
        return features
    
    def freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def freeze_blocks(self, n: int):
        """Freeze all but the last n blocks of the backbone."""
        self.freeze_backbone()
        if hasattr(self.backbone, 'blocks'):
            blocks_to_unfreeze = self.backbone.blocks[-n:]
            for block in blocks_to_unfreeze:
                for param in block.parameters():
                    param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True
    
    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Get list of trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter count statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params
        }
    
    def get_layer_for_gradcam(self, layer_name: str = "blocks[-1][-1].conv_dw") -> nn.Module:
        """
        Get specific layer for Grad-CAM visualization.
        
        Args:
            layer_name: Name of the target layer
            
        Returns:
            Target layer module
        """
        try:
            # Parse layer name and navigate to the layer
            parts = layer_name.replace("blocks", "backbone.blocks").split(".")
            layer = self
            
            for part in parts:
                if part.startswith('[') and part.endswith(']'):
                    # Handle indexing like 'blocks[-1]'
                    idx = int(part[1:-1])
                    layer = layer[idx]
                else:
                    layer = getattr(layer, part)
            
            return layer
        except (AttributeError, IndexError, ValueError) as e:
            print(f"Warning: Could not find layer {layer_name}. Error: {e}")
            # Return the last convolutional layer as fallback
            return self.backbone.blocks[-1]


class ModelManager:
    """Manager class for model operations."""
    
    def __init__(self, model: EfficientNetClassifier):
        """
        Initialize model manager.
        
        Args:
            model: EfficientNet classifier instance
        """
        self.model = model
    
    def save_checkpoint(
        self,
        filepath: str,
        epoch: int,
        optimizer_state: Optional[Dict] = None,
        scheduler_state: Optional[Dict] = None,
        metrics: Optional[Dict] = None
    ):
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save the checkpoint
            epoch: Current epoch number
            optimizer_state: Optimizer state dict
            scheduler_state: Scheduler state dict
            metrics: Training metrics
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'backbone': self.model.backbone_name,
                'num_classes': self.model.num_classes,
                'feature_dim': self.model.feature_dim
            }
        }
        
        if optimizer_state:
            checkpoint['optimizer_state_dict'] = optimizer_state
        if scheduler_state:
            checkpoint['scheduler_state_dict'] = scheduler_state
        if metrics:
            checkpoint['metrics'] = metrics
        
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(
        self,
        filepath: str,
        load_optimizer: bool = False,
        load_scheduler: bool = False
    ) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to the checkpoint file
            load_optimizer: Whether to return optimizer state
            load_scheduler: Whether to return scheduler state
            
        Returns:
            Dictionary containing loaded states
        """
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        result = {
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {})
        }
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            result['optimizer_state_dict'] = checkpoint['optimizer_state_dict']
        
        if load_scheduler and 'scheduler_state_dict' in checkpoint:
            result['scheduler_state_dict'] = checkpoint['scheduler_state_dict']
        
        return result
    
    def export_for_inference(self, filepath: str):
        """
        Export model for inference (state dict only).
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'backbone': self.model.backbone_name,
                'num_classes': self.model.num_classes,
                'feature_dim': self.model.feature_dim
            }
        }, filepath)
