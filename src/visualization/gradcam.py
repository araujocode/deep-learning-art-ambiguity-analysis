import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union
import cv2

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    print("Warning: pytorch-grad-cam not available. Install with: pip install grad-cam")


class GradCAMVisualizer:
    """Grad-CAM visualization for EfficientNet classifier."""
    
    def __init__(self, model: nn.Module, target_layer_name: str = "blocks[-1]"):
        """
        Initialize Grad-CAM visualizer.
        
        Args:
            model: Trained model
            target_layer_name: Name of target layer for CAM generation
        """
        if not GRADCAM_AVAILABLE:
            raise ImportError("pytorch-grad-cam is required for Grad-CAM visualization")
        
        self.model = model
        self.target_layer_name = target_layer_name
        self.target_layers = self._get_target_layers()
        
        # Initialize Grad-CAM
        self.cam = GradCAM(model=model, target_layers=self.target_layers)
    
    def _get_target_layers(self) -> List[nn.Module]:
        """Get target layers for Grad-CAM."""
        try:
            # Navigate to the target layer
            if hasattr(self.model, 'backbone'):
                # For our custom EfficientNetClassifier
                if "blocks[-1]" in self.target_layer_name:
                    return [self.model.backbone.blocks[-1]]
                elif "conv_head" in self.target_layer_name:
                    return [self.model.backbone.conv_head]
            else:
                # For direct timm models
                if "blocks[-1]" in self.target_layer_name:
                    return [self.model.blocks[-1]]
                elif "conv_head" in self.target_layer_name:
                    return [self.model.conv_head]
            
            # Fallback: try to find the last convolutional layer
            for name, module in reversed(list(self.model.named_modules())):
                if isinstance(module, nn.Conv2d):
                    return [module]
            
            raise ValueError("Could not find suitable target layer")
            
        except Exception as e:
            print(f"Warning: Could not find target layer {self.target_layer_name}. Error: {e}")
            # Return the model itself as fallback
            return [self.model]
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        original_image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Preprocessed input tensor (1, 3, H, W)
            target_class: Target class index for CAM
            original_image: Original image as numpy array (H, W, 3) in range [0, 1]
            
        Returns:
            Tuple of (cam_heatmap, cam_overlay)
        """
        targets = [ClassifierOutputTarget(target_class)]
        
        # Generate CAM
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]  # Get first (and only) image
        
        # Create overlay
        cam_overlay = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)
        
        return grayscale_cam, cam_overlay
    
    def visualize_predictions(
        self,
        input_tensor: torch.Tensor,
        original_image: np.ndarray,
        logits: torch.Tensor,
        class_names: List[str],
        top_k: int = 3,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create comprehensive Grad-CAM visualization for top predictions.
        
        Args:
            input_tensor: Preprocessed input tensor
            original_image: Original image as numpy array
            logits: Model output logits
            class_names: List of class names
            top_k: Number of top predictions to visualize
            save_path: Path to save the visualization
            
        Returns:
            Matplotlib figure
        """
        # Get top predictions
        probs = torch.softmax(logits, dim=1)
        top_probs, top_indices = torch.topk(probs, top_k, dim=1)
        top_probs = top_probs[0].cpu().numpy()
        top_indices = top_indices[0].cpu().numpy()
        
        # Create figure
        fig, axes = plt.subplots(1, top_k + 1, figsize=(4 * (top_k + 1), 4))
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Grad-CAM for each top prediction
        for i, (prob, class_idx) in enumerate(zip(top_probs, top_indices)):
            try:
                _, cam_overlay = self.generate_cam(input_tensor, class_idx, original_image)
                
                axes[i + 1].imshow(cam_overlay)
                axes[i + 1].set_title(f"{class_names[class_idx]}\n({prob:.3f})")
                axes[i + 1].axis('off')
                
            except Exception as e:
                print(f"Error generating CAM for class {class_idx}: {e}")
                axes[i + 1].imshow(original_image)
                axes[i + 1].set_title(f"{class_names[class_idx]}\n({prob:.3f})\n(CAM Error)")
                axes[i + 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def compare_ambiguous_vs_clear(
        self,
        ambiguous_data: Tuple[torch.Tensor, np.ndarray, torch.Tensor],
        clear_data: Tuple[torch.Tensor, np.ndarray, torch.Tensor],
        class_names: List[str],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Compare Grad-CAM visualizations for ambiguous vs clear predictions.
        
        Args:
            ambiguous_data: (input_tensor, original_image, logits) for ambiguous sample
            clear_data: (input_tensor, original_image, logits) for clear sample
            class_names: List of class names
            save_path: Path to save the visualization
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # Process ambiguous sample
        amb_input, amb_orig, amb_logits = ambiguous_data
        amb_probs = torch.softmax(amb_logits, dim=1)
        amb_top2_probs, amb_top2_indices = torch.topk(amb_probs, 2, dim=1)
        
        # Process clear sample
        clear_input, clear_orig, clear_logits = clear_data
        clear_probs = torch.softmax(clear_logits, dim=1)
        clear_top1_prob, clear_top1_idx = torch.topk(clear_probs, 1, dim=1)
        
        # Ambiguous sample visualizations
        axes[0, 0].imshow(amb_orig)
        axes[0, 0].set_title("Ambiguous Sample\n(Original)")
        axes[0, 0].axis('off')
        
        # Top-1 CAM for ambiguous
        try:
            _, amb_cam1 = self.generate_cam(amb_input, amb_top2_indices[0, 0], amb_orig)
            axes[0, 1].imshow(amb_cam1)
            axes[0, 1].set_title(f"Top-1: {class_names[amb_top2_indices[0, 0]]}\n({amb_top2_probs[0, 0]:.3f})")
        except:
            axes[0, 1].imshow(amb_orig)
            axes[0, 1].set_title("CAM Error")
        axes[0, 1].axis('off')
        
        # Top-2 CAM for ambiguous
        try:
            _, amb_cam2 = self.generate_cam(amb_input, amb_top2_indices[0, 1], amb_orig)
            axes[0, 2].imshow(amb_cam2)
            axes[0, 2].set_title(f"Top-2: {class_names[amb_top2_indices[0, 1]]}\n({amb_top2_probs[0, 1]:.3f})")
        except:
            axes[0, 2].imshow(amb_orig)
            axes[0, 2].set_title("CAM Error")
        axes[0, 2].axis('off')
        
        # Probability gap info
        gap = amb_top2_probs[0, 0] - amb_top2_probs[0, 1]
        axes[0, 3].text(0.5, 0.5, f"Probability Gap:\n{gap:.3f}\n\nMax Prob:\n{amb_top2_probs[0, 0]:.3f}", 
                       ha='center', va='center', fontsize=12, transform=axes[0, 3].transAxes)
        axes[0, 3].set_title("Ambiguity Metrics")
        axes[0, 3].axis('off')
        
        # Clear sample visualizations
        axes[1, 0].imshow(clear_orig)
        axes[1, 0].set_title("Clear Sample\n(Original)")
        axes[1, 0].axis('off')
        
        # CAM for clear sample
        try:
            _, clear_cam = self.generate_cam(clear_input, clear_top1_idx[0, 0], clear_orig)
            axes[1, 1].imshow(clear_cam)
            axes[1, 1].set_title(f"Prediction: {class_names[clear_top1_idx[0, 0]]}\n({clear_top1_prob[0, 0]:.3f})")
        except:
            axes[1, 1].imshow(clear_orig)
            axes[1, 1].set_title("CAM Error")
        axes[1, 1].axis('off')
        
        # Empty for symmetry
        axes[1, 2].axis('off')
        
        # Clear sample metrics
        axes[1, 3].text(0.5, 0.5, f"Max Prob:\n{clear_top1_prob[0, 0]:.3f}\n\nConfident\nPrediction", 
                       ha='center', va='center', fontsize=12, transform=axes[1, 3].transAxes)
        axes[1, 3].set_title("Confidence Metrics")
        axes[1, 3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def preprocess_image_for_gradcam(image_path: str, transform) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Preprocess image for Grad-CAM visualization.
    
    Args:
        image_path: Path to the image
        transform: Preprocessing transform
        
    Returns:
        Tuple of (preprocessed_tensor, original_image_array)
    """
    # Load original image
    original_image = Image.open(image_path).convert('RGB')
    
    # Resize for display while maintaining aspect ratio
    display_size = 224
    original_image.thumbnail((display_size, display_size), Image.Resampling.LANCZOS)
    
    # Convert to numpy array for Grad-CAM overlay
    original_array = np.array(original_image) / 255.0
    
    # Preprocess for model
    input_tensor = transform(original_image).unsqueeze(0)  # Add batch dimension
    
    return input_tensor, original_array
