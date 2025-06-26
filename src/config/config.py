import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import torch


@dataclass
class DataConfig:
    """Configuration for data handling."""
    data_dir: str = "./data/wikiart_real_processed"
    image_size: int = 224
    batch_size: int = 24  # Lower batch size for larger model
    num_workers: int = 4
    train_split: float = 0.7
    val_split: float = 0.2  # Larger validation set
    test_split: float = 0.1
    
    # Art periods to classify
    art_periods: List[str] = field(default_factory=lambda: [
        "Renaissance", "Baroque", "Romanticism", "Realism", "Impressionism", # Added Realism
        "Post-Impressionism", "Modernism", "Contemporary" # Ukiyo_e removed
    ])
    
    # Augmentation parameters
    random_crop_scale: Tuple[float, float] = (0.6, 1.0)
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    color_jitter_saturation: float = 0.2
    color_jitter_hue: float = 0.05
    random_grayscale_p: float = 0.1
    horizontal_flip_p: float = 0.5


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    # Add b3 and b4 as valid options
    backbone: str = "efficientnet_b2"  # Using a larger model
    # Valid options: efficientnet_b0, b1, b2, b3, b4
    pretrained: bool = True
    num_classes: int = 8 # Adjusted from 7 to 8 (Realism added)
    dropout_rate: float = 0.4  # More regularization
    
    # Fine-tuning phases
    phase1_epochs: int = 20  # More epochs for head
    phase2_epochs: int = 80 # More epochs for full model
    stage1_epochs: Optional[int] = None
    stage2_epochs: Optional[int] = None
    stage3_epochs: Optional[int] = None
    phase1_lr: float = 1e-3
    phase2_lr: float = 5e-5  # Lower learning rate for fine-tuning
    weight_decay: float = 0.05  # Increased weight decay
    
    # Regularization
    label_smoothing: float = 0.2  # Increased label smoothing
    use_mixup: bool = True
    mixup_alpha: float = 0.2
    use_cutmix: bool = False
    cutmix_alpha: float = 1.0
    use_focal_loss: bool = True
    unfreeze_last_n_blocks: int = 2  # Default to 2, can try 3-4
    lr_scheduler: str = "onecycle"  # Options: 'cosine', 'onecycle', 'plateau'


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    output_dir: str = "./experiments"
    save_checkpoints: bool = True
    early_stopping_patience: int = 5
    
    # Logging
    log_interval: int = 10
    save_best_only: bool = True


@dataclass
class AmbiguityConfig:
    """Configuration for ambiguity detection."""
    softmax_pmax_threshold: float = 0.6
    softmax_gap_threshold: float = 0.1
    entropy_percentile_threshold: float = 80.0
    run_analysis: bool = False
    
    # Grad-CAM settings
    target_layer_name: str = "blocks[-1][-1].conv_dw"
    
    # t-SNE settings
    tsne_n_samples: int = 1000
    tsne_perplexity: float = 30.0
    tsne_n_iter: int = 1000
    tsne_random_state: int = 42


@dataclass
class ProjectConfig:
    """Main project configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ambiguity: AmbiguityConfig = field(default_factory=AmbiguityConfig)
    
    def __post_init__(self):
        """Post-initialization to set dependent values."""
        self.model.num_classes = len(self.data.art_periods)
        # Set stage epochs if not provided, so they sum to phase2_epochs
        if self.model.stage1_epochs is None or self.model.stage2_epochs is None or self.model.stage3_epochs is None:
            # Default: 25%, 25%, 50% of phase2_epochs
            total = self.model.phase2_epochs
            self.model.stage1_epochs = int(0.25 * total)
            self.model.stage2_epochs = int(0.25 * total)
            self.model.stage3_epochs = total - self.model.stage1_epochs - self.model.stage2_epochs
        
        # Create output directories
        os.makedirs(self.training.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.training.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.training.output_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.training.output_dir, "visualizations"), exist_ok=True)
