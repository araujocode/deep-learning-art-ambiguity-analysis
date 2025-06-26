import os
import time
import logging
from typing import Dict, List, Tuple, Optional, Callable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
import numpy as np
from torchmetrics.classification import MulticlassAccuracy, MulticlassCalibrationError
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.efficientnet_classifier import EfficientNetClassifier, ModelManager
from .regularization import MixUp, mixup_criterion, CutMix, cutmix_criterion
import torch.nn.functional as F
from torchvision.transforms import RandAugment, AutoAugment, AutoAugmentPolicy


class Trainer:
    """Trainer class for art period classification with two-phase fine-tuning."""
    
    def __init__(
        self,
        model: EfficientNetClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        output_dir: str,
        logger: Optional[logging.Logger] = None,
        use_mixup: bool = False,
        mixup_alpha: float = 0.2,
        use_cutmix: bool = False,
        cutmix_alpha: float = 1.0,
        use_focal_loss: bool = False,
        ambiguity_weights: Optional[torch.Tensor] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: EfficientNet classifier
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            output_dir: Directory to save outputs
            logger: Logger instance
            use_mixup: Whether to use MixUp augmentation
            mixup_alpha: Alpha value for MixUp
            use_cutmix: Whether to use CutMix augmentation
            cutmix_alpha: Alpha value for CutMix
            use_focal_loss: Whether to use Focal Loss
            ambiguity_weights: Ambiguity weights for samples (if using Focal Loss)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True) # Ensure output directory exists
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True) # Ensure checkpoints directory exists
        self.logger = logger or self._setup_logger()
        
        self.model_manager = ModelManager(model)
        
        # Metrics
        self.accuracy_metric = MulticlassAccuracy(num_classes=model.num_classes).to(device)
        self.calibration_metric = MulticlassCalibrationError(
            num_classes=model.num_classes,
            n_bins=15,
            norm='l1'
        ).to(device)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_ece': [],
            'learning_rates': []
        }
        
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
        # Augmentations
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.use_cutmix = use_cutmix
        self.cutmix_alpha = cutmix_alpha
        
        # Focal Loss
        self.use_focal_loss = use_focal_loss
        self.ambiguity_weights = ambiguity_weights
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for training."""
        logger = logging.getLogger('Trainer')
        logger.setLevel(logging.INFO)
        
        # File handler
        log_file = os.path.join(self.output_dir, 'training.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _get_loss(self, label_smoothing: float = 0.1):
        """Get loss function with optional Focal Loss."""
        if self.use_focal_loss:
            return FocalLoss(gamma=2.0, weight=self.ambiguity_weights)
        else:
            return nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=self.ambiguity_weights)

    def train_phase1(
        self,
        num_epochs: int = 3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
        label_smoothing: float = 0.1,
        lr_scheduler: str = "onecycle",
        early_stopping_patience: int = 5
    ) -> Dict[str, List[float]]:
        """
        Phase 1: Train only the classifier head.
        
        Args:
            num_epochs: Number of epochs for phase 1
            learning_rate: Learning rate for phase 1
            weight_decay: Weight decay
            label_smoothing: Label smoothing factor
            lr_scheduler: Learning rate scheduler type ('onecycle', 'plateau', or 'cosine')
            early_stopping_patience: Early stopping patience (epochs)
        
        Returns:
            Training history for phase 1
        """
        self.logger.info("Starting Phase 1: Head-only training")
        
        # Freeze backbone, unfreeze classifier
        self.model.freeze_backbone()
        
        # Setup optimizer and scheduler
        optimizer = optim.AdamW(
            self.model.get_trainable_parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        if lr_scheduler == "onecycle":
            scheduler = OneCycleLR(
                optimizer,
                max_lr=learning_rate,
                steps_per_epoch=len(self.train_loader),
                epochs=num_epochs
            )
        elif lr_scheduler == "plateau":
            scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        else:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=num_epochs,
                eta_min=1e-6
            )
        
        # Loss function
        criterion = self._get_loss(label_smoothing=label_smoothing)
        
        # Train
        phase1_history = self._train_epochs(
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            num_epochs=num_epochs,
            phase_name="Phase1",
            early_stopping_patience=early_stopping_patience
        )
        
        self.logger.info(f"Phase 1 completed. Best val accuracy: {self.best_val_acc:.4f}")
        return phase1_history
    
    def train_phase2(
        self,
        num_epochs: int = 7,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        label_smoothing: float = 0.1,
        unfreeze_last_n_blocks: int = 2,
        early_stopping_patience: int = 12,
        lr_scheduler: str = "onecycle"
    ) -> Dict[str, List[float]]:
        """
        Phase 2: Unfreeze last blocks and continue training.
        
        Args:
            num_epochs: Number of epochs for phase 2
            learning_rate: Learning rate for phase 2
            weight_decay: Weight decay
            label_smoothing: Label smoothing factor
            unfreeze_last_n_blocks: Number of last blocks to unfreeze
            early_stopping_patience: Early stopping patience (epochs)
            lr_scheduler: Learning rate scheduler type ('onecycle', 'plateau', or 'cosine')
            
        Returns:
            Training history for phase 2
        """
        self.logger.info(f"Starting Phase 2: Unfreezing last {unfreeze_last_n_blocks} blocks")
        
        # Unfreeze last blocks
        self.model.freeze_early_layers(unfreeze_last_n_blocks)
        
        # Setup optimizer and scheduler
        optimizer = optim.AdamW(
            self.model.get_trainable_parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        if lr_scheduler == "onecycle":
            scheduler = OneCycleLR(
                optimizer,
                max_lr=learning_rate,
                steps_per_epoch=len(self.train_loader),
                epochs=num_epochs
            )
        elif lr_scheduler == "plateau":
            scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        else:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=num_epochs,
                eta_min=1e-6
            )
        
        # Loss function
        criterion = self._get_loss(label_smoothing=label_smoothing)
        
        # Train with early stopping
        phase2_history = self._train_epochs(
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            num_epochs=num_epochs,
            phase_name="Phase2",
            early_stopping_patience=early_stopping_patience
        )
        
        self.logger.info(f"Phase 2 completed. Best val accuracy: {self.best_val_acc:.4f}")
        return phase2_history
    
    def _train_epochs(
        self,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        criterion: nn.Module,
        num_epochs: int,
        phase_name: str,
        early_stopping_patience: int = None
    ) -> Dict[str, List[float]]:
        """
        Train for specified number of epochs, with optional early stopping.
        
        Args:
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler
            criterion: Loss function
            num_epochs: Number of epochs to train
            phase_name: Name of the training phase
            early_stopping_patience: Early stopping patience (epochs)
            
        Returns:
            Training history
        """
        phase_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_ece': [],
            'learning_rates': []
        }
        
        best_val_acc = 0.0  # Initialize best_val_acc here
        epochs_since_improvement = 0
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            self.logger.info(f"Starting {phase_name} Epoch {epoch + 1}/{num_epochs}, LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Training
            train_loss, train_acc = self._train_epoch(optimizer, criterion, phase_name, epoch + 1, num_epochs)
            
            # Validation
            val_loss, val_acc, val_ece = self._validate_epoch(criterion, phase_name, epoch + 1, num_epochs)
            
            # Scheduler step
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Update history
            phase_history['train_loss'].append(train_loss)
            phase_history['train_acc'].append(train_acc)
            phase_history['val_loss'].append(val_loss)
            phase_history['val_acc'].append(val_acc)
            phase_history['val_ece'].append(val_ece)
            phase_history['learning_rates'].append(current_lr)
            
            # Update global history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_ece'].append(val_ece)
            self.history['learning_rates'].append(current_lr)

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch + len(self.history['val_acc']) - len(phase_history['val_acc']) # Adjust epoch number for overall history
                self.logger.info(f"New best validation accuracy: {val_acc:.4f}. Saving model...")
                self.model_manager.save_checkpoint(
                    filepath=os.path.join(self.output_dir, 'checkpoints', 'best_model.pth'),
                    epoch=self.best_epoch,
                    metrics={'val_loss': val_loss, 'val_acc': val_acc}
                )
                epochs_since_improvement = 0
            else:
                self.logger.info(f"Validation accuracy did not improve from {self.best_val_acc:.4f}.")
                epochs_since_improvement += 1
            
            # Save checkpoint
            current_overall_epoch = epoch + len(self.history['val_acc']) - len(phase_history['val_acc'])
            self.model_manager.save_checkpoint(
                filepath=os.path.join(self.output_dir, 'checkpoints', 'latest_checkpoint.pth'),
                epoch=current_overall_epoch,
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                metrics={
                    'train_loss': train_loss, 
                    'train_acc': train_acc, 
                    'val_loss': val_loss, 
                    'val_acc': val_acc, 
                    'val_ece': val_ece, 
                    'lr': current_lr
                }
            )
            
            epoch_duration = time.time() - epoch_start_time
            self.logger.info(
                f"{phase_name} Epoch {epoch + 1}/{num_epochs} Summary: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val ECE: {val_ece:.4f}, "
                f"LR: {current_lr:.6f}, Duration: {epoch_duration:.2f}s"
            )
            
            # Early stopping check
            if early_stopping_patience is not None and epochs_since_improvement >= early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs with no improvement in validation accuracy for {early_stopping_patience} epochs.")
                break
            
        return phase_history

    def _train_epoch(
        self, 
        optimizer: optim.Optimizer, 
        criterion: nn.Module,
        phase_name: str,
        current_epoch: int,
        total_epochs: int
    ) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        self.accuracy_metric.reset()
        progress_bar = tqdm(self.train_loader, desc=f"{phase_name} Epoch {current_epoch}/{total_epochs} [Training]", unit="batch")
        mixup = MixUp(self.mixup_alpha) if self.use_mixup else None
        cutmix = CutMix(self.cutmix_alpha) if self.use_cutmix else None
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            if self.use_cutmix:
                cutmix_x, y_a, y_b, lam = cutmix(inputs, labels)
                outputs = self.model(cutmix_x)
                loss = cutmix_criterion(criterion, outputs, y_a, y_b, lam)
            elif self.use_mixup:
                mixed_x, y_a, y_b, lam = mixup(inputs, labels)
                outputs = self.model(mixed_x)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            self.accuracy_metric.update(outputs, labels)
            progress_bar.set_postfix(loss=loss.item(), acc=self.accuracy_metric.compute().item())
        avg_loss = total_loss / len(self.train_loader.dataset)
        avg_acc = self.accuracy_metric.compute().item()
        return avg_loss, avg_acc

    def _validate_epoch(
        self, 
        criterion: nn.Module,
        phase_name: str,
        current_epoch: int,
        total_epochs: int
    ) -> Tuple[float, float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        self.accuracy_metric.reset()
        self.calibration_metric.reset()
        
        # Add tqdm progress bar
        progress_bar = tqdm(self.val_loader, desc=f"{phase_name} Epoch {current_epoch}/{total_epochs} [Validation]", unit="batch")

        with torch.no_grad():
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item() * inputs.size(0)
                self.accuracy_metric.update(outputs, labels)
                self.calibration_metric.update(outputs, labels)

                # Update progress bar description
                progress_bar.set_postfix(loss=loss.item(), acc=self.accuracy_metric.compute().item(), ece=self.calibration_metric.compute().item())

        avg_loss = total_loss / len(self.val_loader.dataset)
        avg_acc = self.accuracy_metric.compute().item()
        avg_ece = self.calibration_metric.compute().item()
        return avg_loss, avg_acc, avg_ece
    
    def _save_best_model(
        self,
        epoch: int,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler
    ):
        """Save the best model checkpoint."""
        checkpoint_path = os.path.join(self.output_dir, "checkpoints", "best_model.pth")
        
        metrics = {
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'history': self.history
        };
        
        self.model_manager.save_checkpoint(
            filepath=checkpoint_path,
            epoch=epoch,
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),
            metrics=metrics
        );
        
        self.logger.info(f"Best model saved at epoch {epoch+1} with val accuracy: {self.best_val_acc:.4f}")
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(self.history['train_acc'], label='Train Acc')
        axes[0, 1].plot(self.history['val_acc'], label='Val Acc')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # ECE plot
        axes[1, 0].plot(self.history['val_ece'], label='Val ECE')
        axes[1, 0].set_title('Validation Expected Calibration Error')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('ECE')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate plot
        axes[1, 1].plot(self.history['learning_rates'], label='Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout(pad=3.0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_training_summary(self) -> Dict:
        """Get training summary statistics."""
        return {
            'best_val_accuracy': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'final_train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else 0,
            'final_val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else 0,
            'final_val_ece': self.history['val_ece'][-1] if self.history['val_ece'] else 0,
            'total_epochs': len(self.history['train_loss']),
            'parameter_count': self.model.get_parameter_count()
        }

# Focal Loss implementation
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = logpt.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = pt.gather(1, target.unsqueeze(1)).squeeze(1)
        loss = -((1 - pt) ** self.gamma) * logpt
        if self.weight is not None:
            loss = loss * self.weight[target]
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()

class AmbiguityWeightedLoss(nn.Module):
    """Downweight ambiguous samples using ambiguity scores."""
    def __init__(self, base_loss, ambiguity_scores: torch.Tensor, min_weight: float = 0.3):
        super().__init__()
        self.base_loss = base_loss
        self.ambiguity_scores = ambiguity_scores
        self.min_weight = min_weight
    def forward(self, input, target, indices=None):
        # indices: batch indices in the original dataset
        if indices is not None:
            weights = 1.0 - self.ambiguity_scores[indices].to(input.device)
            weights = torch.clamp(weights, min=self.min_weight, max=1.0)
        else:
            weights = torch.ones(input.size(0), device=input.device)
        loss = self.base_loss(input, target)
        if loss.dim() > 0:
            loss = loss * weights
            return loss.mean()
        else:
            return loss
