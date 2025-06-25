import os
import pandas as pd
from typing import Optional, Callable, List, Dict, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from sklearn.model_selection import train_test_split


class ArtPeriodDataset(Dataset):
    """Custom dataset for art period classification."""
    
    def __init__(
        self,
        data_dir: str,
        art_periods: List[str],
        split: str = "train",
        transform: Optional[Callable] = None,
        image_paths: Optional[List[str]] = None,
        labels: Optional[List[int]] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Root directory containing art images
            art_periods: List of art period names
            split: Dataset split ('train', 'val', 'test')
            transform: Image transformations to apply
            image_paths: Pre-computed image paths (optional)
            labels: Pre-computed labels (optional)
        """
        self.data_dir = data_dir
        self.art_periods = art_periods
        self.split = split
        self.transform = transform
        
        # Create label mapping
        self.period_to_idx = {period: idx for idx, period in enumerate(art_periods)}
        self.idx_to_period = {idx: period for period, idx in self.period_to_idx.items()}
        
        if image_paths is not None and labels is not None:
            self.image_paths = image_paths
            self.labels = labels
        else:
            self.image_paths, self.labels = self._load_data()
    
    def _load_data(self) -> Tuple[List[str], List[int]]:
        """Load image paths and labels from directory structure."""
        image_paths = []
        labels = []
        
        for period in self.art_periods:
            period_dir = os.path.join(self.data_dir, period)
            if not os.path.exists(period_dir):
                print(f"Warning: Directory {period_dir} not found")
                continue
            
            label = self.period_to_idx[period]
            
            for filename in os.listdir(period_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    image_paths.append(os.path.join(period_dir, filename))
                    labels.append(label)
        
        return image_paths, labels
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image_tensor, label)
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get the distribution of classes in the dataset."""
        distribution = {}
        for period in self.art_periods:
            count = self.labels.count(self.period_to_idx[period])
            distribution[period] = count
        return distribution


class DatasetSplitter:
    """Utility class for splitting datasets."""
    
    @staticmethod
    def stratified_split(
        image_paths: List[str],
        labels: List[int],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42
    ) -> Tuple[Tuple[List[str], List[int]], ...]:
        """
        Perform stratified split of the dataset.
        
        Args:
            image_paths: List of image file paths
            labels: List of corresponding labels
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_data, val_data, test_data) where each is (paths, labels)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"
        
        # First split: train vs (val + test)
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            image_paths, labels,
            test_size=(val_ratio + test_ratio),
            stratify=labels,
            random_state=random_state
        )
        
        # Second split: val vs test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels,
            test_size=(1 - val_size),
            stratify=temp_labels,
            random_state=random_state
        )
        
        return (
            (train_paths, train_labels),
            (val_paths, val_labels),
            (test_paths, test_labels)
        )


def create_transforms(image_size: int = 224, is_training: bool = True) -> transforms.Compose:
    """
    Create image transformations for training or validation.
    """
    if is_training:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),  # More aggressive crop
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.3,  # Stronger augmentation
                contrast=0.3,
                saturation=0.3,
                hue=0.08
            ),
            transforms.RandomGrayscale(p=0.15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    return transform


def create_tta_transforms(image_size: int = 224, n: int = 5) -> list:
    """
    Create a list of image transformations for Test-Time Augmentation (TTA).
    Args:
        image_size: Target image size for the model.
        n: Number of TTA transforms to generate.
    Returns:
        List of torchvision transforms.Compose objects.
    """
    tta_transforms = []
    for _ in range(n):
        tta_transforms.append(
            transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        )
    return tta_transforms
