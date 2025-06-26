import os
from typing import Optional, Callable, List, Dict, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import RandAugment, AutoAugment, AutoAugmentPolicy
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
        """Load image paths and labels from directory structure, skipping unreadable files."""
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
                    img_path = os.path.join(period_dir, filename)
                    try:
                        with Image.open(img_path) as img:
                            img.verify()  # Verify image is not corrupt
                        image_paths.append(img_path)
                        labels.append(label)
                    except Exception as e:
                        print(f"Warning: Skipping unreadable image {img_path}: {e}")
        return image_paths, labels
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int):
        """
        Get a sample from the dataset. If loading or transform fails, retry with a random index.
        If all attempts fail, return a CorruptSampleException object (do not raise).
        Returns (image, label, idx) where idx is the original dataset index.
        """
        max_attempts = 3
        attempt = 0
        orig_idx = idx
        # If this dataset is a Subset, map idx to original index
        if hasattr(self, 'indices'):
            orig_idx = self.indices[idx]
        while attempt < max_attempts:
            image_path = self.image_paths[idx]
            label = self.labels[idx]
            try:
                image = Image.open(image_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                return image, label, orig_idx
            except Exception as e:
                print(f"Warning: Error loading or transforming image {image_path}: {e}. Retrying with a different sample.")
                idx = np.random.randint(0, len(self.image_paths))
                if hasattr(self, 'indices'):
                    orig_idx = self.indices[idx]
                else:
                    orig_idx = idx
                attempt += 1
        # If all attempts fail, return exception object (do not raise)
        print("Error: Failed to load a valid image after multiple attempts. Returning CorruptSampleException object.")
        return CorruptSampleException(f"Failed to load image after {max_attempts} attempts.")
    
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


def create_transforms(image_size: int = 224, is_training: bool = True, use_randaugment: bool = True, use_autoaugment: bool = False, use_random_erasing: bool = True) -> transforms.Compose:
    """Create image transformations for training or evaluation."""
    transform_list = []
    if is_training:
        # Stronger augmentations
        if use_randaugment:
            transform_list.append(RandAugment())
        elif use_autoaugment:
            transform_list.append(AutoAugment(policy=AutoAugmentPolicy.IMAGENET))
        transform_list += [
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomGrayscale(p=0.1),
        ]
    else:
        transform_list.append(transforms.Resize((image_size, image_size)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    # Add RandomErasing after normalization (only for training)
    if is_training and use_random_erasing:
        transform_list.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'))
    return transforms.Compose(transform_list)


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


class CorruptSampleException(Exception):
    """Raised when a sample cannot be loaded after several attempts."""
    pass


def safe_collate_fn(batch):
    """
    Collate function that skips samples where CorruptSampleException was raised.
    Usage: DataLoader(..., collate_fn=safe_collate_fn)
    Now supports (image, label, idx) tuples for per-sample weighting.
    """
    filtered = []
    for b in batch:
        if isinstance(b, Exception):
            continue
        filtered.append(b)
    if len(filtered) == 0:
        return torch.empty(0), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    # Support both (image, label) and (image, label, idx)
    if len(filtered[0]) == 3:
        images, labels, indices = zip(*filtered)
        return torch.stack(images, 0), torch.tensor(labels, dtype=torch.long), torch.tensor(indices, dtype=torch.long)
    else:
        images, labels = zip(*filtered)
        return torch.stack(images, 0), torch.tensor(labels, dtype=torch.long), None
