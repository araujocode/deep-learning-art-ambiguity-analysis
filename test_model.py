#!/usr/bin/env python3
"""
Simple test script to verify the art classification model works.
Downloads a few sample art images, tests inference, and shows basic ambiguity.
"""

import os
import sys
import requests
from PIL import Image
import torch
import torch.nn.functional as F
# torchvision.transforms is already imported by `from data.dataset import create_transforms`
# import numpy as np # Not strictly needed if only using torch tensors for calculations shown
import random
from PIL import Image # Ensure PIL is imported here if not globally
    

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config.config import ProjectConfig, AmbiguityConfig # AmbiguityConfig might be needed if detector uses it directly
from src.models.efficientnet_classifier import EfficientNetClassifier
from src.data.dataset import create_transforms # For preprocessing
from src.ambiguity.detector import AmbiguityDetector # For ambiguity check

def download_or_create_sample_images(target_dir="data/sample_test_images"):
    """
    Downloads sample art images. If download fails or target_dir is empty,
    creates simple synthetic images.
    Ensures the target directory exists.
    Returns the path to the directory containing sample images.
    """
    sample_image_urls = {
        "mona_lisa.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/387px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg",
        "starry_night.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/525px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
        "girl_with_pearl.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/1665_Girl_with_a_Pearl_Earring.jpg/423px-1665_Girl_with_a_Pearl_Earring.jpg",
        "the_scream.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Edvard_Munch%2C_1893%2C_The_Scream%2C_oil%2C_tempera_and_pastel_on_cardboard%2C_91_x_73_cm%2C_National_Gallery_of_Norway.jpg/409px-Edvard_Munch%2C_1893%2C_The_Scream%2C_oil%2C_tempera_and_pastel_on_cardboard%2C_91_x_73_cm%2C_National_Gallery_of_Norway.jpg"
    }
    
    data_dir_path = Path(target_dir)
    data_dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Checking for sample images in {data_dir_path}...")
    
    headers = {'User-Agent': 'ArtClassificationTest/1.0 (python-requests)'}
    downloaded_at_least_one = False

    for filename, url in sample_image_urls.items():
        filepath = data_dir_path / filename
        if not filepath.exists():
            try:
                print(f"  Downloading {filename}...")
                response = requests.get(url, headers=headers, timeout=15)
                response.raise_for_status()
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print(f"  ✓ Downloaded {filename}")
                downloaded_at_least_one = True
            except Exception as e:
                print(f"  ✗ Failed to download {filename}: {e}")
        else:
            print(f"  ✓ {filename} already exists.")
            downloaded_at_least_one = True # Count existing as "available"

    # Check if any image files exist in the directory
    existing_images = list(data_dir_path.glob("*.jpg")) + list(data_dir_path.glob("*.png"))
    if not existing_images: # If no images were downloaded and none existed
        print(f"\\nNo images found or downloaded in {data_dir_path}. Creating synthetic test images...")
        create_synthetic_test_images(str(data_dir_path))
        existing_images = list(data_dir_path.glob("*.jpg")) + list(data_dir_path.glob("*.png"))

    if not existing_images:
        print(f"\\n❌ CRITICAL: No sample images available in {data_dir_path} even after attempting download/creation.")
        return None

    return str(data_dir_path)

def create_synthetic_test_images(data_dir_str):
    """Create synthetic test images if downloads fail."""
    
    data_dir = Path(data_dir_str)
    synthetic_images_data = [
        ("synthetic_renaissance.jpg", [139, 69, 19]),    # Brown/sepia
        ("synthetic_impressionism.jpg", [173, 216, 230]),  # Light blue
        ("synthetic_modernism.jpg", [255, 69, 0]),         # Red-orange
    ]
    
    for filename, base_color in synthetic_images_data:
        try:
            print(f"  Creating synthetic image: {filename}...")
            image = Image.new('RGB', (224, 224))
            pixels = []
            for y_coord in range(224):
                for x_coord in range(224):
                    r_val = min(255, max(0, base_color[0] + random.randint(-30, 30) + x_coord // 4))
                    g_val = min(255, max(0, base_color[1] + random.randint(-30, 30) + y_coord // 4))
                    b_val = min(255, max(0, base_color[2] + random.randint(-30, 30) + (x_coord + y_coord) // 8))
                    pixels.append((r_val, g_val, b_val))
            image.putdata(pixels)
            image.save(data_dir / filename)
            print(f"  ✓ Created {filename}")
        except Exception as e:
            print(f"  ✗ Failed to create synthetic {filename}: {e}")

def load_and_preprocess_image(image_path, transform):
    """Load and preprocess a single image."""
    try:
        image = Image.open(image_path).convert('RGB')
        return transform(image).unsqueeze(0)
    except FileNotFoundError:
        print(f"❌ Image not found at {image_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading or preprocessing {image_path}: {e}")
        return None

def test_and_demo_model():
    """
    Tests the model by loading it, running inference on sample images,
    and displaying detailed predictions including ambiguity.
    """
    print("🎨 Art Period Classification - Model Test & Demo")
    print("=" * 60)

    # Configuration (Consider moving art_periods to ProjectConfig if used widely)
    art_periods = ProjectConfig().data.art_periods # Get from config
    model_path_str = './experiments/run_001/checkpoints/best_model.pth' # Example path
    
    model_path = Path(model_path_str)
    if not model_path.exists():
        print(f"❌ Model checkpoint not found at {model_path_str}")
        print("Please ensure the model is trained and the path is correct.")
        return

    # Prepare sample images
    sample_images_dir_str = download_or_create_sample_images()
    if not sample_images_dir_str:
        print("❌ Cannot proceed without sample images.")
        return
        
    sample_images_dir = Path(sample_images_dir_str)
    test_image_paths = list(sample_images_dir.glob("*.jpg")) + list(sample_images_dir.glob("*.png"))

    if not test_image_paths:
        print(f"❌ No .jpg or .png images found in {sample_images_dir}")
        return

    print(f"\\n🖼️  Found {len(test_image_paths)} sample images in {sample_images_dir}")

    # Load trained model
    print(f"\\n🔄 Loading trained model from {model_path_str}...")
    try:
        model = EfficientNetClassifier(
            backbone=ProjectConfig().model.backbone, # Get from config
            num_classes=len(art_periods)
        )
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"✅ Model loaded successfully and moved to {device}.")
        print(f"   Model trained for {checkpoint.get('epoch', 'N/A')} epochs.")
        print(f"   Best validation accuracy during training: {checkpoint.get('metrics', {}).get('best_val_acc', 'N/A'):.4f}")

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Initialize ambiguity detector (using default thresholds from AmbiguityConfig)
    # If AmbiguityConfig is not directly used by detector, pass values from ProjectConfig
    amb_config = ProjectConfig().ambiguity
    detector = AmbiguityDetector(
        softmax_pmax_threshold=amb_config.softmax_pmax_threshold,
        softmax_gap_threshold=amb_config.softmax_gap_threshold,
        entropy_percentile_threshold=amb_config.entropy_percentile_threshold
    )
    print("\\n🔬 Ambiguity detector initialized.")

    # Create preprocessing transform
    transform = create_transforms(is_training=False, image_size=ProjectConfig().data.image_size)

    # Process each test image
    for i, image_path_obj in enumerate(test_image_paths):
        image_path_str = str(image_path_obj)
        print(f"\\n--- Processing Image {i+1}: {image_path_obj.name} ---")
        
        image_tensor = load_and_preprocess_image(image_path_str, transform)
        if image_tensor is None:
            continue
        
        image_tensor = image_tensor.to(device)
            
        with torch.no_grad():
            logits = model(image_tensor)
            probs = torch.softmax(logits, dim=1)
            
            sorted_probs, sorted_indices = torch.sort(probs[0], descending=True)
            
            max_prob_val = sorted_probs[0].item()
            # For entropy, detector might need validation set distribution.
            # For a simple demo, we might not have it or show raw entropy.
            # The current AmbiguityDetector.get_entropy_ambiguity expects val_probs.
            # For this demo, we'll calculate raw entropy and note that thresholding might differ.
            raw_entropy = -(probs[0] * torch.log(probs[0] + 1e-9)).sum().item()

            # Softmax-based ambiguity (doesn't need val_probs)
            softmax_amb_mask, softmax_metrics = detector.get_softmax_ambiguity(logits)
            is_softmax_ambiguous = softmax_amb_mask[0].item()

            print(f"  🎯 Top Prediction: {art_periods[sorted_indices[0]]}")
            print(f"  📊 Confidence (Max Prob): {max_prob_val:.4f}")
            print(f"   entropia (Raw): {raw_entropy:.4f}") # Changed label to "Entropy (Raw)"
            print(f"  ⚠️  Ambiguous (Softmax-based): {'Yes' if is_softmax_ambiguous else 'No'}")
            if is_softmax_ambiguous:
                print(f"     (Pmax: {softmax_metrics['p_max'][0].item():.3f}, Pgap: {softmax_metrics['p_gap'][0].item():.3f})")

            print(f"\\n  📋 Top 3 Predictions:")
            for j in range(min(3, len(sorted_indices))):
                period_name = art_periods[sorted_indices[j]]
                prob_val = sorted_probs[j].item()
                print(f"     {j+1}. {period_name:<22} {prob_val:.4f} {'█' * int(prob_val * 25)}")
    
    print("\\n" + "=" * 60)
    print("✅ Model test & demo complete!")
    print("Note: For full entropy-based ambiguity, the detector typically calibrates on a validation set.")
    print("This demo shows raw entropy and softmax-based ambiguity for the sample images.")

if __name__ == "__main__":
    # Ensure necessary imports from pathlib are available if used standalone
    from pathlib import Path 
    test_and_demo_model()
