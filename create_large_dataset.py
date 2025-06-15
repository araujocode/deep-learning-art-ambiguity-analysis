#!/usr/bin/env python3
"""
Organize a large WikiArt dataset from a local directory for improved training.
This script will use the real WikiArt dataset (expected in './wikiart') 
and organize it by art periods into './data/wikiart_real_processed'.
"""

import os
import pandas as pd
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import random
from collections import defaultdict

def map_styles_to_periods():
    """Map WikiArt styles to our target art periods."""
    style_mapping = {
        # Renaissance (1400-1600)
        'Early Renaissance': 'Renaissance',
        'High Renaissance': 'Renaissance',
        'Northern Renaissance': 'Renaissance',
        'Renaissance': 'Renaissance',
        
        # Baroque (1600-1750)
        'Baroque': 'Baroque',
        'Rococo': 'Baroque', # Often considered late Baroque
        
        # Romanticism (1800-1850)
        'Romanticism': 'Romanticism',
        'Neoclassicism': 'Romanticism', # Close period, often contrasted but related
        
        # Impressionism (1860-1890)
        'Impressionism': 'Impressionism',
        # 'Realism': 'Impressionism', # Realism was a precursor and contemporary movement -> Changed to its own period
        'Realism': 'Realism', # Now its own period
        
        # Post-Impressionism (1880-1905)
        'Post-Impressionism': 'Post-Impressionism',
        'Pointillism': 'Post-Impressionism',
        'Symbolism': 'Post-Impressionism', # Symbolism overlaps significantly
        
        # Modernism (1900-1945) - Broad category
        'Fauvism': 'Modernism',
        'Expressionism': 'Modernism',
        'Cubism': 'Modernism',
        'Analytical Cubism': 'Modernism', # Sub-genre of Cubism
        'Synthetic Cubism': 'Modernism', # Sub-genre of Cubism
        'Futurism': 'Modernism',
        'Constructivism': 'Modernism',
        'Dadaism': 'Modernism', # Often 'Dada'
        'Art Nouveau (Modern)': 'Modernism', # Art Nouveau is often considered proto-Modernist
        'Art_Nouveau_Modern': 'Modernism', # Handling underscore variant
        'Bauhaus': 'Modernism',
        'Abstract_Expressionism': 'Modernism', # Can also be Contemporary, but roots in Modernism
        'Action_painting': 'Modernism', # Sub-genre of Abstract Expressionism
        'Color_Field_Painting': 'Modernism', # Sub-genre of Abstract Expressionism
        
        # Surrealism (1920-1945) - REMOVED as not present in dataset
        
        # Contemporary (1945-present) - Broad category
        # 'Abstract Expressionism': 'Contemporary', # Moved to Modernism for broader grouping, can be argued
        'Pop Art': 'Contemporary',
        'Pop_Art': 'Contemporary',
        'Minimalism': 'Contemporary',
        'Contemporary Realism': 'Contemporary',
        'Contemporary_Realism': 'Contemporary',
        'Photorealism': 'Contemporary',
        'New Realism': 'Contemporary', # New_Realism
        'New_Realism': 'Contemporary',
        'Neo-Expressionism': 'Contemporary',
        'Street Art': 'Contemporary',
        'Digital Art': 'Contemporary',
        'Installation': 'Contemporary',

        # Other styles that might appear and need mapping
        'Naive Art (Primitivism)': 'Modernism', # Primitivism influenced Modernism
        'Naive_Art_Primitivism': 'Modernism',
        # 'Ukiyo_e': 'Post-Impressionism', # Influence on Impressionism/Post-Impressionism -> REMOVED
        'Mannerism_Late_Renaissance': 'Renaissance' # Late Renaissance
    }
    return style_mapping

def organize_wikiart_by_periods(raw_data_path: str, target_data_path: str, target_per_class: int = 3615): # Updated target_per_class
    """Organize WikiArt images from local source by art periods."""
    
    print(f"🔄 Organizing WikiArt dataset from '{raw_data_path}' into art periods at '{target_data_path}'...")
    print(f"Target: {target_per_class} images per class")
    
    raw_path = Path(raw_data_path)
    target_path = Path(target_data_path)
    
    art_periods = ['Renaissance', 'Baroque', 'Romanticism', 'Realism', 
                   'Impressionism', 'Post-Impressionism', 'Modernism', 'Contemporary'] # Added Realism, Ukiyo_e effectively removed
    
    for period in art_periods:
        (target_path / period).mkdir(parents=True, exist_ok=True)
    
    style_mapping = map_styles_to_periods()
    period_images = defaultdict(list)
    
    print(f"📂 Scanning source directory: {raw_path}")
    if not raw_path.exists() or not raw_path.is_dir():
        print(f"❌ Source directory '{raw_path}' not found or is not a directory.")
        print(f"Please ensure the WikiArt dataset is available at this location.")
        return {}

    # Primarily use directory structure for organization
    for style_dir in raw_path.iterdir():
        if style_dir.is_dir():
            style_name = style_dir.name
            # Normalize style name (e.g., replace underscores with spaces if mapping uses spaces)
            normalized_style_name = style_name.replace('_', ' ') 

            mapped_period = None
            if style_name in style_mapping:
                mapped_period = style_mapping[style_name]
            elif normalized_style_name in style_mapping:
                 mapped_period = style_mapping[normalized_style_name]
            else:
                # Try partial matches or case-insensitive
                for map_key, period_val in style_mapping.items():
                    if style_name.lower() == map_key.lower() or \
                       normalized_style_name.lower() == map_key.lower():
                        mapped_period = period_val
                        break
            
            if mapped_period and mapped_period in art_periods:
                print(f"  Found style: '{style_name}' -> Mapped to period: '{mapped_period}'")
                for image_file in style_dir.glob("*.jpg"): # Assuming .jpg, add other extensions if needed
                    period_images[mapped_period].append(image_file)
            # else:
            #     print(f"  Skipping style: '{style_name}' (not in mapping or target periods)")

    # Report collection results
    print(f"\n📊 Images collected by period (before balancing):")
    for period in art_periods:
        count = len(period_images[period])
        print(f"  {period:<20}: {count:>5} images")
        if count == 0:
            print(f"    ⚠️ Warning: No images found for {period}. Check 'raw_data_path' and 'map_styles_to_periods'.")

    # Balance dataset by copying selected images
    final_stats = defaultdict(int)
    for period in art_periods:
        images_for_period = period_images[period]
        if not images_for_period:
            print(f"  Skipping {period} due to no images found.")
            final_stats[period] = 0
            continue
            
        random.shuffle(images_for_period)
        
        selected_images = images_for_period[:target_per_class]
        
        print(f"  Processing {period}: Copying {len(selected_images)} images (target: {target_per_class})")
        
        dest_period_dir = target_path / period
        for i, src_img_path in enumerate(selected_images):
            dest_img_path = dest_period_dir / src_img_path.name
            try:
                shutil.copy(src_img_path, dest_img_path)
                final_stats[period] += 1
            except Exception as e:
                print(f"    ❌ Error copying {src_img_path} to {dest_img_path}: {e}")

            if (i + 1) % 100 == 0 and len(selected_images) > 100 : # Avoid spamming for small numbers
                print(f"    Copied {i+1}/{len(selected_images)} for {period}...")
        
        print(f"  ✅ {period}: {final_stats[period]} images copied.")

    return final_stats

def main():
    """Main function to create a large balanced dataset from local WikiArt."""
    
    print("🎨 Real WikiArt Dataset Organizer")
    print("=" * 50)
    
    # Configuration
    # target_per_class = 10  # Smallest class count for initial test
    target_per_class = 3615 # Minimum count among selected periods for full dataset
    # target_per_class = 300 # For local CPU testing (approx. 2-3 hours)
    
    # Ensure this path points to the parent directory of style folders (e.g., data/wikiart/Baroque, data/wikiart/Impressionism)
    raw_data_path = "./data/wikiart"  # Updated path
    processed_data_path = "./data/wikiart_real_processed" 
    
    print(f"Source data path: {raw_data_path}")
    print(f"Target processed data path: {processed_data_path}")
    print(f"Target per class: {target_per_class} images")
    
    num_target_periods = len(map_styles_to_periods().values()) # Estimate based on style mapping values
    # A more accurate count would be len of the art_periods list used in organize_wikiart_by_periods
    art_periods_list = ['Renaissance', 'Baroque', 'Romanticism', 'Realism', 
                        'Impressionism', 'Post-Impressionism', 'Modernism', 'Contemporary'] # Added Realism, Ukiyo_e removed
    num_target_periods = len(art_periods_list)

    print(f"Target total dataset size: approx. {target_per_class * num_target_periods} images")
    print()
    
    # Ensure the raw data path exists
    if not Path(raw_data_path).exists():
        print(f"❌ ERROR: Raw data path '{raw_data_path}' does not exist.")
        print(f"Please make sure the WikiArt dataset is downloaded and available at this location.")
        print(f"The expected structure is '{raw_data_path}/<StyleName>/<image_files>'.")
        return

    print(f"✅ Using local real WikiArt dataset from: {raw_data_path}")
    stats = organize_wikiart_by_periods(raw_data_path, processed_data_path, target_per_class)
    
    # Report final statistics
    print(f"\n📊 FINAL PROCESSED DATASET STATISTICS:")
    print("=" * 40)
    total_images = sum(stats.values())
    
    if total_images == 0:
        print("❌ No images were processed. Please check logs for errors.")
        return

    for period, count in stats.items():
        percentage = count / total_images * 100 if total_images > 0 else 0
        print(f"{period:<20}: {count:>4} images ({percentage:>5.1f}%)")
    
    print(f"\n{'Total':<20}: {total_images:>4} images")
    if len(stats) > 0 and total_images > 0:
      avg_per_class = total_images // sum(1 for c in stats.values() if c > 0) if sum(1 for c in stats.values() if c > 0) > 0 else 0
      print(f"{'Average per class (actual)':<25}: {avg_per_class:>4} images")
    
    # Training split estimation
    train_size = int(total_images * 0.7)
    val_size = int(total_images * 0.15)
    test_size = total_images - train_size - val_size
    
    print(f"\n📈 EXPECTED TRAINING SPLITS (approximate based on total):")
    if len(stats) > 0 and total_images > 0:
        avg_train_per_class = train_size // sum(1 for c in stats.values() if c > 0) if sum(1 for c in stats.values() if c > 0) > 0 else 0
        avg_val_per_class = val_size // sum(1 for c in stats.values() if c > 0) if sum(1 for c in stats.values() if c > 0) > 0 else 0
        avg_test_per_class = test_size // sum(1 for c in stats.values() if c > 0) if sum(1 for c in stats.values() if c > 0) > 0 else 0
        print(f"Training:   {train_size:>4} images (approx. {avg_train_per_class:>3} per class)")
        print(f"Validation: {val_size:>4} images (approx. {avg_val_per_class:>3} per class)")  
        print(f"Test:       {test_size:>4} images (approx. {avg_test_per_class:>3} per class)")
    
    print(f"\n✅ Real dataset organized at: {processed_data_path}")
    print(f"🚀 Ready for training with {total_images} images!")
    print(f"\nNext steps:")
    print(f"1. Update 'DataConfig.data_dir' in 'src/config/config.py' to: \"{processed_data_path}\"")
    print(f"2. Run this script to process the data.")
    print(f"3. Then, run your training script.")

if __name__ == "__main__":
    main()
