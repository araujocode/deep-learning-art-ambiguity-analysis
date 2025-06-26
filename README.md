# Artistic Style Period Classification Pipeline with Ambiguity Detection

A comprehensive, modular implementation of an artistic style period classification system using EfficientNet, featuring stylistic ambiguity detection and interpretability analysis.

## Project Overview

This project implements a machine learning pipeline for classifying artworks into historical periods (Renaissance, Baroque, Romanticism, Realism, Impressionism, Post-Impressionism, Modernism, Contemporary) with a unique focus on detecting stylistically ambiguous or transitional artworks.

### Key Features

- **Modular Object-Oriented Design**: Clean, maintainable code structure
- **Two-Phase Fine-Tuning**: Progressive unfreezing strategy for optimal transfer learning
- **Ambiguity Detection**: Softmax-based and entropy-based uncertainty quantification
- **Interpretability Tools**: Grad-CAM visualizations and t-SNE embeddings
- **Model Calibration**: Temperature scaling for reliable probability estimates
- **Comprehensive Evaluation**: Multiple metrics and visualization tools

## Project Structure

```
Code/
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── config.py              # Configuration management
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py             # Dataset classes and data loading
│   │   └── data_manager.py        # Data management utilities
│   ├── models/
│   │   ├── __init__.py
│   │   └── efficientnet_classifier.py  # EfficientNet-based classifier
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py             # Training manager with two-phase fine-tuning
│   │   └── regularization.py      # MixUp and other regularization techniques
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py           # Model evaluation and metrics
│   │   └── calibration.py         # Model calibration utilities
│   ├── ambiguity/
│   │   ├── __init__.py
│   │   ├── detector.py            # Ambiguity detection algorithms
│   │   └── analyzer.py            # Ambiguity analysis tools
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── gradcam.py             # Grad-CAM visualization
│   │   ├── tsne_viz.py            # t-SNE embeddings
│   │   └── plotting.py            # General plotting utilities
│   └── utils/
│       ├── __init__.py
│       ├── logging.py             # Logging utilities
│       └── reproducibility.py     # Seed setting and reproducibility
├── scripts/
│   ├── train.py                   # Main training script (includes evaluation and Grad-CAM)
│   ├── evaluate.py                # Standalone evaluation script
│   ├── analyze_ambiguity.py       # Standalone ambiguity analysis script
│   └── demo.py                    # Streamlit demo application
├── notebooks/
│   ├── data_exploration.ipynb     # Data exploration and analysis
│   ├── model_analysis.ipynb       # Model performance analysis
│   └── visualization_gallery.ipynb  # Visualization examples
├── demo/
│   ├── app.py                     # Streamlit demo application
│   ├── model_utils.py             # Demo utilities
│   └── requirements.txt           # Demo requirements
├── data/                          # Data directory
│   ├── wikiart_raw/               # Raw WikiArt dataset downloaded from Kaggle
│   └── wikiart_real_processed/    # Processed and balanced dataset for training
├── experiments/                   # Experiment outputs (models, logs, metrics, visualizations)
├── artifacts/                     # Generated visualizations and results (if used separately)
├── requirements.txt               # Main project requirements
├── environment.yml                # Conda environment specification
├── create_large_dataset.py        # Script to process and balance the raw WikiArt dataset
└── README.md                      # This file
```

## Installation

1. **Clone the repository:**

```bash
git clone <repository-url>
cd "Machine Learning/Trabalho Final/Code"
```

2. **Create conda environment:**

```bash
conda env create -f environment.yml
conda activate art-classification
```

3. **Alternative: Install with pip:**

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Data Preparation

1. **Download the WikiArt dataset:**
    - Go to [Kaggle: WikiArt Dataset](https://www.kaggle.com/datasets/steubk/wikiart).
    - Download the dataset.
2. **Extract the dataset:**
    - Create a directory named `data/wikiart_raw` in the root of this project.
    - Extract the contents of the downloaded archive (the various style folders like `Abstract_Expressionism`, `Baroque`, etc.) directly into the `data/wikiart_raw` directory.
3. **Process and balance the dataset:**
    - Run the script to organize and balance the dataset for the defined art periods:

    ```bash
    python create_large_dataset.py
    ```

    This will populate the `data/wikiart_real_processed` directory, which will be used for training.

### 2. Training

Train the model using the two-phase fine-tuning approach. The script will automatically:

- Evaluate the best model on the test set.
- Save a classification report and confusion matrix.
- Generate Grad-CAM visualizations for a few examples.
- Optionally, run ambiguity analysis if the `--run_ambiguity_analysis` flag is provided.
All outputs are saved in the specified output directory.

```bash
python scripts/train.py \\
    --data_dir ./data/wikiart_real_processed \\
    --output_dir ./experiments/run_001 \\
    --backbone efficientnet_b0 \\
    --batch_size 32 \\
    --phase1_epochs 10 \\
    --phase2_epochs 40 \\
    --use_mixup \\
    --run_ambiguity_analysis # Add this flag to run ambiguity analysis
#   --softmax_pmax_threshold 0.6 # Optional: customize ambiguity parameters
#   --softmax_gap_threshold 0.1
#   --entropy_percentile_threshold 80.0
```

- Adjust `output_dir`, `batch_size`, and `epochs` as needed, especially for GPU training.

- Check the `experiments/run_001` (or your specified output directory) for saved models, logs, evaluation metrics, and Grad-CAM images.

### 3. Evaluation (Standalone)

While the training script performs evaluation, you can also run a standalone evaluation on a trained model:

```bash
python scripts/evaluate.py \
    --model_path ./experiments/run_001/checkpoints/best_model.pth \
    --data_dir ./data/wikiart_real_processed \
    --output_dir ./experiments/run_001/evaluation
```

### 4. Ambiguity Analysis (Standalone)

Ambiguity analysis can be run automatically after training by using the `--run_ambiguity_analysis` flag with `scripts/train.py` (see Training section).
For standalone, more detailed ambiguity analysis on an existing model, or to re-generate visualizations for specific images without retraining, you can use:

```bash
python scripts/analyze_ambiguity.py \\
    --model_path ./experiments/run_001/checkpoints/best_model.pth \\
    --data_dir ./data/wikiart_real_processed \\
    --output_dir ./experiments/run_001/ambiguity_analysis \\
    --generate_gradcam \\
    --generate_tsne
```

### 5. Demo

Run the interactive Streamlit demo:

```bash
cd demo
streamlit run app.py
```

## Architecture Overview

### Core Components

1. **EfficientNetClassifier**: Modular classifier with configurable dropout and feature extraction
2. **Trainer**: Two-phase fine-tuning with comprehensive logging and checkpointing
3. **AmbiguityDetector**: Multiple uncertainty quantification methods
4. **Visualizer**: Grad-CAM and t-SNE visualization tools
5. **Evaluator**: Comprehensive model evaluation with calibration metrics

### Training Strategy

- **Phase 1**: Freeze backbone, train classifier head (e.g., 10 epochs, lr=1e-3)
- **Phase 2**: Unfreeze last block, fine-tune (e.g., 20 epochs, lr=1e-4)
- **Regularization**: Label smoothing (0.1), MixUp (α=0.2), Dropout (0.3)
- **Optimization**: AdamW + CosineAnnealingLR

### Ambiguity Detection

- **Softmax-based**: Flag if max_prob < 0.5 OR top1-top2 gap < 0.1
- **Entropy-based**: Flag if entropy > 90th percentile threshold
- **Manual validation**: Human-in-the-loop verification for threshold tuning

## Configuration

The project uses a comprehensive configuration system. Main configuration options:

```python
# Example configuration (see src/config/config.py for details)
config = ProjectConfig(
    data=DataConfig(
        data_dir="./data/wikiart_real_processed", # Updated
        batch_size=32,
        art_periods=["Renaissance", "Baroque", "Romanticism", "Realism",
                    "Impressionism", "Post-Impressionism", 
                    "Modernism", "Contemporary"] # Updated art periods
    ),
    model=ModelConfig(
        backbone="efficientnet_b0",
        dropout_rate=0.3,
        label_smoothing=0.1,
        use_mixup=True
    ),
    training=TrainingConfig(
        device="cuda",
        output_dir="./experiments",
        early_stopping_patience=5
    )
)
```

## Expected Performance

Target metrics based on the 30-day roadmap:

- **Top-1 Accuracy**: ≥70% on test set
- **Expected Calibration Error**: <0.08 after temperature scaling
- **Ambiguity Detection Precision**: ≥80% on manually curated set
- **Training Time**: ≤10 minutes/epoch on 8GB GPU

## Data Requirements

### Art Periods and Minimum Images per Class

- Renaissance: ≥300 images
- Baroque: ≥300 images  
- Romanticism: ≥300 images
- Realism: ≥300 images
- Impressionism: ≥500 images
- Post-Impressionism: ≥400 images
- Modernism: ≥400 images
- Contemporary: ≥300 images

### Data Sources

- Primary: WikiArt dataset via Kaggle
- Alternative: Painter-by-numbers dataset
- Manual curation for transitional artwork validation set

## Key Results

Upon completion, the project will generate:

1. **Model Checkpoints**: Best performing model weights
2. **Training Logs**: Comprehensive training history and metrics
3. **Visualizations**:
   - Grad-CAM heatmaps for ambiguous vs. clear classifications
   - t-SNE embeddings showing class clusters and transitional zones
   - Confusion matrices and calibration plots
4. **Analysis Reports**: Ambiguity detection performance and interpretability insights
5. **Academic Paper**: 4-6 page workshop-style paper
6. **Demo Application**: Interactive Streamlit app with live predictions

## Contributing

This project follows object-oriented design principles with clear separation of concerns. When contributing:

1. Follow the established module structure
2. Add comprehensive docstrings
3. Include unit tests for new functionality
4. Update configuration classes for new parameters
5. Maintain backward compatibility

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{art-classification-ambiguity-2024,
    title={Artistic Style Period Classification with Ambiguity Detection},
    author={Bruno},
    year={2024},
    institution={UNESP - Universidade Estadual Paulista}
}
```

## License

This project is developed for academic purposes as part of a Machine Learning course at UNESP.

## Acknowledgments

- UNESP Machine Learning Course
- WikiArt dataset contributors  
- PyTorch and timm library developers
- Grad-CAM and t-SNE algorithm authors

---

For detailed implementation notes and academic context, see the original 30-day roadmap document included in this repository.

## Command-Line Interface (CLI) Options

The main training script `scripts/train.py` supports the following CLI arguments:

**Data arguments:**
- `--data_dir` (str, required): Path to the dataset directory
- `--output_dir` (str, default: ./experiments/run_001): Output directory for experiments

**Model arguments:**
- `--backbone` (str, default: efficientnet_b0): EfficientNet backbone to use (choices: efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4)
- `--dropout_rate` (float, default: 0.3): Dropout rate before classifier

**Training arguments:**
- `--batch_size` (int, default: 32): Training batch size
- `--phase1_epochs` (int, default: 10): Number of epochs for phase 1 training
- `--phase2_epochs` (int, default: 40): Number of epochs for phase 2 training
- `--stage1_epochs` (int, default: 10): Epochs for phase 2 stage 1 (last 2 blocks, progressive unfreezing)
- `--stage2_epochs` (int, default: 10): Epochs for phase 2 stage 2 (last 4 blocks, progressive unfreezing)
- `--stage3_epochs` (int, default: 20): Epochs for phase 2 stage 3 (full unfreeze, progressive unfreezing)
- `--phase1_lr` (float, default: 1e-3): Learning rate for phase 1
- `--phase2_lr` (float, default: 1e-4): Learning rate for phase 2
- `--weight_decay` (float, default: 1e-2): Weight decay for optimizer
- `--early_stopping_patience_phase1` (int, default: 5): Early stopping patience for phase 1
- `--early_stopping_patience_phase2` (int, default: 16): Early stopping patience for phase 2

**Regularization & Augmentation:**
- `--label_smoothing` (float, default: 0.1): Label smoothing factor
- `--use_mixup` (flag): Use MixUp augmentation
- `--mixup_alpha` (float, default: 0.2): MixUp alpha parameter
- `--use_cutmix` (flag): Use CutMix augmentation
- `--cutmix_alpha` (float, default: 1.0): CutMix alpha parameter
- `--use_focal_loss` (flag): Use Focal Loss instead of CrossEntropyLoss
- `--use_randaugment` (flag): Use RandAugment for training augmentations
- `--use_autoaugment` (flag): Use AutoAugment for training augmentations
- `--use_random_erasing` (flag): Use Random Erasing for training augmentations

**Fine-tuning schedule:**
- `--progressive_unfreezing` (flag): Use progressive unfreezing schedule in phase 2 (recommended)

**Ambiguity Analysis:**
- `--ambiguity_scores_path` (str): Path to ambiguity scores file (npy, pt, or csv)
- `--run_ambiguity_analysis` (flag): Run ambiguity analysis after training
- `--softmax_pmax_threshold` (float, default: 0.6): Ambiguity: Softmax p_max threshold
- `--softmax_gap_threshold` (float, default: 0.1): Ambiguity: Softmax probability gap threshold
- `--entropy_percentile_threshold` (float, default: 80.0): Ambiguity: Entropy percentile threshold for calibration

**Other arguments:**
- `--seed` (int, default: 42): Random seed for reproducibility
- `--num_workers` (int, default: 4): Number of workers for data loading
- `--device` (str, default: auto): Device to use (cuda/cpu/auto)

---

Example usage:
```bash
python scripts/train.py \
  --data_dir ./data/wikiart_real_processed \
  --output_dir ./experiments/my_run \
  --backbone efficientnet_b2 \
  --batch_size 32 \
  --use_mixup --use_randaugment --use_random_erasing \
  --progressive_unfreezing --stage1_epochs 10 --stage2_epochs 10 --stage3_epochs 20
```

---
