# Artistic Style Period Classification Pipeline with Ambiguity Detection

A comprehensive, modular implementation of an artistic style period classification system using EfficientNet-B0, featuring stylistic ambiguity detection and interpretability analysis.

## Project Overview

This project implements a machine learning pipeline for classifying artworks into historical periods (Renaissance, Baroque, Romanticism, Impressionism, Post-Impressionism, Modernism, Surrealism, Contemporary) with a unique focus on detecting stylistically ambiguous or transitional artworks.

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
│   ├── train.py                   # Main training script
│   ├── evaluate.py                # Evaluation script
│   ├── analyze_ambiguity.py       # Ambiguity analysis script
│   └── demo.py                    # Streamlit demo application
├── notebooks/
│   ├── data_exploration.ipynb     # Data exploration and analysis
│   ├── model_analysis.ipynb       # Model performance analysis
│   └── visualization_gallery.ipynb  # Visualization examples
├── demo/
│   ├── app.py                     # Streamlit demo application
│   ├── model_utils.py             # Demo utilities
│   └── requirements.txt           # Demo requirements
├── data/                          # Data directory (to be created)
├── experiments/                   # Experiment outputs
├── artifacts/                     # Generated visualizations and results
├── requirements.txt               # Main project requirements
├── environment.yml               # Conda environment specification
└── README.md                     # This file
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

Download and organize the WikiArt dataset:

```bash
# Create data directory structure
mkdir -p data/wikiart_processed/{Renaissance,Baroque,Romanticism,Impressionism,Post-Impressionism,Modernism,Surrealism,Contemporary}

# Download data (manual process - see data section for details)
# Organize images into respective art period folders
```

### 2. Training

Train the model using the two-phase fine-tuning approach:

```bash
python scripts/train.py \
    --data_dir ./data/wikiart_processed \
    --output_dir ./experiments/run_001 \
    --backbone efficientnet_b0 \
    --batch_size 32 \
    --phase1_epochs 3 \
    --phase2_epochs 7 \
    --use_mixup
```

### 3. Evaluation

Evaluate the trained model:

```bash
python scripts/evaluate.py \
    --model_path ./experiments/run_001/checkpoints/best_model.pth \
    --data_dir ./data/wikiart_processed \
    --output_dir ./experiments/run_001/evaluation
```

### 4. Ambiguity Analysis

Analyze stylistic ambiguity and generate visualizations:

```bash
python scripts/analyze_ambiguity.py \
    --model_path ./experiments/run_001/checkpoints/best_model.pth \
    --data_dir ./data/wikiart_processed \
    --output_dir ./experiments/run_001/ambiguity_analysis \
    --generate_gradcam \
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

- **Phase 1**: Freeze backbone, train classifier head (3 epochs, lr=1e-3)
- **Phase 2**: Unfreeze last block, fine-tune (7 epochs, lr=1e-4)
- **Regularization**: Label smoothing (0.1), MixUp (α=0.2), Dropout (0.3)
- **Optimization**: AdamW + CosineAnnealingLR

### Ambiguity Detection

- **Softmax-based**: Flag if max_prob < 0.5 OR top1-top2 gap < 0.1
- **Entropy-based**: Flag if entropy > 90th percentile threshold
- **Manual validation**: Human-in-the-loop verification for threshold tuning

## Configuration

The project uses a comprehensive configuration system. Main configuration options:

```python
# Example configuration
config = ProjectConfig(
    data=DataConfig(
        data_dir="./data/wikiart_processed",
        batch_size=32,
        art_periods=["Renaissance", "Baroque", "Romanticism", 
                    "Impressionism", "Post-Impressionism", 
                    "Modernism", "Surrealism", "Contemporary"]
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
- Impressionism: ≥500 images
- Post-Impressionism: ≥400 images
- Modernism: ≥400 images
- Surrealism: ≥300 images
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
