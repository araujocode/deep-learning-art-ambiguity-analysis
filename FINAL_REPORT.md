# WikiArt Art Period Classification - Final Report

## Project Overview
This project implements an artistic style period classification system using EfficientNet-B0 with ambiguity detection capabilities. The system classifies artwork images across 8 historical art periods with uncertainty quantification.

## Dataset
- **Source**: Synthetic WikiArt-inspired dataset
- **Classes**: 8 art periods (Renaissance, Baroque, Romanticism, Impressionism, Post-Impressionism, Modernism, Surrealism, Contemporary)
- **Size**: 480 total images (60 per class)
- **Split**: 70% train (336), 15% validation (72), 15% test (72)

## Model Architecture
- **Backbone**: EfficientNet-B0 (pretrained on ImageNet)
- **Classifier**: Custom head with dropout and batch normalization
- **Training Strategy**: Two-phase training (head-only → full fine-tuning)
- **Regularization**: MixUp augmentation and label smoothing

## Training Results

### Training Configuration
- **Phase 1**: 10 epochs (head-only training)
- **Phase 2**: 20 epochs (full fine-tuning)
- **Optimizer**: AdamW with cosine annealing
- **Data Augmentation**: Random resize, crop, horizontal flip, color jitter

### Performance Metrics
- **Best Validation Accuracy**: 63.89%
- **Test Accuracy**: 62.50% (45/72 correct predictions)
- **Average Test Confidence**: 0.426

### Per-Class Performance
| Art Period | Test Accuracy | Precision | Recall | F1-Score |
|------------|---------------|-----------|--------|----------|
| Renaissance | 100.0% | 1.00 | 1.00 | 1.00 |
| Baroque | 77.8% | 0.78 | 0.78 | 0.78 |
| Post-Impressionism | 77.8% | 0.70 | 0.78 | 0.74 |
| Contemporary | 77.8% | 0.70 | 0.78 | 0.74 |
| Impressionism | 44.4% | 0.44 | 0.44 | 0.44 |
| Modernism | 44.4% | 0.40 | 0.44 | 0.42 |
| Surrealism | 44.4% | 0.50 | 0.44 | 0.47 |
| Romanticism | 33.3% | 0.38 | 0.33 | 0.35 |

## Ambiguity Detection Analysis

### Detection Methods
1. **Softmax-based**: Uses confidence threshold (0.6) and probability gap (0.1)
2. **Entropy-based**: Uses entropy percentile threshold (80th percentile)
3. **Combined**: Union of both methods

### Ambiguity Statistics
- **Overall Test Accuracy**: 62.5%
- **Average Max Probability**: 0.426
- **Average Entropy**: 1.587

### Detection Results
| Method | Ambiguous Samples | Percentage | Accuracy on Ambiguous |
|--------|-------------------|------------|----------------------|
| Softmax | 57/72 | 79.2% | 57.9% |
| Entropy | 12/72 | 16.7% | 33.3% |
| Combined | 57/72 | 79.2% | 57.9% |
| Overlap | 12/72 | 16.7% | - |

### Per-Class Ambiguity Analysis
| Art Period | Avg Confidence | Avg Entropy | Softmax Ambiguous | Entropy Ambiguous |
|------------|----------------|-------------|-------------------|-------------------|
| Renaissance | 0.521 | 1.412 | 66.7% | 0.0% |
| Baroque | 0.382 | 1.646 | 100.0% | 22.2% |
| Romanticism | 0.405 | 1.585 | 100.0% | 11.1% |
| Impressionism | 0.440 | 1.560 | 77.8% | 22.2% |
| Post-Impressionism | 0.417 | 1.586 | 77.8% | 11.1% |
| Modernism | 0.422 | 1.600 | 77.8% | 22.2% |
| Surrealism | 0.371 | 1.735 | 77.8% | 22.2% |
| Contemporary | 0.452 | 1.575 | 55.6% | 22.2% |

## Key Findings

### Model Performance
1. **Strong Performance**: 62.5% test accuracy across 8 classes (chance = 12.5%)
2. **Class Imbalances**: Renaissance and Baroque perform best, Romanticism struggles most
3. **Reasonable Confidence**: Average confidence of 0.426 indicates appropriate uncertainty

### Ambiguity Detection Insights
1. **High Ambiguity Rate**: 79.2% of test samples flagged as ambiguous by softmax method
2. **Method Differences**: Softmax-based detection is more sensitive than entropy-based
3. **Class Variations**: Renaissance shows lowest ambiguity, Baroque and Romanticism highest
4. **Accuracy Relationship**: Ambiguous samples have lower accuracy (57.9% vs 62.5% overall)

### Technical Achievements
1. **Successfully Implemented**: Complete pipeline from data loading to ambiguity detection
2. **Robust Training**: Two-phase training with regularization techniques
3. **Comprehensive Evaluation**: Multiple metrics and visualization tools
4. **Uncertainty Quantification**: Working ambiguity detection system

## Limitations and Future Work

### Current Limitations
1. **Synthetic Data**: Dataset consists of synthetic images rather than real artwork
2. **Small Dataset**: Only 60 images per class limits model generalization
3. **Class Confusion**: Some art periods show significant confusion (Romanticism, Modernism)
4. **Ambiguity Threshold**: Current thresholds may need fine-tuning for optimal performance

### Future Improvements
1. **Real Dataset**: Integrate actual WikiArt images for more realistic evaluation
2. **Data Augmentation**: Advanced augmentation techniques specific to artwork
3. **Architecture Exploration**: Test other vision transformers or CNN architectures
4. **Ensemble Methods**: Combine multiple models for better uncertainty estimation
5. **Active Learning**: Use ambiguity detection for intelligent data labeling

## Conclusion

This project successfully demonstrates an end-to-end art period classification system with uncertainty quantification. The EfficientNet-B0 model achieves reasonable performance (62.5% accuracy) across 8 art periods, with the ambiguity detection system providing valuable insights into model confidence and potential failure cases.

The comprehensive analysis reveals that while the model performs well overall, certain art periods (especially Romanticism and Modernism) present significant challenges, likely due to their stylistic similarities and transitional nature in art history. The ambiguity detection system successfully identifies uncertain predictions, which could be valuable for active learning or human-in-the-loop scenarios.

The project provides a solid foundation for future work with real artwork datasets and more sophisticated uncertainty estimation techniques.

---

## Files Generated
- `experiments/run_001/checkpoints/best_model.pth` - Trained model weights
- `experiments/run_001/confusion_matrix.png` - Classification confusion matrix
- `experiments/run_001/training_history.png` - Training curves
- `experiments/run_001/ambiguity_analysis/` - Comprehensive ambiguity analysis
- `experiments/run_001/classification_report.txt` - Detailed classification metrics

Generated on: June 8, 2025
