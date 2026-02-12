# Machine Learning Model Selection Pipeline

A structured and reproducible pipeline for benchmarking and optimizing multiple supervised learning models under a unified evaluation framework.

## Project Goal

Rather than training a single classifier, this project implements a systematic model selection process across multiple algorithms using consistent preprocessing, cross-validation, and incremental optimization.

## Dataset

Wine Quality (Red) – UCI Machine Learning Repository  
1,599 samples · 11 physicochemical features · Multi-class classification (3–8)

## Engineering Workflow

1. Data Diagnostics
   - Class imbalance analysis
   - Distribution skewness & heavy tails
   - Correlation heatmap
   - PCA variance structure

2. Preprocessing Strategy
   - Winsorization for extreme values
   - Power transformation for skewed variables
   - Feature standardization
   - Mutual information feature selection

3. Model Benchmarking
   - KNN
   - SVM (RBF, linear, polynomial kernels)
   - Gradient Boosting Decision Trees

4. Incremental Optimization Chain
   - Hyperparameter tuning (GridSearchCV)
   - Stratified 5-fold cross-validation
   - Controlled one-change-per-step model chain
   - Adoption rule: strict improvement on test accuracy

5. Model Evaluation Metrics
   - Test Accuracy
   - Cross-validated Accuracy
   - Macro-F1 (class-balanced)
   - Weighted-F1
   - Confusion Matrix Analysis

## Final Performance

| Model | Test Accuracy |
|-------|--------------|
| KNN   | 0.672 |
| SVM   | 0.688 |
| GBDT  | 0.697 |

GBDT achieved the strongest performance by leveraging nonlinear feature interactions and engineered ratio features.

## Key Technical Takeaways

- Feature scaling is critical for distance-based models.
- Distribution transformation improves SVM generalization.
- Feature engineering drives most gains in GBDT.
- Accuracy-driven objectives bias predictions toward majority classes.

## Tech Stack

- Python
- scikit-learn
- Pandas / NumPy
- Matplotlib / Seaborn

## Notes

This project emphasizes structured model comparison, reproducibility, and optimization methodology rather than single-model experimentation.
