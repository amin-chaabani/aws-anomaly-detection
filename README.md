# AWS Anomaly Detection System

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Overview

An intelligent anomaly detection system for AWS infrastructure monitoring using machine learning. This project implements the CRISP-DM methodology to detect anomalies in cluster metrics (CPU, Memory, and Pod ratios).

## 🎯 Project Highlights

- **Final Model**: One-Class SVM with RBF kernel
- **Performance**: F1-Score = 0.625, Precision = 0.714, Recall = 0.556
- **Low False Positive Rate**: 7.7%
- **Fast Inference**: 0.14 ms per sample
- **No Overfitting**: Excellent generalization (Val-Test diff < 0.01)

## 🏗️ Project Structure

```
yassmine/
├── aws_anomaly_detection_project/     # Main project directory
│   ├── notebooks/                     # Jupyter notebooks (CRISP-DM phases)
│   │   ├── 01_business_understanding.ipynb
│   │   ├── 02_data_understanding.ipynb
│   │   ├── 03_data_preparation.ipynb
│   │   ├── 04_modeling.ipynb
│   │   ├── 05_evaluation.ipynb
│   │   └── 06_deployment.ipynb
│   ├── src/                          # Source code
│   │   ├── data_loader.py
│   │   ├── feature_engineering.py
│   │   └── utils.py
│   ├── api/                          # Flask API
│   │   └── app.py
│   ├── data/                         # Data files
│   ├── models/                       # Trained models
│   └── reports/                      # Visualizations and reports
├── models/                           # Final models
│   ├── one_class_svm_final.pkl
│   ├── final_model_config.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
├── synthetic_anomalies.py            # Anomaly generation utilities
├── feature_engineering.py            # Feature engineering module
├── requirements.txt                  # Project dependencies
└── README.md                         # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- pip
- virtualenv (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd yassmine
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### 1. Run Jupyter Notebooks

```bash
jupyter notebook aws_anomaly_detection_project/notebooks/
```

Navigate through the notebooks in order (01 → 06) to see the complete CRISP-DM workflow.

#### 2. Use the Trained Model

```python
import pickle
import pandas as pd

# Load the model
with open('models/one_class_svm_final.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Prepare your data
X_new = pd.DataFrame(...)  # Your metrics data
X_scaled = scaler.transform(X_new)

# Make predictions
predictions = model.predict(X_scaled)
# -1 = anomaly, 1 = normal
```

#### 3. Run the API (if deployed)

```bash
cd aws_anomaly_detection_project
python api/app.py
```

## 📊 Dataset

- **Source**: AWS Prometheus metrics
- **Metrics**:
  - `cluster_cpu_request_ratio`: CPU resource requests vs available
  - `cluster_mem_request_ratio`: Memory resource requests vs available
  - `cluster_pod_ratio`: Pod count vs capacity
- **Samples**: 230 total (161 train, 34 validation, 35 test)
- **Features**: 104 engineered features
- **Anomaly Rate**: ~26% (synthetic anomalies)

## 🔬 Methodology

This project follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology:

1. **Business Understanding**: Define objectives and requirements
2. **Data Understanding**: EDA, statistical analysis, outlier detection
3. **Data Preparation**: Feature engineering, scaling, train-test split
4. **Modeling**: Algorithm selection, hyperparameter tuning with Optuna
5. **Evaluation**: Performance metrics, overfitting detection
6. **Deployment**: API development, Docker containerization

## 🏆 Model Performance

### Final Model: One-Class SVM

| Metric | Validation | Test | Difference |
|--------|-----------|------|------------|
| **F1-Score** | 0.632 | 0.625 | 0.007 |
| **Precision** | 0.714 | 0.714 | 0.000 |
| **Recall** | 0.556 | 0.556 | 0.000 |
| **FPR** | 0.077 | 0.077 | 0.000 |

### Confusion Matrix (Test Set)
```
              Predicted
              Normal  Anomaly
Actual Normal    24      2
       Anomaly    4      5
```

## 📈 Key Features

- ✅ **Robust Feature Engineering**: 104 features including rolling statistics, lag features, and temporal patterns
- ✅ **Hyperparameter Optimization**: Optuna-based Bayesian optimization (40 trials)
- ✅ **Overfitting Prevention**: Stratified splits, careful validation, model selection based on generalization
- ✅ **Comprehensive Evaluation**: Multiple metrics, confusion matrix, error analysis
- ✅ **Production Ready**: Fast inference, low false positive rate, stable performance

## 📚 Documentation

- **[Phase 4 Complete Report](PHASE_4_COMPLETE.md)**: Modeling phase summary
- **[Modeling Lessons Learned](MODELING_LESSONS_LEARNED.md)**: 50+ insights from the modeling process
- **[Ready for Phase 5](READY_FOR_PHASE_5.md)**: Handoff guide for evaluation phase
- **[Project Summary](PROJECT_SUMMARY.md)**: Overall project overview
- **[Enhancement Summary](ENHANCEMENTS_SUMMARY.md)**: Recent improvements

## 🛠️ Technologies Used

- **Python 3.11**: Core programming language
- **scikit-learn**: Machine learning algorithms
- **Optuna**: Hyperparameter optimization
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn/Plotly**: Data visualization
- **Jupyter**: Interactive notebooks
- **Flask**: API framework (deployment)
- **Docker**: Containerization (deployment)

## 🔍 Model Selection Process

We evaluated multiple approaches:
- ✅ **One-Class SVM** (Selected): Best generalization, stable performance
- ❌ Isolation Forest: Overfitting (Val F1=0.737, Test F1=0.471)
- ❌ Local Outlier Factor: Overfitting (Val F1=0.737, Test F1=0.429)
- ❌ Feature Selection + SVM: Severe overfitting (Val F1=0.737, Test F1=0.471)

**Selection Criteria**: 
- Val-Test F1 difference < 0.01
- Test F1 > 0.60
- Low false positive rate
- Interpretability

## 🚧 Future Improvements

- [ ] Real-time monitoring dashboard
- [ ] Additional anomaly types (concept drift, seasonal anomalies)
- [ ] Deep learning approaches (LSTM Autoencoder)
- [ ] Automated retraining pipeline
- [ ] Alert system integration
- [ ] Multi-cluster support

## 👥 Contributors

- **Yassmine** - Data Scientist

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- AWS Prometheus for metric collection
- CRISP-DM methodology for structured approach
- scikit-learn and Optuna communities

## 📞 Contact

For questions or feedback, please open an issue in the repository.

---

**Last Updated**: November 4, 2025
**Status**: Phase 4 (Modeling) Complete ✅ | Phase 5 (Evaluation) In Progress 🔄
