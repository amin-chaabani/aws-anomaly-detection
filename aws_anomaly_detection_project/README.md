# AWS Cluster Anomaly Detection System

**Machine Learning solution for real-time AWS cluster resource monitoring and anomaly detection**

---

## 📋 Overview

Production-ready anomaly detection system designed to monitor AWS Kubernetes cluster metrics and detect resource anomalies in real-time. Built following CRISP-DM methodology with comprehensive documentation and deployment capabilities.

### Key Features
- ✅ Real-time anomaly detection using Machine Learning
- ✅ Multiple algorithms (Isolation Forest, One-Class SVM, LOF)
- ✅ Advanced feature engineering (350+ features)
- ✅ REST API for integration
- ✅ Email alert system via Alertmanager
- ✅ Docker containerization
- ✅ Complete CRISP-DM documentation

---

## 🏗️ Project Structure

```
aws_anomaly_detection_project/
├── notebooks/              # CRISP-DM Jupyter Notebooks (Phase 1-6)
├── src/                    # Reusable source code modules
├── api/                    # Flask REST API
├── models/                 # Trained ML models
├── data/                   # Cluster metrics data
├── reports/                # Analysis reports and visualizations
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker image configuration
├── docker-compose.yml     # Multi-container setup
└── alertmanager.yml       # Email notification configuration
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- 4GB RAM minimum

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd aws_anomaly_detection_project
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure email alerts** (Optional)
- Edit `alertmanager.yml`
- Add your Gmail App Password
- Configure recipient email addresses

4. **Run with Docker**
```bash
docker-compose up -d --build
```

---

## 📓 Notebooks (CRISP-DM Workflow)

Execute notebooks in order:

1. **01_business_understanding.ipynb** - Business objectives and success criteria
2. **02_data_understanding.ipynb** - Exploratory data analysis with visualizations
3. **03_data_preparation.ipynb** - Feature engineering (350+ features)
4. **04_modeling.ipynb** - Model training with hyperparameter optimization
5. **05_evaluation.ipynb** - Performance evaluation and metrics
6. **06_deployment.ipynb** - API deployment guide

### Run Notebooks
```bash
jupyter notebook
```

---

## 🌐 API Usage

### Start API
```bash
# Development
python api/app.py

# Production (Docker)
docker-compose up -d
```

### Endpoints

#### Health Check
```bash
GET http://localhost:5000/health
```

#### Model Information
```bash
GET http://localhost:5000/model_info
```

#### Single Prediction
```bash
POST http://localhost:5000/predict
Content-Type: application/json

{
  "cluster_cpu_request_ratio": 0.75,
  "cluster_mem_request_ratio": 0.68,
  "cluster_pod_ratio": 0.82
}
```

#### Batch Predictions
```bash
POST http://localhost:5000/batch_predict
Content-Type: application/json

{
  "samples": [
    {
      "cluster_cpu_request_ratio": 0.75,
      "cluster_mem_request_ratio": 0.68,
      "cluster_pod_ratio": 0.82
    }
  ]
}
```

---

## 🔧 Configuration

### Email Alerts (Alertmanager)

1. Generate Gmail App Password:
   - https://myaccount.google.com/apppasswords

2. Update `alertmanager.yml`:
```yaml
smtp_auth_password: 'your-app-password-here'
```

3. Configure recipients:
```yaml
to: 'your-email@domain.com'
```

---

## 📊 Model Performance

| Model | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Isolation Forest | 87.3% | 83.7% | 85.5% |
| One-Class SVM | 82.1% | 86.4% | 84.2% |
| LOF | 79.5% | 81.2% | 80.3% |
| **Ensemble** | **89.2%** | **85.8%** | **87.4%** |

**False Positive Rate:** 2.9%  
**Detection Latency:** < 5 minutes  
**API Response Time:** ~50ms

---

## 🐳 Docker Deployment

### Build and Run
```bash
# Build image
docker-compose build

# Start services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Services
- **API**: http://localhost:5000
- **Alertmanager**: http://localhost:9093

---

## 📦 Dependencies

Core libraries:
- `numpy`, `pandas` - Data manipulation
- `scikit-learn` - Machine learning
- `flask` - REST API
- `optuna` - Hyperparameter optimization
- `matplotlib`, `seaborn`, `plotly` - Visualization
- `jupyter` - Notebooks

See `requirements.txt` for complete list.

---

## 🧪 Testing

Run API tests:
```bash
python test_api.py
```

---

## 📈 Monitoring

### Metrics Tracked
- **CPU Request Ratio**: Cluster CPU utilization
- **Memory Request Ratio**: Cluster memory utilization  
- **Pod Ratio**: Pod scheduling efficiency

### Alert Levels
- **Warning**: Anomaly detected (1-hour repeat)
- **Critical**: Severe anomaly (30-minute repeat)
- **Info**: Informational alerts (24-hour repeat)

---

## 🔒 Security

- Non-root user in Docker container
- Environment variable configuration
- TLS/SSL for email (Gmail SMTP)
- API CORS protection

---

## 📚 Documentation

All notebooks include:
- Clear markdown explanations
- Commented code
- Visualizations
- Key findings and insights

---

## 🤝 Contributing

This project follows:
- **PEP 8** Python style guide
- **CRISP-DM** methodology
- **Semantic versioning**
- **Google-style** docstrings

---

## 📄 License

Internal project - All rights reserved

---

## 👥 Team

Data Science Team  
**Contact**: [Your contact information]

---

## 📞 Support

For questions or issues:
1. Review notebook documentation (notebooks/)
2. Check API logs: `docker-compose logs`
3. Verify configuration files
4. Contact the development team

---

## ✅ Project Status

**Version**: 1.0.0  
**Status**: Production-Ready ✅  
**Last Updated**: November 2025

### Completed Deliverables
- ✅ Complete CRISP-DM notebooks (6 phases)
- ✅ Feature engineering pipeline (350+ features)
- ✅ Trained ML models with optimization
- ✅ Flask REST API with 5 endpoints
- ✅ Docker deployment configuration
- ✅ Email alert system
- ✅ Comprehensive documentation

---

**Built with ❤️ following industry best practices**
