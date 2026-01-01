# Banking Fraud Detection with PySpark and Machine Learning

## Overview

This project applies anomaly detection techniques, spanning traditional machine learning and deep learning approaches, to identify fraudulent transactions in large-scale banking datasets. Leveraging PySpark for distributed data processing and integrating models from scikit-learn and deep learning frameworks, the goal is to build a production-ready system that detects anomalous patterns and outliers indicative of fraud. The solution is designed to support business stakeholders in strengthening AML controls and improving fraud detection for regulatory compliance.

### Project Goals

This project is developed as an end-to-end machine learning system with the following objectives:

- **Distributed Data Processing**: Design scalable fraud detection pipelines using PySpark to efficiently process large volumes of banking transaction data.
- **Model Experimentation**: Experiment with and compare multiple anomaly detection models while tracking results using MLflow.
- **Production Deployment**: Deploy a production-grade inference pipeline using FastAPI, containerised and orchestrated with Kubernetes/Fargate.
- **Model Monitoring**: Implement monitoring pipelines to detect performance degradation, data drift, and model decay in production environments.

### Key Features

- **Anomaly Detection Algorithms**: Supports a range of techniques, including:
  - Isolation Forest
  - One-Class SVM
  - Local Outlier Factor (LOF)
  - DBSCAN clustering
  - Statistical-based methods
- **Banking Domain Focus**: Tailored for anomaly detection in banking transaction and customer datasets.
- **End-to-End MLOps**: Implements a complete MLOps lifecycle, from model experimentation and deployment to monitoring and maintenance.

## Project Structure

```
anomaly-detection/
├── data/                    # Dataset directory
│   └── ADBenchmarks-anomaly-detection-datasets/
├── notebooks/              # Jupyter notebooks for analysis
├── src/                    # Source code
│   ├── config.py          # Configuration and path management
│   └── main.py            # Main application entry point
├── pyproject.toml         # Project dependencies and configuration
└── README.md
```

## Getting Started

### Prerequisites

This project requires the following tools:

- **[uv](https://github.com/astral-sh/uv)**: A fast, reliable Python package manager and modern alternative to pip, used for dependency management and virtual environment creation.
- **[pre-commit](https://pre-commit.com/)**: A framework for managing git hooks to enforce code quality, formatting standards, and commit message conventions.

### Installation

#### 1. Install uv

```bash
# MacOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

#### 2. Clone and Setup Project

```bash
# Clone repository
git clone https://github.com/yourusername/anomaly-detection.git
cd anomaly-detection

# Sync dependencies (automatically creates virtual environment)
uv sync

# Install project in editable mode
uv pip install -e .

# Activate virtual environment
source .venv/bin/activate # MacOS / Linux
.venv\Scripts\activate # Windows
```

#### 3. Install Pre-commit Hooks

```bash
# Install pre-commit hooks for code quality and commit message validation
pre-commit install --install-hooks
pre-commit install --hook-type commit-msg
```

### Running the Project

#### Launch Jupyter Notebook
```bash
# Launch Jupyter Notebook
jupyter notebook
```

## Dependencies

Core libraries:
- **PySpark**: Distributed data processing
- **scikit-learn**: Machine learning algorithms
- **MLflow**: Experiment tracking and model registry
- **FastAPI**: API framework for model deployment
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scipy**: Scientific computing
- **matplotlib/seaborn**: Data visualisation
- **loguru**: Logging

See `pyproject.toml` for complete list of dependencies.

## Dataset Information

This project uses banking marketing campaign data in ARFF format. The dataset contains categorical features related to:
- **Customer Demographics**: Job type, marital status, education level
- **Financial Information**: Credit default status, housing loan, personal loan
- **Campaign Details**: Contact method, month, day of week
- **Previous Campaign Outcomes**: Previous marketing campaign results

The target variable indicates whether the client subscribed to a term deposit, and anomaly detection techniques are applied to identify unusual patterns that may indicate fraudulent behavior or data quality issues.

### Data Source
The banking dataset is stored in `data/bank-additional-ful-nominal.arff` and contains categorical attributes suitable for anomaly detection in banking contexts.

## Authors

[Keith Sng](mailto:keith.sngth@gmail.com)<br>
Samuel Koh

## References

Pang, G., Shen, C., Cao, L., & Hengel, A. V. D. (2021). Deep learning for anomaly detection: A review. *ACM Computing Surveys (CSUR)*, *54*(2), 1-38. https://doi.org/10.1145/3439950
