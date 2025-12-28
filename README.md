# Anomaly Detection with PySpark and Machine Learning

## Overview

This project implements anomaly detection techniques using PySpark and various machine learning libraries. The project leverages distributed computing capabilities of PySpark combined with powerful ML algorithms from scikit-learn and other frameworks to detect outliers and anomalies in large-scale datasets.

### Key Features

- **Distributed Processing**: Utilises PySpark for handling large-scale data efficiently
- **Multiple ML Algorithms**: Implements various anomaly detection techniques including:
  - Isolation Forest
  - One-Class SVM
  - Local Outlier Factor (LOF)
  - DBSCAN clustering
  - Statistical methods
- **Multiple Data Types**: Supports categorical, numerical, time series, graph, image, and video data
- **Benchmark Datasets**: Includes ADBenchmark datasets for evaluation and testing

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

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management. uv is a modern alternative to pip that's significantly faster and more reliable.

#### Install uv

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Or via pip:**
```bash
pip install uv
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/anomaly-detection.git
cd anomaly-detection
```

2. **Create a virtual environment and install dependencies**
```bash
# uv will automatically create a virtual environment and install all dependencies
uv sync
```

3. **Install the project in editable mode**
```bash
# This allows you to import the project modules anywhere
uv pip install -e .
```

4. **Activate the virtual environment**
```bash
# macOS/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### Running the Project

**Launch Jupyter Notebook:**
```bash
jupyter notebook
```

**Run the main script:**
```bash
python src/main.py
```

**Access project paths in your code:**
```python
from src.config import DATA_DIR, CATEGORICAL_DATA_DIR, NUMERICAL_DATA_DIR

# Load data using project-relative paths
data_path = CATEGORICAL_DATA_DIR / 'your-dataset.arff'
```

## Dependencies

Core libraries:
- **PySpark**: Distributed data processing
- **scikit-learn**: Machine learning algorithms
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scipy**: Scientific computing
- **matplotlib/seaborn**: Data visualisation
- **loguru**: Logging

See `pyproject.toml` for complete list of dependencies.

## Usage Example

```python
from pyspark.sql import SparkSession
from src.config import NUMERICAL_DATA_DIR
import pandas as pd

# Initialise Spark session
spark = SparkSession.builder \
    .appName("AnomalyDetection") \
    .getOrCreate()

# Load data
data_path = NUMERICAL_DATA_DIR / 'DevNet datasets' / 'annthyroid_21feat_normalised.csv'
df = spark.read.csv(str(data_path), header=True, inferSchema=True)

# Apply anomaly detection algorithms
# ... your analysis code here
```

## Dataset Information

This project uses the [ADBenchmark datasets](https://github.com/mala-lab/ADBenchmarks-anomaly-detection-datasets) which include:
- Categorical data (ARFF format)
- Numerical data (CSV format)
- Time series data
- Graph data
- Image data
- Video data

## Authors

[Keith Sng](mailto:keith.sngth@gmail.com)

## Acknowledgments

- ADBench benchmark datasets for anomaly detection
- PySpark and scikit-learn communities
