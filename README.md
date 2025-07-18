# Anomaly Detection Pipeline

A modular and scalable anomaly detection pipeline for web authentication logs using machine learning.

## Project Structure

```
Kullish-RP/
├── src/
│   └── standard_datascience_project/
│       ├── components/           # Pipeline components
│       │   ├── data_generator.py
│       │   ├── data_analyzer.py
│       │   ├── feature_engineer.py
│       │   ├── model_trainer.py
│       │   └── model_evaluator.py
│       ├── entity/              # Configuration entities
│       │   └── config_entity.py
│       ├── pipeline/            # Main pipeline
│       │   └── anomaly_detection_pipeline.py
│       ├── utils/               # Utility functions
│       │   ├── logger.py
│       │   └── save_load.py
│       └── __init__.py
├── data/                        # Data directory
├── logs/                        # Log files
├── results/                     # Experiment results
├── plots/                       # Generated plots
├── config.yaml                  # Configuration file
├── run_anomaly_detection.py     # Main execution script
├── ml.py                        # Original notebook code
└── requirements.txt
```

## Features

- **Modular Design**: Each component is independent and reusable
- **Configuration Management**: Centralized configuration using dataclasses
- **Comprehensive Logging**: Detailed logging for debugging and monitoring
- **Data Generation**: Synthetic data generation for testing
- **Feature Engineering**: Automated feature preprocessing
- **Multiple Models**: Support for various ML algorithms
- **Evaluation**: Comprehensive model evaluation with visualizations
- **Results Persistence**: Save and load models, results, and configurations

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Kullish-RP
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the complete pipeline:

```bash
python run_anomaly_detection.py
```

### Using the Pipeline Programmatically

```python
from src.standard_datascience_project.pipeline.anomaly_detection_pipeline import AnomalyDetectionPipeline
from src.standard_datascience_project.entity.config_entity import AnomalyDetectionConfig

# Initialize configuration
config = AnomalyDetectionConfig()

# Create and run pipeline
pipeline = AnomalyDetectionPipeline(config)
results = pipeline.run_pipeline(generate_data=True, save_plots=True)

# Access results
print(f"Best Model: {results['best_model']}")
print(f"Best Score: {results['best_score']}")
```

### Configuration

Modify `config.yaml` or create custom configuration:

```python
from src.standard_datascience_project.entity.config_entity import AnomalyDetectionConfig, DataGenerationConfig

# Custom configuration
custom_config = AnomalyDetectionConfig(
    data_config=DataGenerationConfig(
        num_users=2000,
        num_records=50000,
        anomaly_rate=0.03
    )
)
```

## Components

### 1. Data Generator (`data_generator.py`)
- Generates synthetic web authentication logs
- Configurable anomaly injection
- Realistic data patterns

### 2. Data Analyzer (`data_analyzer.py`)
- Exploratory data analysis
- Statistical summaries
- Data visualizations

### 3. Feature Engineer (`feature_engineer.py`)
- Automated feature preprocessing
- Time-based feature extraction
- Categorical encoding

### 4. Model Trainer (`model_trainer.py`)
- Multi-model training
- Configurable algorithms
- Prediction generation

### 5. Model Evaluator (`model_evaluator.py`)
- Comprehensive evaluation metrics
- ROC curves and confusion matrices
- Model comparison

## Output

The pipeline generates:

- **Data**: `data/synthetic_web_auth_logs.csv`
- **Logs**: `logs/anomaly_detection_YYYYMMDD_HHMMSS.log`
- **Results**: `results/experiment_results.json`
- **Configuration**: `results/experiment_config.json`
- **Plots**: `plots/` directory with visualizations

## Supported Models

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- Neural Network (MLP)

## Evaluation Metrics

- Classification Report (Precision, Recall, F1-Score)
- Confusion Matrix
- ROC Curve and AUC
- Model Comparison Summary

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
