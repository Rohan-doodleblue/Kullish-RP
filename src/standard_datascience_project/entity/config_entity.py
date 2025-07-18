from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass
class DataGenerationConfig:
    """Configuration for synthetic data generation"""
    num_users: int = 1000
    num_records: int = 20000
    anomaly_rate: float = 0.05
    success_login_rate: float = 0.95
    uncommon_ip_prob: float = 0.03
    failed_attempt_limit: int = 5
    behavioral_score_low: int = 40
    behavioral_score_high: int = 120
    seed: int = 50

@dataclass
class ModelConfig:
    """Configuration for model training"""
    test_size: float = 0.2
    random_state: int = 50
    models: dict = None
    
    def __post_init__(self):
        if self.models is None:
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.svm import SVC
            from sklearn.neural_network import MLPClassifier
            
            self.models = {
                'Logistic Regression': LogisticRegression(max_iter=1000),
                'Random Forest': RandomForestClassifier(n_estimators=100),
                'SVM': SVC(probability=True),
                'Neural Network': MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000)
            }

@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""
    numerical_features: List[str] = None
    categorical_features: List[str] = None
    
    def __post_init__(self):
        if self.numerical_features is None:
            self.numerical_features = ['Session Duration', 'Failed Attempts', 'Behavioral Score']
        if self.categorical_features is None:
            self.categorical_features = ['Device Type', 'Location', 'Login Status', 'IP Address']

@dataclass
class AnomalyDetectionConfig:
    """Main configuration class"""
    data_config: DataGenerationConfig = None
    model_config: ModelConfig = None
    feature_config: FeatureConfig = None
    
    def __post_init__(self):
        if self.data_config is None:
            self.data_config = DataGenerationConfig()
        if self.model_config is None:
            self.model_config = ModelConfig()
        if self.feature_config is None:
            self.feature_config = FeatureConfig() 