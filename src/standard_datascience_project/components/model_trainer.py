import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from ..entity.config_entity import ModelConfig

class ModelTrainer:
    """Component for training and evaluating multiple models"""
    
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.trained_models = {}
        self.model_predictions = {}
        
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Train all configured models"""
        
        print("Training models...")
        
        for name, model in self.model_config.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            print(f"{name} trained successfully.")
        
        return self.trained_models
    
    def predict(self, X_test: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """Generate predictions for all trained models"""
        
        for name, model in self.trained_models.items():
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)[:, 1]
            
            self.model_predictions[name] = {
                'predictions': predictions,
                'probabilities': probabilities
            }
        
        return self.model_predictions
    
    def get_model_predictions(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Get model predictions"""
        return self.model_predictions
    
    def get_trained_models(self) -> Dict[str, Any]:
        """Get trained models"""
        return self.trained_models 