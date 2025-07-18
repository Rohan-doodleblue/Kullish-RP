import pickle
import json
import os
from typing import Any, Dict
import pandas as pd

def save_model(model: Any, filepath: str) -> None:
    """Save a trained model to disk"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")

def load_model(filepath: str) -> Any:
    """Load a trained model from disk"""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filepath}")
    return model

def save_results(results: Dict[str, Any], filepath: str) -> None:
    """Save experiment results to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Filter and convert results to JSON-serializable format
    serializable_results = {}
    for key, value in results.items():
        # Skip complex objects that can't be serialized
        if key in ['feature_engineer', 'model_trainer', 'model_evaluator', 'config']:
            continue
            
        if hasattr(value, 'tolist'):  # numpy array
            serializable_results[key] = value.tolist()
        elif hasattr(value, 'to_dict'):  # pandas DataFrame
            serializable_results[key] = value.to_dict('records')
        elif isinstance(value, dict):
            serializable_results[key] = {}
            for k, v in value.items():
                if hasattr(v, 'tolist'):
                    serializable_results[key][k] = v.tolist()
                elif hasattr(v, 'to_dict'):
                    serializable_results[key][k] = v.to_dict('records')
                else:
                    serializable_results[key][k] = v
        elif isinstance(value, (str, int, float, bool, list)):
            serializable_results[key] = value
        else:
            # Skip other non-serializable objects
            continue
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"Results saved to {filepath}")

def load_results(filepath: str) -> Dict[str, Any]:
    """Load experiment results from JSON file"""
    with open(filepath, 'r') as f:
        results = json.load(f)
    print(f"Results loaded from {filepath}")
    return results

def save_config(config: Any, filepath: str) -> None:
    """Save configuration to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convert dataclass to dict
    if hasattr(config, '__dict__'):
        config_dict = config.__dict__.copy()
    else:
        config_dict = config
    
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    print(f"Configuration saved to {filepath}")

def load_config(filepath: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    with open(filepath, 'r') as f:
        config = json.load(f)
    print(f"Configuration loaded from {filepath}")
    return config 