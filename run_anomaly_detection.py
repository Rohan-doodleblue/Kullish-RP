#!/usr/bin/env python3
"""
Main script to run the Anomaly Detection Pipeline
"""

import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from standard_datascience_project.pipeline.anomaly_detection_pipeline import AnomalyDetectionPipeline
from standard_datascience_project.entity.config_entity import AnomalyDetectionConfig
from standard_datascience_project.utils.logger import setup_logger
from standard_datascience_project.utils.save_load import save_model, save_config

def main():
    """Main function to run the anomaly detection pipeline"""
    
    # Setup logger
    logger = setup_logger()
    logger.info("Starting Anomaly Detection Pipeline")
    
    try:
        # Initialize configuration
        config = AnomalyDetectionConfig()
        
        # Create and run pipeline
        pipeline = AnomalyDetectionPipeline(config)
        
        # Run the complete pipeline
        results = pipeline.run_pipeline(
            generate_data=True,  # Set to False if you want to use existing data
            save_plots=True      # Set to True to save plots to disk
        )
        
        # Save best model and configuration
        best_model_name = results['best_model']
        best_model = results['model_trainer'].get_trained_models()[best_model_name]
        save_model(best_model, f'results/best_model_{best_model_name.lower().replace(" ", "_")}.pkl')
        save_config(config, 'results/experiment_config.json')
        
        logger.info("Pipeline completed successfully!")
        
        # Print summary
        print("\n" + "="*50)
        print("EXPERIMENT SUMMARY")
        print("="*50)
        print(f"Best Model: {results['best_model']}")
        print(f"Best ROC AUC Score: {results['best_score']:.4f}")
        print(f"Data Shape: {results['data'].shape}")
        print(f"Anomaly Rate: {results['data']['Anomaly'].mean():.2%}")
        print(f"Best Model Saved: results/best_model_{results['best_model'].lower().replace(' ', '_')}.pkl")
        print(f"Configuration Saved: results/experiment_config.json")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 