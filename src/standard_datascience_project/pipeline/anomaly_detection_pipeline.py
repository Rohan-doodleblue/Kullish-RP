import os
import pandas as pd
from typing import Dict, Any, Optional
from ..entity.config_entity import AnomalyDetectionConfig
from ..components.data_generator import SyntheticDataGenerator
from ..components.data_analyzer import DataAnalyzer
from ..components.feature_engineer import FeatureEngineer
from ..components.model_trainer import ModelTrainer
from ..components.model_evaluator import ModelEvaluator

class AnomalyDetectionPipeline:
    """Main pipeline for anomaly detection experiment"""
    
    def __init__(self, config: Optional[AnomalyDetectionConfig] = None):
        self.config = config or AnomalyDetectionConfig()
        self.data = None
        self.analysis_results = None
        self.feature_engineer = None
        self.model_trainer = None
        self.model_evaluator = None
        
    def run_pipeline(self, generate_data: bool = True, save_plots: bool = False) -> Dict[str, Any]:
        """Run the complete anomaly detection pipeline"""
        
        print("=== ANOMALY DETECTION PIPELINE STARTED ===")
        
        # Step 1: Data Generation
        if generate_data:
            print("\n1. Generating synthetic data...")
            data_generator = SyntheticDataGenerator(self.config.data_config)
            self.data = data_generator.generate_data()
            data_generator.save_data(self.data)
        else:
            print("\n1. Loading existing data...")
            self.data = pd.read_csv('data/synthetic_web_auth_logs.csv')
            self.data['Timestamp'] = pd.to_datetime(self.data['Timestamp'])
        
        # Step 2: Data Analysis
        print("\n2. Performing data analysis...")
        data_analyzer = DataAnalyzer(self.config.feature_config)
        self.analysis_results = data_analyzer.analyze_data(self.data)
        data_analyzer.print_analysis_summary(self.analysis_results)
        
        # Step 3: Feature Engineering
        print("\n3. Engineering features...")
        self.feature_engineer = FeatureEngineer(self.config.feature_config, self.config.model_config)
        X_train, X_test, y_train, y_test = self.feature_engineer.prepare_data(self.data)
        
        # Step 4: Model Training
        print("\n4. Training models...")
        self.model_trainer = ModelTrainer(self.config.model_config)
        self.model_trainer.train_models(X_train, y_train)
        
        # Step 5: Model Prediction
        print("\n5. Generating predictions...")
        model_predictions = self.model_trainer.predict(X_test)
        
        # Step 6: Model Evaluation
        print("\n6. Evaluating models...")
        self.model_evaluator = ModelEvaluator()
        evaluation_results = self.model_evaluator.evaluate_models(model_predictions, y_test)
        
        # Step 7: Visualization
        print("\n7. Creating visualizations...")
        data_analyzer.create_visualizations(self.data, save_plots=save_plots)
        self.model_evaluator.create_evaluation_plots(save_plots=save_plots)
        
        # Step 8: Results Summary
        print("\n8. Generating results summary...")
        self.model_evaluator.print_comparison_summary()
        best_model, best_score = self.model_evaluator.get_best_model()
        
        print("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")
        
        # Return comprehensive results
        results = {
            'data': self.data,
            'analysis_results': self.analysis_results,
            'feature_engineer': self.feature_engineer,
            'model_trainer': self.model_trainer,
            'model_evaluator': self.model_evaluator,
            'best_model': best_model,
            'best_score': best_score,
            'evaluation_results': evaluation_results,
            'config': self.config
        }
        
        return results
    
    def get_data(self) -> pd.DataFrame:
        """Get the processed data"""
        return self.data
    
    def get_analysis_results(self) -> Dict[str, Any]:
        """Get data analysis results"""
        return self.analysis_results
    
    def get_evaluation_results(self) -> Dict[str, Any]:
        """Get model evaluation results"""
        if self.model_evaluator:
            return self.model_evaluator.get_evaluation_results()
        return {}
    
    def get_best_model_info(self) -> tuple:
        """Get best model information"""
        if self.model_evaluator:
            return self.model_evaluator.get_best_model()
        return None, None 