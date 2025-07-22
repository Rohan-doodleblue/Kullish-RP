import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from typing import Dict, Any, Tuple

class ModelEvaluator:
    """Component for evaluating and visualizing model performance"""
    
    def __init__(self):
        self.evaluation_results = {}
        
    def evaluate_models(self, model_predictions: Dict[str, Dict[str, np.ndarray]], 
                       y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate all models and store results"""
        
        for name, outcomes in model_predictions.items():
            print(f"\nEvaluation Results for {name}:")
            print(classification_report(y_test, outcomes['predictions']))
            
            # Calculate metrics
            cm = confusion_matrix(y_test, outcomes['predictions'])
            fpr, tpr, _ = roc_curve(y_test, outcomes['probabilities'])
            roc_auc = auc(fpr, tpr)
            
            self.evaluation_results[name] = {
                'confusion_matrix': cm,
                'fpr': fpr,
                'tpr': tpr,
                'roc_auc': roc_auc,
                'classification_report': classification_report(y_test, outcomes['predictions'], output_dict=True)
            }
        
        return self.evaluation_results
    
    def plot_confusion_matrix(self, cm: np.ndarray, classes: list, 
                            title: str = 'Confusion matrix', cmap=plt.cm.Blues):
        """Plot confusion matrix"""
        plt.subplot(1, 2, 1)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        plt.xticks(np.arange(len(classes)), classes, rotation=45)
        plt.yticks(np.arange(len(classes)), classes)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    
    def plot_roc_curve(self, fpr: np.ndarray, tpr: np.ndarray, 
                      model_name: str, roc_auc: float):
        """Plot ROC curve"""
        plt.subplot(1, 2, 2)
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {model_name}')
        plt.legend(loc="lower right")
    
    def create_evaluation_plots(self, save_plots: bool = False):
        """Create evaluation plots for all models"""
        
        for name, results in self.evaluation_results.items():
            plt.figure(figsize=(12, 6))
            
            self.plot_confusion_matrix(
                results['confusion_matrix'], 
                classes=['Normal', 'Anomaly'], 
                title=f'{name} Confusion Matrix'
            )
            
            self.plot_roc_curve(
                results['fpr'], 
                results['tpr'], 
                name, 
                results['roc_auc']
            )
            
            plt.tight_layout()
            
            if save_plots:
                plt.savefig(f'plots/{name}_evaluation.png', dpi=300, bbox_inches='tight')
            
            # plt.show()
    
    def get_best_model(self) -> Tuple[str, float]:
        """Get the best performing model based on ROC AUC"""
        roc_auc_scores = {
            name: results['roc_auc'] 
            for name, results in self.evaluation_results.items()
        }
        
        best_model = max(roc_auc_scores, key=roc_auc_scores.get)
        best_score = roc_auc_scores[best_model]
        
        print(f"Best performing model: {best_model} with ROC AUC = {best_score:.2f}")
        
        return best_model, best_score
    
    def print_comparison_summary(self):
        """Print comparison summary of all models"""
        print("\n=== MODEL COMPARISON SUMMARY ===")
        
        for name, results in self.evaluation_results.items():
            print(f"\n{name}:")
            print(f"  ROC AUC: {results['roc_auc']:.4f}")
            print(f"  Precision: {results['classification_report']['weighted avg']['precision']:.4f}")
            print(f"  Recall: {results['classification_report']['weighted avg']['recall']:.4f}")
            print(f"  F1-Score: {results['classification_report']['weighted avg']['f1-score']:.4f}")
    
    def get_evaluation_results(self) -> Dict[str, Any]:
        """Get evaluation results"""
        return self.evaluation_results 