import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from src.standard_datascience_project import logger
from src.standard_datascience_project.entity.config_entity import DataTransformationConfig


class TransformationEvaluator:
    """Evaluates the quality and correctness of data transformations"""
    
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.evaluation_results = {}
        
    def evaluate_transformations(self, original_data: pd.DataFrame, transformed_data: pd.DataFrame, 
                               target_column: str = None) -> Dict:
        """Comprehensive evaluation of transformations"""
        logger.info("Starting transformation evaluation")
        
        evaluation_results = {}
        
        # 1. Data Integrity Checks
        evaluation_results['data_integrity'] = self._check_data_integrity(original_data, transformed_data)
        
        # 2. Statistical Quality Checks
        evaluation_results['statistical_quality'] = self._check_statistical_quality(original_data, transformed_data)
        
        # 3. Feature Quality Assessment
        evaluation_results['feature_quality'] = self._check_feature_quality(original_data, transformed_data)
        
        # 4. Target Variable Analysis (if applicable)
        if target_column and target_column in transformed_data.columns:
            evaluation_results['target_analysis'] = self._analyze_target_variable(
                original_data, transformed_data, target_column
            )
        
        # 5. Transformation Impact Analysis
        evaluation_results['transformation_impact'] = self._analyze_transformation_impact(
            original_data, transformed_data
        )
        
        # 6. Data Leakage Detection
        evaluation_results['data_leakage'] = self._check_data_leakage(original_data, transformed_data)
        
        # 7. Overall Quality Score
        evaluation_results['overall_score'] = self._calculate_overall_score(evaluation_results)
        
        self.evaluation_results = evaluation_results
        self._save_evaluation_report(evaluation_results)
        self._generate_visualizations(original_data, transformed_data, target_column)
        
        return evaluation_results
    
    def _check_data_integrity(self, original: pd.DataFrame, transformed: pd.DataFrame) -> Dict:
        """Check for data integrity issues"""
        logger.info("Checking data integrity")
        
        integrity_checks = {
            'total_rows_preserved': len(transformed) > 0,
            'no_infinite_values': not np.isinf(transformed.select_dtypes(include=[np.number])).any().any(),
            'no_nan_values': not transformed.isnull().any().any(),
            'no_duplicate_rows': not transformed.duplicated().any(),
            'reasonable_data_types': True,
            'warnings': []
        }
        
        # Check for potential issues
        if len(transformed) < len(original) * 0.5:
            integrity_checks['warnings'].append(f"Significant data loss: {len(transformed)}/{len(original)} rows")
            integrity_checks['total_rows_preserved'] = False
        
        # Check for infinite values
        numerical_cols = transformed.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            inf_count = np.isinf(transformed[numerical_cols]).sum().sum()
            if inf_count > 0:
                integrity_checks['warnings'].append(f"Found {inf_count} infinite values")
                integrity_checks['no_infinite_values'] = False
        
        # Check for NaN values
        nan_count = transformed.isnull().sum().sum()
        if nan_count > 0:
            integrity_checks['warnings'].append(f"Found {nan_count} NaN values")
            integrity_checks['no_nan_values'] = False
        
        # Check for duplicate rows
        duplicate_count = transformed.duplicated().sum()
        if duplicate_count > 0:
            integrity_checks['warnings'].append(f"Found {duplicate_count} duplicate rows")
            integrity_checks['no_duplicate_rows'] = False
        
        return integrity_checks
    
    def _check_statistical_quality(self, original: pd.DataFrame, transformed: pd.DataFrame) -> Dict:
        """Check statistical quality of transformations"""
        logger.info("Checking statistical quality")
        
        quality_metrics = {
            'variance_preservation': {},
            'distribution_similarity': {},
            'correlation_preservation': {},
            'warnings': []
        }
        
        # Compare numerical columns
        original_numerical = original.select_dtypes(include=[np.number])
        transformed_numerical = transformed.select_dtypes(include=[np.number])
        
        if len(original_numerical.columns) > 0 and len(transformed_numerical.columns) > 0:
            # Find common columns
            common_cols = set(original_numerical.columns) & set(transformed_numerical.columns)
            
            for col in common_cols:
                if col in original_numerical.columns and col in transformed_numerical.columns:
                    # Variance preservation
                    orig_var = original_numerical[col].var()
                    trans_var = transformed_numerical[col].var()
                    
                    if orig_var > 0:
                        variance_ratio = trans_var / orig_var
                        quality_metrics['variance_preservation'][col] = variance_ratio
                        
                        if variance_ratio < 0.1:
                            quality_metrics['warnings'].append(f"Low variance preservation in {col}: {variance_ratio:.3f}")
                    
                    # Distribution similarity (KS test)
                    try:
                        ks_stat, p_value = stats.ks_2samp(
                            original_numerical[col].dropna(), 
                            transformed_numerical[col].dropna()
                        )
                        quality_metrics['distribution_similarity'][col] = {
                            'ks_statistic': ks_stat,
                            'p_value': p_value,
                            'similar': p_value > 0.05
                        }
                        
                        if p_value < 0.05:
                            quality_metrics['warnings'].append(f"Significant distribution change in {col}: p={p_value:.3f}")
                    except:
                        quality_metrics['distribution_similarity'][col] = {'error': 'Could not compute KS test'}
        
        return quality_metrics
    
    def _check_feature_quality(self, original: pd.DataFrame, transformed: pd.DataFrame) -> Dict:
        """Check feature quality after transformation"""
        logger.info("Checking feature quality")
        
        feature_quality = {
            'feature_count': len(transformed.columns),
            'numerical_features': len(transformed.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(transformed.select_dtypes(include=['object', 'category']).columns),
            'high_variance_features': 0,
            'low_variance_features': 0,
            'constant_features': 0,
            'warnings': []
        }
        
        # Check for constant features
        for col in transformed.columns:
            if transformed[col].nunique() == 1:
                feature_quality['constant_features'] += 1
                feature_quality['warnings'].append(f"Constant feature detected: {col}")
            elif transformed[col].nunique() == 2:
                feature_quality['low_variance_features'] += 1
            else:
                feature_quality['high_variance_features'] += 1
        
        # Check for reasonable feature count
        if feature_quality['feature_count'] > 1000:
            feature_quality['warnings'].append(f"Very high feature count: {feature_quality['feature_count']}")
        elif feature_quality['feature_count'] < 2:
            feature_quality['warnings'].append(f"Very low feature count: {feature_quality['feature_count']}")
        
        return feature_quality
    
    def _analyze_target_variable(self, original: pd.DataFrame, transformed: pd.DataFrame, 
                                target_column: str) -> Dict:
        """Analyze target variable quality"""
        logger.info("Analyzing target variable")
        
        target_analysis = {
            'target_preserved': target_column in transformed.columns,
            'target_type_consistent': True,
            'class_balance': {},
            'target_distribution': {},
            'warnings': []
        }
        
        if target_column in transformed.columns:
            target = transformed[target_column]
            
            # Check target type consistency
            if target.dtype != original[target_column].dtype:
                target_analysis['target_type_consistent'] = False
                target_analysis['warnings'].append(f"Target type changed: {original[target_column].dtype} → {target.dtype}")
            
            # Analyze class balance for classification
            if target.dtype == 'object' or target.dtype == 'category' or target.nunique() <= 10:
                class_counts = target.value_counts()
                target_analysis['class_balance'] = {
                    'n_classes': len(class_counts),
                    'min_class_count': class_counts.min(),
                    'max_class_count': class_counts.max(),
                    'balance_ratio': class_counts.min() / class_counts.max()
                }
                
                if target_analysis['class_balance']['balance_ratio'] < 0.1:
                    target_analysis['warnings'].append("Severe class imbalance detected")
            
            # Analyze distribution for regression
            else:
                target_analysis['target_distribution'] = {
                    'mean': target.mean(),
                    'std': target.std(),
                    'skewness': target.skew(),
                    'kurtosis': target.kurtosis()
                }
        
        return target_analysis
    
    def _analyze_transformation_impact(self, original: pd.DataFrame, transformed: pd.DataFrame) -> Dict:
        """Analyze the impact of transformations"""
        logger.info("Analyzing transformation impact")
        
        impact_analysis = {
            'rows_removed': len(original) - len(transformed),
            'columns_removed': len(original.columns) - len(transformed.columns),
            'columns_added': len(transformed.columns) - len(set(original.columns) & set(transformed.columns)),
            'data_reduction_ratio': 1 - (len(transformed) * len(transformed.columns)) / (len(original) * len(original.columns)),
            'warnings': []
        }
        
        # Check for excessive data loss
        if impact_analysis['rows_removed'] > len(original) * 0.5:
            impact_analysis['warnings'].append(f"Excessive row removal: {impact_analysis['rows_removed']} rows")
        
        if impact_analysis['columns_removed'] > len(original.columns) * 0.8:
            impact_analysis['warnings'].append(f"Excessive column removal: {impact_analysis['columns_removed']} columns")
        
        if impact_analysis['columns_added'] > len(original.columns) * 2:
            impact_analysis['warnings'].append(f"Excessive feature creation: {impact_analysis['columns_added']} new columns")
        
        return impact_analysis
    
    def _check_data_leakage(self, original: pd.DataFrame, transformed: pd.DataFrame) -> Dict:
        """Check for potential data leakage"""
        logger.info("Checking for data leakage")
        
        leakage_checks = {
            'target_in_features': False,
            'future_information': False,
            'warnings': []
        }
        
        # Check if target column is in features (if we have target info)
        # This would be checked in the calling code
        
        # Check for potential future information (e.g., date-based features that might include future dates)
        date_columns = []
        for col in transformed.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                date_columns.append(col)
        
        if date_columns:
            leakage_checks['warnings'].append(f"Date/time features detected: {date_columns}")
        
        return leakage_checks
    
    def _calculate_overall_score(self, evaluation_results: Dict) -> Dict:
        """Calculate overall quality score"""
        logger.info("Calculating overall quality score")
        
        score = 100  # Start with perfect score
        deductions = []
        
        # Data integrity deductions
        integrity = evaluation_results['data_integrity']
        if not integrity['total_rows_preserved']:
            score -= 20
            deductions.append("Data loss")
        if not integrity['no_infinite_values']:
            score -= 15
            deductions.append("Infinite values")
        if not integrity['no_nan_values']:
            score -= 15
            deductions.append("NaN values")
        if not integrity['no_duplicate_rows']:
            score -= 10
            deductions.append("Duplicate rows")
        
        # Statistical quality deductions
        stat_quality = evaluation_results['statistical_quality']
        for warning in stat_quality['warnings']:
            score -= 5
            deductions.append("Statistical issues")
        
        # Feature quality deductions
        feature_quality = evaluation_results['feature_quality']
        if feature_quality['constant_features'] > 0:
            score -= feature_quality['constant_features'] * 2
            deductions.append("Constant features")
        
        # Target analysis deductions
        if 'target_analysis' in evaluation_results:
            target_analysis = evaluation_results['target_analysis']
            for warning in target_analysis['warnings']:
                score -= 5
                deductions.append("Target issues")
        
        # Transformation impact deductions
        impact = evaluation_results['transformation_impact']
        for warning in impact['warnings']:
            score -= 5
            deductions.append("Transformation impact")
        
        # Ensure score is not negative
        score = max(0, score)
        
        # Determine grade
        if score >= 90:
            grade = "A"
        elif score >= 80:
            grade = "B"
        elif score >= 70:
            grade = "C"
        elif score >= 60:
            grade = "D"
        else:
            grade = "F"
        
        return {
            'score': score,
            'grade': grade,
            'deductions': deductions,
            'recommendations': self._generate_recommendations(evaluation_results)
        }
    
    def _generate_recommendations(self, evaluation_results: Dict) -> List[str]:
        """Generate recommendations based on evaluation results"""
        recommendations = []
        
        # Data integrity recommendations
        integrity = evaluation_results['data_integrity']
        if not integrity['total_rows_preserved']:
            recommendations.append("Review transformation pipeline to reduce data loss")
        if not integrity['no_infinite_values']:
            recommendations.append("Check outlier handling and scaling methods")
        if not integrity['no_nan_values']:
            recommendations.append("Improve missing value handling strategy")
        
        # Feature quality recommendations
        feature_quality = evaluation_results['feature_quality']
        if feature_quality['constant_features'] > 0:
            recommendations.append("Remove or transform constant features")
        if feature_quality['feature_count'] > 1000:
            recommendations.append("Consider feature selection to reduce dimensionality")
        
        # Target analysis recommendations
        if 'target_analysis' in evaluation_results:
            target_analysis = evaluation_results['target_analysis']
            if not target_analysis['target_type_consistent']:
                recommendations.append("Ensure target variable type consistency")
            if 'Severe class imbalance detected' in target_analysis['warnings']:
                recommendations.append("Consider class balancing techniques")
        
        return recommendations
    
    def _save_evaluation_report(self, evaluation_results: Dict):
        """Save detailed evaluation report"""
        import json
        
        def make_serializable(obj):
            """Recursively convert numpy types to JSON serializable types"""
            if isinstance(obj, dict):
                return {key: make_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj
            else:
                return str(obj)  # Convert any other types to string
        
        # Create a serializable version of the results
        serializable_results = make_serializable(evaluation_results)
        
        # Save to file
        report_path = Path(self.config.root_dir) / "transformation_evaluation.json"
        with open(report_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved evaluation report to {report_path}")
        
        # Print summary
        overall_score = evaluation_results['overall_score']
        logger.info(f"Transformation Quality Score: {overall_score['score']}/100 ({overall_score['grade']})")
        
        if overall_score['deductions']:
            logger.info("Issues found:")
            for deduction in overall_score['deductions']:
                logger.info(f"  - {deduction}")
        
        if overall_score['recommendations']:
            logger.info("Recommendations:")
            for rec in overall_score['recommendations']:
                logger.info(f"  - {rec}")
    
    def _generate_visualizations(self, original: pd.DataFrame, transformed: pd.DataFrame, target_column: str = None):
        """Generate visualization plots for evaluation"""
        try:
            # Create comparison plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Transformation Quality Assessment', fontsize=16)
            
            # 1. Data shape comparison
            axes[0, 0].bar(['Original', 'Transformed'], [len(original), len(transformed)])
            axes[0, 0].set_title('Number of Rows')
            axes[0, 0].set_ylabel('Count')
            
            # 2. Feature count comparison
            axes[0, 1].bar(['Original', 'Transformed'], [len(original.columns), len(transformed.columns)])
            axes[0, 1].set_title('Number of Features')
            axes[0, 1].set_ylabel('Count')
            
            # 3. Missing values comparison
            orig_missing = original.isnull().sum().sum()
            trans_missing = transformed.isnull().sum().sum()
            axes[1, 0].bar(['Original', 'Transformed'], [orig_missing, trans_missing])
            axes[1, 0].set_title('Missing Values')
            axes[1, 0].set_ylabel('Count')
            
            # 4. Target distribution (if available)
            if target_column and target_column in transformed.columns:
                transformed[target_column].value_counts().plot(kind='bar', ax=axes[1, 1])
                axes[1, 1].set_title('Target Distribution (Transformed)')
                axes[1, 1].tick_params(axis='x', rotation=45)
            else:
                axes[1, 1].text(0.5, 0.5, 'No target column', ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Target Distribution')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = Path(self.config.root_dir) / "transformation_evaluation.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved evaluation visualization to {plot_path}")
            
        except Exception as e:
            logger.warning(f"Could not generate visualizations: {e}") 