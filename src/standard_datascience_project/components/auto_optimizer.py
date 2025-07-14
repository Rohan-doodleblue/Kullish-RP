import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
import itertools
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.ensemble import IsolationForest
from scipy import stats
import joblib

from src.standard_datascience_project import logger
from src.standard_datascience_project.entity.config_entity import DataTransformationConfig


class AutoOptimizer:
    """Auto-optimizer that uses cross-validation to select the best transformation strategies"""
    
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.params = config.transformation_params
        self.best_config = None
        self.optimization_results = []
        
    def optimize_transformations(self, data: pd.DataFrame, target_column: str = None) -> Dict:
        """Main optimization method that finds the best transformation pipeline"""
        logger.info("Starting auto-optimization of transformation strategies")
        
        # Determine problem type
        problem_type = self._determine_problem_type(data, target_column)
        logger.info(f"Detected problem type: {problem_type}")
        
        # Generate transformation combinations
        transformation_combinations = self._generate_combinations()
        logger.info(f"Generated {len(transformation_combinations)} transformation combinations")
        
        # Evaluate each combination
        best_score = -np.inf
        best_config = None
        
        for i, config in enumerate(transformation_combinations):
            logger.info(f"Evaluating combination {i+1}/{len(transformation_combinations)}")
            
            try:
                score, transformed_data = self._evaluate_combination(data, config, target_column, problem_type)
                
                result = {
                    'config': config,
                    'score': score,
                    'data_shape': transformed_data.shape,
                    'combination_id': i
                }
                self.optimization_results.append(result)
                
                if score > best_score:
                    best_score = score
                    best_config = config
                    logger.info(f"New best score: {best_score:.4f}")
                    
            except Exception as e:
                logger.warning(f"Combination {i+1} failed: {e}")
                continue
        
        if best_config is None:
            logger.warning("No valid configuration found, using default")
            best_config = self._get_default_config()
        elif best_score == -np.inf:
            logger.warning("All configurations failed, using default")
            best_config = self._get_default_config()
        
        self.best_config = best_config
        logger.info(f"Best configuration selected with score: {best_score:.4f}")
        
        return best_config
    
    def _determine_problem_type(self, data: pd.DataFrame, target_column: str) -> str:
        """Determine if this is a classification or regression problem"""
        if target_column is None or target_column not in data.columns:
            return "unsupervised"
        
        target = data[target_column]
        
        # Check if target is categorical
        if target.dtype == 'object' or target.dtype == 'category':
            return "classification"
        elif target.dtype == 'bool':
            return "classification"
        else:
            # Check number of unique values
            unique_vals = target.nunique()
            if unique_vals <= 10:  # Heuristic for classification
                return "classification"
            else:
                return "regression"
    
    def _generate_combinations(self) -> List[Dict]:
        """Generate all possible combinations of transformation strategies"""
        combinations = []
        
        # Missing value strategies
        missing_strategies = []
        if self.params['missing_values']['options']['drop']['enabled']:
            missing_strategies.append(('drop', None))
        if self.params['missing_values']['options']['impute']['enabled']:
            for method in self.params['missing_values']['options']['impute']['methods']:
                missing_strategies.append(('impute', method))
        if self.params['missing_values']['options']['flag']['enabled']:
            missing_strategies.append(('flag', None))
        
        # Outlier strategies
        outlier_strategies = []
        if self.params['outliers']['options']['iqr']['enabled']:
            for multiplier in self.params['outliers']['options']['iqr']['multipliers']:
                outlier_strategies.append(('iqr', multiplier))
        if self.params['outliers']['options']['zscore']['enabled']:
            for threshold in self.params['outliers']['options']['zscore']['thresholds']:
                outlier_strategies.append(('zscore', threshold))
        if self.params['outliers']['options']['isolation_forest']['enabled']:
            for contamination in self.params['outliers']['options']['isolation_forest']['contamination']:
                outlier_strategies.append(('isolation_forest', contamination))
        if self.params['outliers']['options']['none']['enabled']:
            outlier_strategies.append(('none', None))
        
        # Outlier handling
        outlier_handling = []
        for handling in ['cap', 'remove', 'flag']:
            if self.params['outliers']['handling_options'][handling]['enabled']:
                outlier_handling.append(handling)
        
        # Scaling methods
        scaling_methods = []
        if self.params['preprocessing']['options']['scaling']['enabled']:
            scaling_methods = self.params['preprocessing']['options']['scaling']['methods']
        
        # Encoding methods
        encoding_methods = []
        if self.params['preprocessing']['options']['encoding']['enabled']:
            encoding_methods = self.params['preprocessing']['options']['encoding']['methods']
        
        # Feature selection methods
        feature_selection_methods = []
        if self.params['preprocessing']['options']['feature_selection']['enabled']:
            feature_selection_methods = self.params['preprocessing']['options']['feature_selection']['methods']
        
        # Generate combinations (limit to reasonable number)
        max_combinations = 50  # Limit to prevent excessive computation
        count = 0
        
        for missing_strat in missing_strategies:
            for outlier_strat in outlier_strategies:
                for outlier_hand in outlier_handling:
                    for scaling in scaling_methods:
                        for encoding in encoding_methods:
                            for fs_method in feature_selection_methods:
                                if count >= max_combinations:
                                    break
                                
                                config = {
                                    'missing_values': {
                                        'strategy': missing_strat[0],
                                        'method': missing_strat[1]
                                    },
                                    'outliers': {
                                        'strategy': outlier_strat[0],
                                        'parameter': outlier_strat[1],
                                        'handling': outlier_hand
                                    },
                                    'preprocessing': {
                                        'scaling': scaling,
                                        'encoding': encoding,
                                        'feature_selection': fs_method
                                    }
                                }
                                combinations.append(config)
                                count += 1
                            if count >= max_combinations:
                                break
                        if count >= max_combinations:
                            break
                    if count >= max_combinations:
                        break
                if count >= max_combinations:
                    break
            if count >= max_combinations:
                break
        
        return combinations
    
    def _evaluate_combination(self, data: pd.DataFrame, config: Dict, target_column: str, problem_type: str) -> Tuple[float, pd.DataFrame]:
        """Evaluate a single transformation combination using cross-validation"""
        try:
            # Apply transformations
            transformed_data = self._apply_transformations(data, config)
            
            if problem_type == "unsupervised":
                # For unsupervised, use data quality metrics
                score = self._evaluate_unsupervised(transformed_data)
            else:
                # For supervised, use cross-validation
                score = self._evaluate_supervised(transformed_data, target_column, problem_type)
            
            return score, transformed_data
            
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            return -np.inf, data
    
    def _apply_transformations(self, data: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Apply a specific transformation configuration"""
        data_transformed = data.copy()
        
        # Handle missing values
        missing_config = config['missing_values']
        if missing_config['strategy'] == 'drop':
            data_transformed = data_transformed.dropna()
        elif missing_config['strategy'] == 'impute':
            method = missing_config['method']
            # Separate numerical and categorical columns
            numerical_cols = data_transformed.select_dtypes(include=[np.number]).columns
            categorical_cols = data_transformed.select_dtypes(include=['object', 'category']).columns
            
            # Impute numerical columns
            if len(numerical_cols) > 0:
                if method in ['mean', 'median', 'mode']:
                    imputer = SimpleImputer(strategy=method)
                    data_transformed[numerical_cols] = imputer.fit_transform(data_transformed[numerical_cols])
                else:
                    imputer = SimpleImputer(strategy='constant', fill_value=0)
                    data_transformed[numerical_cols] = imputer.fit_transform(data_transformed[numerical_cols])
            
            # Impute categorical columns
            if len(categorical_cols) > 0:
                imputer = SimpleImputer(strategy='most_frequent')
                data_transformed[categorical_cols] = imputer.fit_transform(data_transformed[categorical_cols])
                
        elif missing_config['strategy'] == 'flag':
            for col in data_transformed.columns:
                if data_transformed[col].isnull().any():
                    data_transformed[f"{col}_missing"] = data_transformed[col].isnull().astype(int)
            
            # Then impute with appropriate strategy for each data type
            numerical_cols = data_transformed.select_dtypes(include=[np.number]).columns
            categorical_cols = data_transformed.select_dtypes(include=['object', 'category']).columns
            
            if len(numerical_cols) > 0:
                data_transformed[numerical_cols] = data_transformed[numerical_cols].fillna(data_transformed[numerical_cols].median())
            if len(categorical_cols) > 0:
                data_transformed[categorical_cols] = data_transformed[categorical_cols].fillna(data_transformed[categorical_cols].mode().iloc[0] if len(data_transformed[categorical_cols].mode()) > 0 else "Unknown")
        
        # Handle outliers
        outlier_config = config['outliers']
        if outlier_config['strategy'] != 'none':
            numerical_cols = data_transformed.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if outlier_config['strategy'] == 'iqr':
                    Q1, Q3 = data_transformed[col].quantile([0.25, 0.75])
                    IQR = Q3 - Q1
                    multiplier = outlier_config['parameter']
                    lower_bound = Q1 - multiplier * IQR
                    upper_bound = Q3 + multiplier * IQR
                    
                    if outlier_config['handling'] == 'cap':
                        data_transformed[col] = data_transformed[col].clip(lower=lower_bound, upper=upper_bound)
                    elif outlier_config['handling'] == 'remove':
                        mask = (data_transformed[col] >= lower_bound) & (data_transformed[col] <= upper_bound)
                        data_transformed = data_transformed[mask]
                    elif outlier_config['handling'] == 'flag':
                        outlier_mask = (data_transformed[col] < lower_bound) | (data_transformed[col] > upper_bound)
                        data_transformed[f"{col}_outlier"] = outlier_mask.astype(int)
                
                elif outlier_config['strategy'] == 'zscore':
                    z_scores = np.abs(stats.zscore(data_transformed[col]))
                    threshold = outlier_config['parameter']
                    outlier_mask = z_scores > threshold
                    
                    if outlier_config['handling'] == 'cap':
                        data_transformed[col] = data_transformed[col].clip(
                            lower=data_transformed[col][z_scores < threshold].min(),
                            upper=data_transformed[col][z_scores < threshold].max()
                        )
                    elif outlier_config['handling'] == 'remove':
                        data_transformed = data_transformed[~outlier_mask]
                    elif outlier_config['handling'] == 'flag':
                        data_transformed[f"{col}_outlier"] = outlier_mask.astype(int)
        
        # Preprocessing
        preprocess_config = config['preprocessing']
        
        # Scaling
        if preprocess_config['scaling'] != 'none':
            numerical_cols = data_transformed.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                if preprocess_config['scaling'] == 'standard':
                    scaler = StandardScaler()
                elif preprocess_config['scaling'] == 'robust':
                    scaler = RobustScaler()
                elif preprocess_config['scaling'] == 'minmax':
                    scaler = MinMaxScaler()
                
                data_transformed[numerical_cols] = scaler.fit_transform(data_transformed[numerical_cols])
        
        # Encoding
        categorical_cols = data_transformed.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            if preprocess_config['encoding'] == 'label':
                for col in categorical_cols:
                    data_transformed[col] = pd.Categorical(data_transformed[col]).codes
            elif preprocess_config['encoding'] == 'onehot':
                data_transformed = pd.get_dummies(data_transformed, columns=categorical_cols)
        
        # Feature selection
        if preprocess_config['feature_selection'] != 'none':
            numerical_cols = data_transformed.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 10:  # Only if we have enough features
                if preprocess_config['feature_selection'] == 'variance':
                    selector = VarianceThreshold(threshold=0.01)
                else:
                    selector = SelectKBest(k=min(20, len(numerical_cols)))
                
                selected_features = selector.fit_transform(data_transformed[numerical_cols])
                selected_cols = numerical_cols[selector.get_support()]
                data_transformed = data_transformed[selected_cols]
        
        return data_transformed
    
    def _evaluate_supervised(self, data: pd.DataFrame, target_column: str, problem_type: str) -> float:
        """Evaluate supervised learning performance using cross-validation"""
        if target_column not in data.columns:
            return -np.inf
        
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Remove any remaining non-numeric columns
        X = X.select_dtypes(include=[np.number])
        
        if len(X.columns) == 0:
            return -np.inf
        
        # Handle edge cases
        if len(X) < 10:  # Too few samples
            return -np.inf
        
        if len(X.columns) < 2:  # Too few features
            return -np.inf
        
        # Remove rows with infinite or NaN values
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[mask]
        y = y[mask]
        
        if len(X) < 5:  # Too few samples after cleaning
            return -np.inf
        
        # Choose model based on problem type
        if problem_type == "classification":
            # Ensure we have at least 2 classes
            if y.nunique() < 2:
                return -np.inf
            
            # Use simpler model for small datasets
            if len(X) < 100:
                model = LogisticRegression(random_state=42, max_iter=1000)
            else:
                model = RandomForestClassifier(n_estimators=50, random_state=42)
            
            cv = StratifiedKFold(n_splits=min(3, len(X)//10), shuffle=True, random_state=42)
            scoring = 'accuracy'
        else:
            # Use simpler model for small datasets
            if len(X) < 100:
                model = LinearRegression()
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=42)
            
            cv = KFold(n_splits=min(3, len(X)//10), shuffle=True, random_state=42)
            scoring = 'r2'
        
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            return np.mean(scores)
        except Exception as e:
            logger.debug(f"Cross-validation failed: {e}")
            return -np.inf
    
    def _evaluate_unsupervised(self, data: pd.DataFrame) -> float:
        """Evaluate unsupervised data quality"""
        # Use data quality metrics
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            return -np.inf
        
        # Handle edge cases
        if len(data) < 5:  # Too few samples
            return -np.inf
        
        if len(numerical_cols) < 1:  # No numerical columns
            return -np.inf
        
        # Remove rows with infinite or NaN values
        data_clean = data[numerical_cols].replace([np.inf, -np.inf], np.nan)
        data_clean = data_clean.dropna()
        
        if len(data_clean) < 3:  # Too few samples after cleaning
            return -np.inf
        
        # Calculate quality metrics
        quality_score = 0
        
        # Variance (higher is better, but avoid division by zero)
        variances = data_clean.var()
        if len(variances) > 0:
            mean_variance = np.mean(variances)
            if np.isfinite(mean_variance):
                quality_score += mean_variance
        
        # No missing values bonus
        missing_ratio = data_clean.isnull().sum().sum() / (len(data_clean) * len(data_clean.columns))
        quality_score += (1 - missing_ratio) * 10
        
        # Reasonable data size
        if len(data_clean) > 10:
            quality_score += 5
        
        # Feature diversity (more features is better, up to a point)
        if len(data_clean.columns) > 1:
            quality_score += min(len(data_clean.columns), 20)  # Cap at 20
        
        return quality_score
    
    def _get_default_config(self) -> Dict:
        """Get a default configuration when optimization fails"""
        return {
            'missing_values': {
                'strategy': 'impute',
                'method': 'median'
            },
            'outliers': {
                'strategy': 'none',
                'parameter': None,
                'handling': 'cap'
            },
            'preprocessing': {
                'scaling': 'robust',
                'encoding': 'label',
                'feature_selection': 'none'
            }
        }
    
    def save_optimization_results(self):
        """Save optimization results to file"""
        if self.optimization_results:
            results_df = pd.DataFrame(self.optimization_results)
            results_df = results_df.sort_values('score', ascending=False)
            
            # Save detailed results
            results_df.to_csv(Path(self.config.root_dir) / "optimization_results.csv", index=False)
            
            # Save best configuration
            if self.best_config:
                import json
                with open(Path(self.config.root_dir) / "best_config.json", 'w') as f:
                    json.dump(self.best_config, f, indent=2)
            
            logger.info(f"Saved optimization results. Best score: {results_df['score'].max():.4f}")
            
            # Print top 5 results
            logger.info("Top 5 configurations:")
            for i, row in results_df.head().iterrows():
                logger.info(f"Rank {i+1}: Score {row['score']:.4f}, Config: {row['config']}") 