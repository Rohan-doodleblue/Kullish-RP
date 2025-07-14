import os
import numpy as np
from pathlib import Path
from typing import Dict
from src.standard_datascience_project import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.ensemble import IsolationForest
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from src.standard_datascience_project.entity.config_entity import DataTransformationConfig
from src.standard_datascience_project.components.auto_optimizer import AutoOptimizer
from src.standard_datascience_project.components.transformation_evaluator import TransformationEvaluator
import pandas as pd

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    
    ## Note: You can add different data transformation techniques such as Scaler, PCA and all
    
    def transform_data(self):
        """Comprehensive data transformation pipeline with auto-optimization"""
        # Load data
        data = self._load_data()
        
        # Apply schema-based conversions
        data = self._apply_schema_conversions(data)
        
        # Detect target column
        target_column = self._detect_target_column(data, Path("temp.csv"))
        
        # Auto-optimize transformation strategies if enabled
        if self.config.transformation_params['auto_selection']['enable_cross_validation']:
            logger.info("Starting auto-optimization of transformation strategies")
            optimizer = AutoOptimizer(self.config)
            best_config = optimizer.optimize_transformations(data, target_column)
            optimizer.save_optimization_results()
            
            # Store the best configuration for use in preprocessing
            self.best_config = best_config
            
            # Apply the best configuration
            data = self._apply_optimized_transformations(data, best_config)
        else:
            # Use default transformation pipeline
            logger.info("Using default transformation pipeline")
            self.best_config = None
            data = self._handle_missing_values(data)
            data = self._handle_outliers(data)
            data = self._handle_columns(data)
            data = self._engineer_features(data)
        
        # Split data
        train, test = self._split_data(data, target_column)
        
        # Preprocess features
        train_processed, test_processed = self._preprocess_features(train, test, target_column)
        
        # Save transformed data
        self._save_transformed_data(train_processed, test_processed, target_column)
        
        # Save transformation report
        self._save_transformation_report(data, train_processed, test_processed, target_column)
        
        # Evaluate transformation quality
        logger.info("Evaluating transformation quality")
        evaluator = TransformationEvaluator(self.config)
        evaluation_results = evaluator.evaluate_transformations(
            original_data=data, 
            transformed_data=train_processed, 
            target_column=target_column
        )
        
        # Log evaluation summary
        overall_score = evaluation_results['overall_score']
        logger.info(f"Transformation Quality Score: {overall_score['score']}/100 ({overall_score['grade']})")
        
        if overall_score['score'] < 70:
            logger.warning("Low transformation quality detected. Review the pipeline.")
            for warning in evaluation_results['data_integrity']['warnings']:
                logger.warning(f"Data integrity issue: {warning}")
        
        logger.info("Data transformation completed successfully")
        print(f"Final train shape: {train_processed.shape}")
        print(f"Final test shape: {test_processed.shape}")
        print(f"Transformation Quality: {overall_score['score']}/100 ({overall_score['grade']})")
    
    def _load_data(self) -> pd.DataFrame:
        """Load data from the specified path"""
        data_path = Path(self.config.data_path)
        supported_extensions = ['.csv', '.json', '.parquet', '.xlsx', '.xls']
        
        data_file = None
        for ext in supported_extensions:
            files = list(data_path.glob(f"*{ext}"))
            if files:
                data_file = files[0]
                break
        
        if data_file is None:
            raise FileNotFoundError(f"No data files found in {data_path}")
        
        # Load data based on file type
        file_extension = data_file.suffix.lower()
        if file_extension == '.csv':
            data = pd.read_csv(data_file)
        elif file_extension == '.json':
            data = pd.read_json(data_file)
        elif file_extension == '.parquet':
            data = pd.read_parquet(data_file)
        elif file_extension in ['.xlsx', '.xls']:
            data = pd.read_excel(data_file)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        logger.info(f"Loaded data from {data_file} with shape {data.shape}")
        return data
    
    def _apply_schema_conversions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply schema-based data type conversions"""
        try:
            schema_file = Path("schema.yaml")
            if schema_file.exists():
                import yaml
                with open(schema_file, 'r') as f:
                    schema = yaml.safe_load(f)
                
                if 'COLUMNS' in schema:
                    logger.info("Applying schema-based data type conversions")
                    for col, expected_type in schema['COLUMNS'].items():
                        if col in data.columns:
                            try:
                                if expected_type == 'category':
                                    data[col] = data[col].astype('category')
                                elif expected_type == 'bool':
                                    data[col] = data[col].astype('bool')
                                elif expected_type == 'int64':
                                    data[col] = pd.to_numeric(data[col], errors='coerce').astype('Int64')
                                elif expected_type == 'float64':
                                    data[col] = pd.to_numeric(data[col], errors='coerce')
                                logger.info(f"Converted column '{col}' to {expected_type}")
                            except Exception as e:
                                logger.warning(f"Could not convert column '{col}' to {expected_type}: {e}")
            else:
                logger.info("No schema file found. Generating dynamic schema from data.")
                self._generate_dynamic_schema(data, Path("temp.csv"))
        except Exception as e:
            logger.warning(f"Schema-based conversion failed: {e}")
        
        return data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on configuration"""
        params = self.config.transformation_params['missing_values']
        strategy = params['strategy']
        
        logger.info(f"Handling missing values with strategy: {strategy}")
        
        # Calculate missing value statistics
        missing_stats = data.isnull().sum()
        missing_ratio = missing_stats / len(data)
        
        # Drop columns with too many missing values
        max_ratio = params['max_missing_ratio']
        columns_to_drop = missing_ratio[missing_ratio > max_ratio].index.tolist()
        
        if columns_to_drop:
            logger.info(f"Dropping columns with >{max_ratio*100}% missing values: {columns_to_drop}")
            data = data.drop(columns=columns_to_drop)
        
        if strategy == "drop":
            # Drop rows with any missing values
            initial_rows = len(data)
            data = data.dropna()
            dropped_rows = initial_rows - len(data)
            logger.info(f"Dropped {dropped_rows} rows with missing values")
            
        elif strategy == "impute":
            # Impute missing values
            method = params['imputation_method']
            constant_value = params['constant_value']
            
            # Separate numerical and categorical columns
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns
            
            # Impute numerical columns
            if len(numerical_cols) > 0:
                if method == "constant":
                    imputer = SimpleImputer(strategy='constant', fill_value=constant_value)
                else:
                    imputer = SimpleImputer(strategy=method)
                data[numerical_cols] = imputer.fit_transform(data[numerical_cols])
            
            # Impute categorical columns
            if len(categorical_cols) > 0:
                imputer = SimpleImputer(strategy='most_frequent')
                data[categorical_cols] = imputer.fit_transform(data[categorical_cols])
            
            logger.info(f"Imputed missing values using {method} method")
            
        elif strategy == "flag":
            # Create flag columns for missing values
            for col in data.columns:
                if data[col].isnull().any():
                    data[f"{col}_missing"] = data[col].isnull().astype(int)
            
            # Then impute with simple strategy
            data = data.fillna(data.median() if len(data.select_dtypes(include=[np.number])) > 0 else data.mode().iloc[0])
            logger.info("Created missing value flags and imputed values")
        
        return data
    
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers based on configuration"""
        params = self.config.transformation_params['outliers']
        strategy = params['strategy']
        handling = params['outlier_handling']
        
        if strategy == "none":
            return data
        
        logger.info(f"Handling outliers with strategy: {strategy}")
        
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            outliers_mask = self._detect_outliers(data[col], strategy, params)
            
            if outliers_mask.any():
                outlier_count = outliers_mask.sum()
                logger.info(f"Found {outlier_count} outliers in column {col}")
                
                if handling == "remove":
                    data = data[~outliers_mask]
                    logger.info(f"Removed {outlier_count} outlier rows")
                elif handling == "cap":
                    if strategy == "iqr":
                        Q1, Q3 = data[col].quantile([0.25, 0.75])
                        IQR = Q3 - Q1
                        lower_bound = Q1 - params['iqr_multiplier'] * IQR
                        upper_bound = Q3 + params['iqr_multiplier'] * IQR
                        data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
                    elif strategy == "zscore":
                        z_scores = np.abs(stats.zscore(data[col]))
                        threshold = params['zscore_threshold']
                        data[col] = data[col].clip(
                            lower=data[col][z_scores < threshold].min(),
                            upper=data[col][z_scores < threshold].max()
                        )
                    logger.info(f"Capped outliers in column {col}")
                elif handling == "flag":
                    data[f"{col}_outlier"] = outliers_mask.astype(int)
                    logger.info(f"Created outlier flag for column {col}")
        
        return data
    
    def _detect_outliers(self, series: pd.Series, strategy: str, params: dict) -> pd.Series:
        """Detect outliers using specified strategy"""
        if strategy == "iqr":
            Q1, Q3 = series.quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower_bound = Q1 - params['iqr_multiplier'] * IQR
            upper_bound = Q3 + params['iqr_multiplier'] * IQR
            return (series < lower_bound) | (series > upper_bound)
        
        elif strategy == "zscore":
            z_scores = np.abs(stats.zscore(series))
            return z_scores > params['zscore_threshold']
        
        elif strategy == "isolation_forest":
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            predictions = iso_forest.fit_predict(series.values.reshape(-1, 1))
            return predictions == -1
        
        return pd.Series([False] * len(series))
    
    def _handle_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle column selection and filtering"""
        params = self.config.transformation_params['columns']
        
        logger.info("Handling column selection and filtering")
        
        # Drop high cardinality categorical columns
        if params['drop_high_cardinality']:
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns
            high_cardinality_cols = []
            
            for col in categorical_cols:
                if data[col].nunique() > params['max_cardinality']:
                    high_cardinality_cols.append(col)
            
            if high_cardinality_cols:
                logger.info(f"Dropping high cardinality columns: {high_cardinality_cols}")
                data = data.drop(columns=high_cardinality_cols)
        
        # Drop low variance columns
        if params['drop_low_variance']:
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            low_variance_cols = []
            
            for col in numerical_cols:
                variance_ratio = data[col].var() / data[col].mean()**2 if data[col].mean() != 0 else 0
                if variance_ratio < params['min_variance_ratio']:
                    low_variance_cols.append(col)
            
            if low_variance_cols:
                logger.info(f"Dropping low variance columns: {low_variance_cols}")
                data = data.drop(columns=low_variance_cols)
        
        # Drop duplicate columns
        if params['drop_duplicates']:
            initial_cols = len(data.columns)
            data = data.loc[:, ~data.columns.duplicated()]
            dropped_cols = initial_cols - len(data.columns)
            if dropped_cols > 0:
                logger.info(f"Dropped {dropped_cols} duplicate columns")
        
        # Drop ID-like columns
        if params['drop_id_columns']:
            id_like_cols = []
            for col in data.columns:
                col_lower = col.lower()
                if any(id_word in col_lower for id_word in ['id', 'index', 'no', 'number']):
                    if data[col].nunique() == len(data):  # All unique values
                        id_like_cols.append(col)
            
            if id_like_cols:
                logger.info(f"Dropping ID-like columns: {id_like_cols}")
                data = data.drop(columns=id_like_cols)
        
        return data
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer new features based on configuration"""
        params = self.config.transformation_params['feature_engineering']
        
        logger.info("Engineering features")
        
        # Create date features
        if params['create_date_features']:
            date_columns = []
            for col in data.columns:
                if data[col].dtype == 'object':
                    try:
                        pd.to_datetime(data[col])
                        date_columns.append(col)
                    except:
                        continue
            
            for col in date_columns:
                data[col] = pd.to_datetime(data[col])
                data[f"{col}_year"] = data[col].dt.year
                data[f"{col}_month"] = data[col].dt.month
                data[f"{col}_day"] = data[col].dt.day
                data[f"{col}_dayofweek"] = data[col].dt.dayofweek
                logger.info(f"Created date features for {col}")
        
        # Create interaction features
        if params['create_interaction_features']:
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) >= 2:
                # Create pairwise interactions for top correlated features
                corr_matrix = data[numerical_cols].corr().abs()
                upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                high_corr_pairs = [(col1, col2) for col1, col2 in zip(upper_tri.index, upper_tri.columns) 
                                  if upper_tri.loc[col1, col2] > 0.7]
                
                for col1, col2 in high_corr_pairs[:5]:  # Limit to 5 interactions
                    data[f"{col1}_{col2}_interaction"] = data[col1] * data[col2]
                    logger.info(f"Created interaction feature: {col1}_{col2}_interaction")
        
        return data
    
    def _split_data(self, data: pd.DataFrame, target_column: str) -> tuple:
        """Split data into train and test sets"""
        if target_column and target_column in data.columns:
            logger.info(f"Splitting data with target column: {target_column}")
            train, test = train_test_split(data, test_size=0.25, random_state=42, stratify=data[target_column])
        else:
            logger.info("Splitting data without stratification")
            train, test = train_test_split(data, test_size=0.25, random_state=42)
        
        return train, test
    
    def _preprocess_features(self, train: pd.DataFrame, test: pd.DataFrame, target_column: str) -> tuple:
        """Preprocess features (scaling, encoding)"""
        params = self.config.transformation_params['preprocessing']
        
        logger.info("Preprocessing features")
        
        # Separate features and target
        if target_column and target_column in train.columns:
            train_features = train.drop(columns=[target_column])
            train_target = train[target_column]
            test_features = test.drop(columns=[target_column])
            test_target = test[target_column]
        else:
            train_features = train.copy()
            test_features = test.copy()
            train_target = None
            test_target = None
        
        # Scale numerical features
        if params['options']['scaling']['enabled']:
            numerical_cols = train_features.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                # Use the scaling method from the optimized configuration if available
                scaling_method = 'robust'  # default
                if hasattr(self, 'best_config') and self.best_config:
                    scaling_method = self.best_config['preprocessing']['scaling']
                else:
                    # Use default from params if available
                    try:
                        scaling_method = params['options']['scaling']['methods'][0]
                    except:
                        scaling_method = 'robust'
                
                if scaling_method == 'standard':
                    scaler = StandardScaler()
                elif scaling_method == 'robust':
                    scaler = RobustScaler()
                elif scaling_method == 'minmax':
                    scaler = MinMaxScaler()
                else:
                    scaler = RobustScaler()  # default
                
                train_features[numerical_cols] = scaler.fit_transform(train_features[numerical_cols])
                test_features[numerical_cols] = scaler.transform(test_features[numerical_cols])
                
                # Save scaler
                import joblib
                joblib.dump(scaler, os.path.join(self.config.root_dir, "scaler.pkl"))
                logger.info(f"Scaled numerical features using {scaling_method}")
        
        # Encode categorical features
        categorical_cols = train_features.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            # Use the encoding method from the optimized configuration if available
            encoding_method = 'label'  # default
            if hasattr(self, 'best_config') and self.best_config:
                encoding_method = self.best_config['preprocessing']['encoding']
            
            if encoding_method == "label":
                label_encoders = {}
                for col in categorical_cols:
                    le = LabelEncoder()
                    train_features[col] = le.fit_transform(train_features[col].astype(str))
                    test_features[col] = le.transform(test_features[col].astype(str))
                    label_encoders[col] = le
                
                # Save encoders
                import joblib
                joblib.dump(label_encoders, os.path.join(self.config.root_dir, "label_encoders.pkl"))
                logger.info("Label encoded categorical features")
            
            elif encoding_method == "onehot":
                train_features = pd.get_dummies(train_features, columns=categorical_cols)
                test_features = pd.get_dummies(test_features, columns=categorical_cols)
                
                # Align columns
                missing_cols = set(train_features.columns) - set(test_features.columns)
                for col in missing_cols:
                    test_features[col] = 0
                test_features = test_features[train_features.columns]
                
                logger.info("One-hot encoded categorical features")
        
        # Recombine features and target
        if train_target is not None:
            train_processed = pd.concat([train_features, train_target], axis=1)
            test_processed = pd.concat([test_features, test_target], axis=1)
        else:
            train_processed = train_features
            test_processed = test_features
        
        return train_processed, test_processed
    
    def _save_transformed_data(self, train: pd.DataFrame, test: pd.DataFrame, target_column: str):
        """Save transformed data"""
        # Save full datasets
        train.to_csv(os.path.join(self.config.root_dir, "train_transformed.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test_transformed.csv"), index=False)
        
        # If target column exists, create separate feature/target files
        if target_column and target_column in train.columns:
            train_features = train.drop(columns=[target_column])
            train_target = train[target_column]
            test_features = test.drop(columns=[target_column])
            test_target = test[target_column]
            
            train_features.to_csv(os.path.join(self.config.root_dir, "train_features_transformed.csv"), index=False)
            train_target.to_csv(os.path.join(self.config.root_dir, "train_target_transformed.csv"), index=False)
            test_features.to_csv(os.path.join(self.config.root_dir, "test_features_transformed.csv"), index=False)
            test_target.to_csv(os.path.join(self.config.root_dir, "test_target_transformed.csv"), index=False)
            
            # Save target info
            target_info = {
                'target_column': target_column,
                'target_type': str(train_target.dtype),
                'unique_values': int(train_target.nunique()),
                'is_classification': train_target.nunique() <= 10
            }
            
            import json
            with open(os.path.join(self.config.root_dir, "target_info.json"), 'w') as f:
                json.dump(target_info, f, indent=2)
            
            logger.info(f"Saved feature/target split with target column: {target_column}")
    
    def _save_transformation_report(self, original_data: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, target_column: str):
        """Save transformation report"""
        report = {
            'transformation_summary': {
                'original_shape': original_data.shape,
                'train_shape': train.shape,
                'test_shape': test.shape,
                'target_column': target_column,
                'columns_removed': len(original_data.columns) - len(train.columns),
                'rows_removed': len(original_data) - len(train) - len(test)
            },
            'transformation_params': self.config.transformation_params
        }
        
        import json
        with open(os.path.join(self.config.root_dir, "transformation_report.json"), 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("Saved transformation report")
    
    def _apply_optimized_transformations(self, data: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Apply the optimized transformation configuration"""
        logger.info("Applying optimized transformation configuration")
        
        # Apply missing value handling
        missing_config = config['missing_values']
        if missing_config['strategy'] == 'drop':
            data = data.dropna()
        elif missing_config['strategy'] == 'impute':
            method = missing_config['method']
            # Separate numerical and categorical columns
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns
            
            # Impute numerical columns
            if len(numerical_cols) > 0:
                if method in ['mean', 'median', 'mode']:
                    imputer = SimpleImputer(strategy=method)
                    data[numerical_cols] = imputer.fit_transform(data[numerical_cols])
                else:
                    imputer = SimpleImputer(strategy='constant', fill_value=0)
                    data[numerical_cols] = imputer.fit_transform(data[numerical_cols])
            
            # Impute categorical columns
            if len(categorical_cols) > 0:
                imputer = SimpleImputer(strategy='most_frequent')
                data[categorical_cols] = imputer.fit_transform(data[categorical_cols])
                
        elif missing_config['strategy'] == 'flag':
            for col in data.columns:
                if data[col].isnull().any():
                    data[f"{col}_missing"] = data[col].isnull().astype(int)
            
            # Then impute with appropriate strategy for each data type
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns
            
            if len(numerical_cols) > 0:
                data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].median())
            if len(categorical_cols) > 0:
                data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0] if len(data[categorical_cols].mode()) > 0 else "Unknown")
        
        # Apply outlier handling
        outlier_config = config['outliers']
        if outlier_config['strategy'] != 'none':
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if outlier_config['strategy'] == 'iqr':
                    Q1, Q3 = data[col].quantile([0.25, 0.75])
                    IQR = Q3 - Q1
                    multiplier = outlier_config['parameter']
                    lower_bound = Q1 - multiplier * IQR
                    upper_bound = Q3 + multiplier * IQR
                    
                    if outlier_config['handling'] == 'cap':
                        data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
                    elif outlier_config['handling'] == 'remove':
                        mask = (data[col] >= lower_bound) & (data[col] <= upper_bound)
                        data = data[mask]
                    elif outlier_config['handling'] == 'flag':
                        outlier_mask = (data[col] < lower_bound) | (data[col] > upper_bound)
                        data[f"{col}_outlier"] = outlier_mask.astype(int)
                
                elif outlier_config['strategy'] == 'zscore':
                    z_scores = np.abs(stats.zscore(data[col]))
                    threshold = outlier_config['parameter']
                    outlier_mask = z_scores > threshold
                    
                    if outlier_config['handling'] == 'cap':
                        data[col] = data[col].clip(
                            lower=data[col][z_scores < threshold].min(),
                            upper=data[col][z_scores < threshold].max()
                        )
                    elif outlier_config['handling'] == 'remove':
                        data = data[~outlier_mask]
                    elif outlier_config['handling'] == 'flag':
                        data[f"{col}_outlier"] = outlier_mask.astype(int)
        
        # Apply preprocessing
        preprocess_config = config['preprocessing']
        
        # Scaling
        if preprocess_config['scaling'] != 'none':
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                if preprocess_config['scaling'] == 'standard':
                    scaler = StandardScaler()
                elif preprocess_config['scaling'] == 'robust':
                    scaler = RobustScaler()
                elif preprocess_config['scaling'] == 'minmax':
                    scaler = MinMaxScaler()
                
                data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
        
        # Encoding
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            if preprocess_config['encoding'] == 'label':
                for col in categorical_cols:
                    data[col] = pd.Categorical(data[col]).codes
            elif preprocess_config['encoding'] == 'onehot':
                data = pd.get_dummies(data, columns=categorical_cols)
        
        # Feature selection
        if preprocess_config['feature_selection'] != 'none':
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 10:
                if preprocess_config['feature_selection'] == 'variance':
                    selector = VarianceThreshold(threshold=0.01)
                else:
                    selector = SelectKBest(k=min(20, len(numerical_cols)))
                
                selected_features = selector.fit_transform(data[numerical_cols])
                selected_cols = numerical_cols[selector.get_support()]
                data = data[selected_cols]
        
        logger.info(f"Applied optimized transformations. Final shape: {data.shape}")
        return data
    
    def train_test_splitting(self):
        """Legacy method - now calls the new transform_data method"""
        self.transform_data()
    
    def _generate_dynamic_schema(self, data: pd.DataFrame, data_file: Path):
        """Generate a dynamic schema based on the actual data"""
        try:
            import yaml
            
            # Determine data types based on pandas dtypes and data characteristics
            schema_columns = {}
            for col in data.columns:
                dtype = str(data[col].dtype)
                
                # Map pandas dtypes to schema dtypes
                if dtype == 'int64':
                    schema_columns[col] = 'int64'
                elif dtype == 'float64':
                    schema_columns[col] = 'float64'
                elif dtype == 'bool':
                    schema_columns[col] = 'bool'
                elif dtype == 'object':
                    # Check if it's categorical (low cardinality) or string
                    unique_ratio = data[col].nunique() / len(data)
                    if unique_ratio < 0.5:  # Less than 50% unique values
                        schema_columns[col] = 'category'
                    else:
                        schema_columns[col] = 'object'
                else:
                    schema_columns[col] = 'object'
            
            # Create schema dictionary
            schema = {
                'COLUMNS': schema_columns,
                'DATA_INFO': {
                    'columns': len(data.columns),
                    'rows': len(data),
                    'file_name': data_file.name,
                    'file_type': data_file.suffix[1:] if data_file.suffix else 'unknown'
                }
            }
            
            # Save dynamic schema
            schema_path = Path("schema_dynamic.yaml")
            with open(schema_path, 'w') as f:
                yaml.dump(schema, f, default_flow_style=False)
            
            logger.info(f"Dynamic schema generated and saved to {schema_path}")
            logger.info(f"Schema includes {len(schema_columns)} columns")
            
        except Exception as e:
            logger.error(f"Failed to generate dynamic schema: {e}")
    
    def _update_schema_if_needed(self, data: pd.DataFrame, data_file: Path):
        """Update existing schema if there are data type mismatches"""
        try:
            schema_file = Path("schema.yaml")
            if not schema_file.exists():
                return
            
            import yaml
            with open(schema_file, 'r') as f:
                schema = yaml.safe_load(f)
            
            if 'COLUMNS' not in schema:
                return
            
            # Check for mismatches and update schema
            updated = False
            for col in data.columns:
                if col in schema['COLUMNS']:
                    expected_type = schema['COLUMNS'][col]
                    actual_type = str(data[col].dtype)
                    
                    # Update schema for common mismatches
                    if expected_type == 'category' and actual_type == 'object':
                        schema['COLUMNS'][col] = 'object'
                        updated = True
                        logger.info(f"Updated schema: {col} from category to object")
                    elif expected_type == 'bool' and actual_type == 'bool':
                        # This is fine, no update needed
                        pass
                    elif actual_type not in ['int64', 'float64', 'object', 'bool']:
                        schema['COLUMNS'][col] = 'object'
                        updated = True
                        logger.info(f"Updated schema: {col} to object (unknown type: {actual_type})")
            
            # Save updated schema
            if updated:
                # First create a backup
                schema_backup = Path("schema_backup.yaml")
                with open(schema_backup, 'w') as f:
                    yaml.dump(schema, f, default_flow_style=False)
                logger.info(f"Schema backed up to {schema_backup}")
                
                # Then update the original schema file
                with open(schema_file, 'w') as f:
                    yaml.dump(schema, f, default_flow_style=False)
                logger.info(f"Original schema file {schema_file} updated")
                
        except Exception as e:
            logger.warning(f"Failed to update schema: {e}")
    
    def _detect_target_column(self, data: pd.DataFrame, data_file: Path) -> str:
        """Dynamically detect target column based on various heuristics"""
        try:
            # First, check if schema has a target column defined
            schema_file = Path("schema.yaml")
            if schema_file.exists():
                import yaml
                with open(schema_file, 'r') as f:
                    schema = yaml.safe_load(f)
                
                if 'TARGET_COLUMN' in schema and 'name' in schema['TARGET_COLUMN']:
                    target_name = schema['TARGET_COLUMN']['name']
                    if target_name in data.columns:
                        logger.info(f"Using target column from schema: {target_name}")
                        return target_name
            
            # Auto-detect target column based on common patterns
            target_candidates = []
            
            # Common target column names
            common_targets = [
                'target', 'label', 'class', 'category', 'outcome', 'result',
                'quality', 'score', 'rating', 'prediction', 'target_variable',
                'dependent_variable', 'y', 'response', 'output'
            ]
            
            # Check for common target names
            for col in data.columns:
                col_lower = col.lower()
                if any(target in col_lower for target in common_targets):
                    target_candidates.append((col, 'common_name'))
            
            # Check for binary columns (potential classification targets)
            for col in data.columns:
                if data[col].dtype == 'bool':
                    target_candidates.append((col, 'binary'))
                elif data[col].dtype == 'object':
                    unique_vals = data[col].nunique()
                    if unique_vals == 2:
                        target_candidates.append((col, 'binary_object'))
            
            # Check for numeric columns with reasonable range (potential regression targets)
            for col in data.columns:
                if data[col].dtype in ['int64', 'float64']:
                    # Skip columns that look like IDs or indices
                    if not any(id_word in col.lower() for id_word in ['id', 'index', 'no', 'number']):
                        # Check if it's not all unique (not an ID)
                        if data[col].nunique() < len(data) * 0.9:  # Less than 90% unique
                            target_candidates.append((col, 'numeric'))
            
            # Score and select best target column
            if target_candidates:
                # Prioritize by type
                type_scores = {
                    'common_name': 10,
                    'binary': 8,
                    'binary_object': 7,
                    'numeric': 5
                }
                
                scored_candidates = []
                for col, col_type in target_candidates:
                    score = type_scores.get(col_type, 0)
                    # Bonus for columns with fewer missing values
                    missing_ratio = data[col].isnull().sum() / len(data)
                    score -= missing_ratio * 5  # Penalty for missing values
                    scored_candidates.append((col, score, col_type))
                
                # Sort by score and return the best
                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                best_target = scored_candidates[0]
                
                logger.info(f"Auto-detected target column: {best_target[0]} (type: {best_target[2]}, score: {best_target[1]:.2f})")
                return best_target[0]
            
            logger.info("No suitable target column detected")
            return None
            
        except Exception as e:
            logger.error(f"Error detecting target column: {e}")
            return None