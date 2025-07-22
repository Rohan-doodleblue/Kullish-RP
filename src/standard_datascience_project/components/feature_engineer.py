import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, Any
from ..entity.config_entity import FeatureConfig, ModelConfig

def extract_time_features(x):
    x['hour'] = x['Timestamp'].dt.hour
    x['day_of_week'] = x['Timestamp'].dt.dayofweek
    return x[['hour', 'day_of_week']]

class FeatureEngineer:
    """Component for feature engineering and data preprocessing"""
    
    def __init__(self, feature_config: FeatureConfig, model_config: ModelConfig):
        self.feature_config = feature_config
        self.model_config = model_config
        self.preprocessor = None
        
    def create_preprocessor(self) -> ColumnTransformer:
        """Create the preprocessing pipeline"""
        
        # Numerical features transformer
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        # Categorical features transformer
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Time features transformer
        time_transformer = Pipeline(steps=[
            ('time_features', FunctionTransformer(extract_time_features, validate=False)),
            ('scaler', StandardScaler())
        ])
        
        # Create the main preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.feature_config.numerical_features),
                ('cat', categorical_transformer, self.feature_config.categorical_features),
                ('time', time_transformer, ['Timestamp'])
            ]
        )
        
        return self.preprocessor
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training by splitting and transforming"""
        
        # Create preprocessor if not already created
        if self.preprocessor is None:
            self.create_preprocessor()
        
        # Split features and target
        X = df.drop('Anomaly', axis=1)
        y = df['Anomaly']
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.model_config.test_size, 
            random_state=self.model_config.random_state
        )
        
        # Transform the data
        X_train_prepared = self.preprocessor.fit_transform(X_train)
        X_test_prepared = self.preprocessor.transform(X_test)
        
        return X_train_prepared, X_test_prepared, y_train, y_test
    
    def get_feature_names(self) -> list:
        """Get feature names after preprocessing"""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not created yet. Call create_preprocessor() first.")
        
        feature_names = []
        
        # Get numerical feature names
        feature_names.extend(self.feature_config.numerical_features)
        
        # Get categorical feature names
        cat_transformer = self.preprocessor.named_transformers_['cat']
        if hasattr(cat_transformer, 'get_feature_names_out'):
            cat_features = cat_transformer.get_feature_names_out(self.feature_config.categorical_features)
            feature_names.extend(cat_features)
        
        # Get time feature names
        feature_names.extend(['hour', 'day_of_week'])
        
        return feature_names 