import pandas as pd
import os
from pathlib import Path
from src.standard_datascience_project import logger
from src.standard_datascience_project.utils.common import save_json
import yaml
from typing import Dict, Any, List, Union

class SchemaGenerator:
    """Automatically generates schema based on data structure"""
    
    def __init__(self, data_dir: Union[str, Path]):
        # Ensure data_dir is a Path object
        self.data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
        
    def detect_file_type(self, file_path: Path) -> str:
        """Detect the type of data file"""
        extension = file_path.suffix.lower()
        
        if extension == '.csv':
            return 'csv'
        elif extension == '.json':
            return 'json'
        elif extension == '.parquet':
            return 'parquet'
        elif extension == '.xlsx' or extension == '.xls':
            return 'excel'
        else:
            return 'unknown'
    
    def load_data(self, file_path: Path) -> pd.DataFrame:
        """Load data based on file type"""
        file_type = self.detect_file_type(file_path)
        
        try:
            if file_type == 'csv':
                return pd.read_csv(file_path)
            elif file_type == 'json':
                return pd.read_json(file_path)
            elif file_type == 'parquet':
                return pd.read_parquet(file_path)
            elif file_type == 'excel':
                return pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise
    
    def infer_column_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Infer column data types from pandas DataFrame"""
        column_types = {}
        
        for column in df.columns:
            dtype = str(df[column].dtype)
            
            # Map pandas dtypes to schema types
            if 'int' in dtype:
                column_types[column] = 'int64'
            elif 'float' in dtype:
                column_types[column] = 'float64'
            elif 'bool' in dtype:
                column_types[column] = 'bool'
            elif 'datetime' in dtype:
                column_types[column] = 'datetime64'
            else:
                # For object/string types, check if it's categorical
                if df[column].nunique() / len(df) < 0.5:  # Less than 50% unique values
                    column_types[column] = 'category'
                else:
                    column_types[column] = 'object'
        
        return column_types
    
    def suggest_target_column(self, df: pd.DataFrame) -> str:
        """Suggest target column based on common patterns"""
        # Common target column names
        target_patterns = [
            'target', 'label', 'class', 'category', 'quality', 'rating',
            'score', 'outcome', 'result', 'prediction', 'y', 'target_variable'
        ]
        
        # Check for exact matches
        for pattern in target_patterns:
            if pattern in df.columns:
                return pattern
        
        # Check for columns ending with target patterns
        for col in df.columns:
            for pattern in target_patterns:
                if col.lower().endswith(pattern):
                    return col
        
        # If no obvious target, return the last column (common convention)
        return df.columns[-1]
    
    def generate_schema(self, data_file: Path) -> Dict[str, Any]:
        """Generate complete schema from data file"""
        logger.info(f"Generating schema for {data_file}")
        
        # Load data
        df = self.load_data(data_file)
        logger.info(f"Loaded data with shape: {df.shape}")
        
        # Infer column types
        column_types = self.infer_column_types(df)
        logger.info(f"Inferred {len(column_types)} column types")
        
        # Suggest target column
        target_column = self.suggest_target_column(df)
        logger.info(f"Suggested target column: {target_column}")
        
        # Create schema structure
        schema = {
            'COLUMNS': column_types,
            'TARGET_COLUMN': {
                'name': target_column
            },
            'DATA_INFO': {
                'rows': len(df),
                'columns': len(df.columns),
                'file_type': self.detect_file_type(data_file),
                'file_name': data_file.name
            }
        }
        
        return schema
    
    def find_data_files(self) -> List[Path]:
        """Find data files in the data directory"""
        data_files = []
        supported_extensions = ['.csv', '.json', '.parquet', '.xlsx', '.xls']
        
        logger.info(f"Searching for data files in: {self.data_dir}")
        
        for ext in supported_extensions:
            files = list(self.data_dir.glob(f"*{ext}"))
            data_files.extend(files)
            if files:
                logger.info(f"Found {len(files)} files with extension {ext}")
        
        logger.info(f"Total data files found: {len(data_files)}")
        return data_files
    
    def update_schema_yaml(self, schema: Dict[str, Any], schema_path: Path = None):
        """Update schema.yaml file with generated schema"""
        if schema_path is None:
            schema_path = Path("schema.yaml")
        
        try:
            with open(schema_path, 'w') as f:
                yaml.dump(schema, f, default_flow_style=False, indent=2)
            logger.info(f"Schema updated in {schema_path}")
        except Exception as e:
            logger.error(f"Error updating schema: {e}")
            raise
    
    def generate_and_update_schema(self) -> Dict[str, Any]:
        """Main method to generate and update schema"""
        # Find data files
        data_files = self.find_data_files()
        
        if not data_files:
            raise FileNotFoundError(f"No data files found in {self.data_dir}")
        
        # Use the first data file found
        data_file = data_files[0]
        logger.info(f"Using data file: {data_file}")
        
        # Generate schema
        schema = self.generate_schema(data_file)
        
        # Update schema.yaml
        self.update_schema_yaml(schema)
        
        return schema 