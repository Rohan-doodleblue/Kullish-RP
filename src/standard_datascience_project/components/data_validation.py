import os
from src.standard_datascience_project import logger
import pandas as pd
from pathlib import Path
from src.standard_datascience_project.entity.config_entity import DataValidationConfig


class DataValiadtion:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def find_data_file(self) -> Path:
        """Find the first data file in the unzip directory"""
        unzip_dir = Path(self.config.unzip_data_dir)
        supported_extensions = ['.csv', '.json', '.parquet', '.xlsx', '.xls']
        
        for ext in supported_extensions:
            files = list(unzip_dir.glob(f"*{ext}"))
            if files:
                data_file = files[0]
                logger.info(f"Found data file: {data_file}")
                return data_file
        
        raise FileNotFoundError(f"No data files found in {unzip_dir}")

    def load_data(self, file_path: Path) -> pd.DataFrame:
        """Load data based on file type"""
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension == '.csv':
                return pd.read_csv(file_path)
            elif file_extension == '.json':
                return pd.read_json(file_path)
            elif file_extension == '.parquet':
                return pd.read_parquet(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                return pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise

    def validate_all_columns(self)-> bool:
        try:
            validation_status = None

            # Find the data file dynamically
            data_file = self.find_data_file()
            
            # Load data based on file type
            data = self.load_data(data_file)
            all_cols = list(data.columns)

            all_schema = self.config.all_schema.keys()

            logger.info(f"Validating {len(all_cols)} columns against schema")
            logger.info(f"Data columns: {all_cols}")
            logger.info(f"Schema columns: {list(all_schema)}")
            
            # Check if all data columns exist in schema
            missing_columns = [col for col in all_cols if col not in all_schema]
            
            if missing_columns:
                validation_status = False
                logger.error(f"Validation failed. Missing columns in schema: {missing_columns}")
            else:
                validation_status = True
                logger.info("All columns validated successfully")

            # Write validation status to file
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {validation_status}\n")
                if missing_columns:
                    f.write(f"Missing columns: {missing_columns}\n")
                f.write(f"Total columns validated: {len(all_cols)}\n")
                f.write(f"Data file: {data_file}\n")

            return validation_status
        
        except Exception as e:
            logger.error(f"Validation error: {e}")
            # Write error status to file
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: False\n")
                f.write(f"Error: {str(e)}\n")
            raise e

    

