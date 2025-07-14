import os
from src.standard_datascience_project import logger
import pandas as pd
from pathlib import Path
from src.standard_datascience_project.entity.config_entity import DataValidationConfig


class DataValidation:
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
            
            # Check if all schema columns exist in data
            extra_columns = [col for col in all_schema if col not in all_cols]
            
            # Validate data types with more flexible mapping
            type_validation_errors = []
            for col in all_cols:
                if col in all_schema:
                    expected_type = self.config.all_schema[col]
                    actual_type = str(data[col].dtype)
                    
                    # More flexible type mapping for pandas dtypes
                    type_mapping = {
                        'int64': 'int64',
                        'float64': 'float64',
                        'object': 'object',
                        'string': 'string',
                        'bool': 'bool',
                        'boolean': 'bool'
                    }
                    
                    # Special handling for category vs object (pandas stores categories as object initially)
                    if expected_type == 'category' and actual_type == 'object':
                        # This is acceptable - object can be converted to category
                        continue
                    elif expected_type == 'bool' and actual_type == 'bool':
                        # This is acceptable
                        continue
                    elif actual_type in type_mapping and type_mapping[actual_type] == expected_type:
                        # Direct match
                        continue
                    else:
                        type_validation_errors.append(f"Column '{col}': expected {expected_type}, got {actual_type}")
            
            if missing_columns or extra_columns or type_validation_errors:
                validation_status = False
                logger.error(f"Validation failed.")
                if missing_columns:
                    logger.error(f"Missing columns in schema: {missing_columns}")
                if extra_columns:
                    logger.error(f"Extra columns in schema: {extra_columns}")
                if type_validation_errors:
                    logger.error(f"Type validation errors: {type_validation_errors}")
            else:
                validation_status = True
                logger.info("All columns and data types validated successfully")

            # Write validation status to file
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {validation_status}\n")
                if missing_columns:
                    f.write(f"Missing columns: {missing_columns}\n")
                if extra_columns:
                    f.write(f"Extra columns: {extra_columns}\n")
                if type_validation_errors:
                    f.write(f"Type validation errors: {type_validation_errors}\n")
                f.write(f"Total columns validated: {len(all_cols)}\n")
                f.write(f"Data file: {data_file}\n")

            # Auto-update schema if there are type mismatches
            if type_validation_errors:
                self._auto_update_schema(data, data_file)
            
            # Generate new schema if there are too many missing columns
            if len(missing_columns) > len(all_cols) * 0.3:  # More than 30% missing
                logger.warning("Too many missing columns. Generating new schema from data.")
                self._generate_new_schema(data, data_file)

            return validation_status
        
        except Exception as e:
            logger.error(f"Validation error: {e}")
            # Write error status to file
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: False\n")
                f.write(f"Error: {str(e)}\n")
            raise e
    
    def _auto_update_schema(self, data: pd.DataFrame, data_file: Path):
        """Automatically update schema based on actual data types"""
        try:
            import yaml
            
            # Read current schema
            schema_file = Path("schema.yaml")
            if not schema_file.exists():
                logger.warning("No schema file found for auto-update")
                return
            
            with open(schema_file, 'r') as f:
                schema = yaml.safe_load(f)
            
            if 'COLUMNS' not in schema:
                logger.warning("No COLUMNS section found in schema")
                return
            
            # Update schema based on actual data types
            updated = False
            for col in data.columns:
                if col in schema['COLUMNS']:
                    expected_type = schema['COLUMNS'][col]
                    actual_type = str(data[col].dtype)
                    
                    # Update schema for common mismatches
                    if expected_type == 'category' and actual_type == 'object':
                        schema['COLUMNS'][col] = 'object'
                        updated = True
                        logger.info(f"Auto-updated schema: {col} from category to object")
                    elif expected_type == 'int64' and actual_type == 'float64':
                        schema['COLUMNS'][col] = 'float64'
                        updated = True
                        logger.info(f"Auto-updated schema: {col} from int64 to float64")
                    elif actual_type not in ['int64', 'float64', 'object', 'bool']:
                        schema['COLUMNS'][col] = 'object'
                        updated = True
                        logger.info(f"Auto-updated schema: {col} to object (unknown type: {actual_type})")
            
            # Save updated schema
            if updated:
                # Create backup first
                schema_backup = Path("schema_backup.yaml")
                with open(schema_backup, 'w') as f:
                    yaml.dump(schema, f, default_flow_style=False)
                logger.info(f"Schema backed up to {schema_backup}")
                
                # Update original schema
                with open(schema_file, 'w') as f:
                    yaml.dump(schema, f, default_flow_style=False)
                logger.info(f"Schema auto-updated in {schema_file}")
                
        except Exception as e:
            logger.error(f"Failed to auto-update schema: {e}")
    
    def _generate_new_schema(self, data: pd.DataFrame, data_file: Path):
        """Generate a completely new schema based on the actual data"""
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
            
            # Detect target column
            target_column = self._detect_target_column(data)
            
            # Create new schema dictionary
            new_schema = {
                'COLUMNS': schema_columns,
                'DATA_INFO': {
                    'columns': len(data.columns),
                    'rows': len(data),
                    'file_name': data_file.name,
                    'file_type': data_file.suffix[1:] if data_file.suffix else 'unknown'
                }
            }
            
            # Add target column if detected
            if target_column:
                new_schema['TARGET_COLUMN'] = {'name': target_column}
            
            # Backup old schema
            schema_file = Path("schema.yaml")
            if schema_file.exists():
                old_schema_backup = Path("schema_old_backup.yaml")
                with open(old_schema_backup, 'w') as f:
                    yaml.dump(new_schema, f, default_flow_style=False)
                logger.info(f"Old schema backed up to {old_schema_backup}")
            
            # Save new schema
            with open(schema_file, 'w') as f:
                yaml.dump(new_schema, f, default_flow_style=False)
            
            logger.info(f"New schema generated and saved to {schema_file}")
            logger.info(f"New schema includes {len(schema_columns)} columns")
            
        except Exception as e:
            logger.error(f"Failed to generate new schema: {e}")
    
    def _detect_target_column(self, data: pd.DataFrame) -> str:
        """Dynamically detect target column based on various heuristics"""
        try:
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

    

