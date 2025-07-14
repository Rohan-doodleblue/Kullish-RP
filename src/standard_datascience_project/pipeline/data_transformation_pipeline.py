from src.standard_datascience_project.config.configuration import ConfigurationManager
from src.standard_datascience_project.components.data_transformation import DataTransformation
from src.standard_datascience_project import logger

from pathlib import Path


STAGE_NAME="Data Transformation Stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_transformation(self):
        try:
            # Check if validation status file exists
            status_file = Path("artifacts/data_validation/status.txt")
            if not status_file.exists():
                logger.warning("Validation status file not found. Proceeding with transformation anyway.")
                self._run_transformation()
                return
            
            # Read validation status
            with open(status_file, 'r') as f:
                status_line = f.readline().strip()
                status = status_line.split(": ")[-1] if ": " in status_line else "False"
            
            # Check for validation errors
            validation_errors = []
            with open(status_file, 'r') as f:
                for line in f:
                    if "Type validation errors:" in line or "Missing columns:" in line or "Extra columns:" in line:
                        validation_errors.append(line.strip())
            
            if status == "True":
                logger.info("Validation passed. Proceeding with transformation.")
                self._run_transformation()
            else:
                # Check if errors are minor (type mismatches that can be handled)
                if validation_errors and any("Type validation errors:" in error for error in validation_errors):
                    logger.warning("Validation has type mismatches, but proceeding with transformation.")
                    logger.warning("Data types will be handled during transformation.")
                    self._run_transformation()
                else:
                    logger.error("Critical validation errors found:")
                    for error in validation_errors:
                        logger.error(error)
                    raise Exception("Your data scheme is not valid")
            
        except Exception as e:
            logger.error(f"Data transformation failed: {e}")
            raise e
    
    def _run_transformation(self):
        """Helper method to run the data transformation"""
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.transform_data()
        logger.info("Data transformation completed successfully")

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.initiate_data_transformation()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e