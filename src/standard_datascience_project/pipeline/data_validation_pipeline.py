from src.standard_datascience_project.config.configuration import ConfigurationManager
from src.standard_datascience_project.components.data_validation import DataValiadtion
from src.standard_datascience_project import logger

STAGE_NAME = "Data Validation stage"

class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_validation(self):
        try:
            config = ConfigurationManager()
            data_validation_config = config.get_data_validation_config()
            data_validation = DataValiadtion(config=data_validation_config)
            validation_status = data_validation.validate_all_columns()
            
            if validation_status:
                logger.info("Data validation completed successfully")
            else:
                logger.warning("Data validation completed with issues - check status file for details")
                
            return validation_status
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            raise e

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataValidationTrainingPipeline()
        obj.initiate_data_validation()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e