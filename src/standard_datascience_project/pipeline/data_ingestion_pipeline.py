from src.standard_datascience_project.config.configuration import ConfigurationManager
from src.standard_datascience_project.components.data_ingestion import DataIngestion
from src.standard_datascience_project import logger


STAGE_NAME="Data Ingestion Stage"

class DataIngestionTrainingPipeline:
    def __init__(self, force_download=False):
        self.force_download = force_download

    def initiate_data_ingestion(self):
        config=ConfigurationManager()
        data_ingestion_config=config.get_data_ingestion_config()
        data_ingestion=DataIngestion(config=data_ingestion_config, force_download=self.force_download)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()
        
        # Generate schema automatically after data extraction
        logger.info("Generating dynamic schema based on extracted data...")
        schema = data_ingestion.generate_schema()
        
        if schema:
            logger.info("✅ Dynamic schema generation completed successfully!")
        else:
            logger.warning("⚠️ Schema generation failed, using existing schema")


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline(force_download=True)  # Force download for testing
        obj.initiate_data_ingestion()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e