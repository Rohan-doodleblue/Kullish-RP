import os
from src.standard_datascience_project import logger 
from src.standard_datascience_project.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline


STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   
   # Check if force download is requested (optional)
   force_download = os.getenv('FORCE_DOWNLOAD', 'false').lower() == 'true'
   if force_download:
       logger.info("Force download enabled - will re-download existing files")
   
   data_ingestion = DataIngestionTrainingPipeline(force_download=force_download)
   data_ingestion.initiate_data_ingestion()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e