import os
import urllib.request as request
from src.standard_datascience_project  import logger
import zipfile
from src.standard_datascience_project.entity.config_entity import (DataIngestionConfig)
from src.standard_datascience_project.components.schema_generator import SchemaGenerator
from pathlib import Path
import shutil
import re


class DataIngestion:
    def __init__(self,config:DataIngestionConfig, force_download=False):
        self.config=config
        self.force_download = force_download
    
    def convert_gdrive_url(self, url):
        """Convert Google Drive sharing URL to direct download URL"""
        # Pattern for Google Drive sharing URLs
        gdrive_patterns = [
            r'https://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)',
            r'https://drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)',
            r'https://docs\.google\.com/uc\?export=download&id=([a-zA-Z0-9_-]+)'
        ]
        
        for pattern in gdrive_patterns:
            match = re.search(pattern, url)
            if match:
                file_id = match.group(1)
                # Convert to direct download URL
                direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
                logger.info(f"Converted Google Drive URL to direct download: {direct_url}")
                return direct_url
        
        # If no pattern matches, return original URL
        return url
    
    def download_file(self):
        if not os.path.exists(self.config.local_data_file) or self.force_download:
            if self.force_download and os.path.exists(self.config.local_data_file):
                logger.info(f"Force download enabled, removing existing file: {self.config.local_data_file}")
                os.remove(self.config.local_data_file)
            
            # Convert Google Drive URL if needed
            download_url = self.convert_gdrive_url(self.config.source_URL)
            
            filename, headers = request.urlretrieve(
                url = download_url,
                filename = self.config.local_data_file
            )
            logger.info(f"{filename} download! with following info: \n{headers}")
        else:
            logger.info(f"File already exists")
            
        # Check if the file is corrupted and re-download if necessary
        if self.is_file_corrupted():
            logger.warning("Detected corrupted file, re-downloading...")
            os.remove(self.config.local_data_file)
            
            # Convert Google Drive URL if needed
            download_url = self.convert_gdrive_url(self.config.source_URL)
            
            filename, headers = request.urlretrieve(
                url = download_url,
                filename = self.config.local_data_file
            )
            logger.info(f"Re-downloaded: {filename}")
            
            # Check again after re-download
            if self.is_file_corrupted():
                raise Exception("File is still corrupted after re-download. Please check the URL.")
    
    def is_file_corrupted(self):
        """Check if the downloaded file is corrupted or invalid"""
        try:
            # Try to open as ZIP file
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                # If it opens successfully, it's a valid ZIP
                return False
        except zipfile.BadZipFile:
            # Check if it's an HTML file (error page)
            try:
                with open(self.config.local_data_file, 'r', encoding='utf-8') as f:
                    content = f.read(100)  # Read first 100 characters
                    if '<html' in content.lower() or '<!doctype' in content.lower():
                        logger.warning("Downloaded file appears to be an HTML error page")
                        return True
            except:
                pass
            
            # Check file size - if it's too small, it might be corrupted
            file_size = os.path.getsize(self.config.local_data_file)
            if file_size < 1000:  # Less than 1KB
                logger.warning(f"Downloaded file is very small ({file_size} bytes), might be corrupted")
                return True
                
            return False
        except Exception as e:
            logger.error(f"Error checking file: {e}")
            return True

    def extract_zip_file(self):
        """
        Extracts the zip file into the data directory
        Also handles non-zip files by copying them to the data directory
        """
        unzip_path = Path(self.config.unzip_dir)  # Ensure it's a Path object
        os.makedirs(unzip_path, exist_ok=True)
        
        # Check if the downloaded file is a zip file
        try:
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)
            logger.info(f"Successfully extracted ZIP file to {unzip_path}")
        except zipfile.BadZipFile:
            # If it's not a zip file, copy it directly to the data directory
            logger.info(f"File is not a ZIP file, copying as data file...")
            file_extension = Path(self.config.local_data_file).suffix.lower()
            
            # Check if the file is already in the target directory
            if Path(self.config.local_data_file).parent == unzip_path:
                logger.info(f"File is already in the target directory: {self.config.local_data_file}")
            else:
                # Copy the file to the data directory with appropriate name
                if file_extension in ['.csv', '.json', '.parquet', '.xlsx', '.xls']:
                    # It's a data file, copy it directly
                    dest_file = unzip_path / f"data{file_extension}"
                    shutil.copy2(self.config.local_data_file, dest_file)
                    logger.info(f"Copied data file to {dest_file}")
                else:
                    # Unknown file type, copy with original name
                    dest_file = unzip_path / Path(self.config.local_data_file).name
                    shutil.copy2(self.config.local_data_file, dest_file)
                    logger.info(f"Copied file to {dest_file}")
    
    def generate_schema(self):
        """
        Automatically generate and update schema based on extracted data
        """
        try:
            logger.info("Starting automatic schema generation...")
            logger.info(f"Data directory: {self.config.unzip_dir} (type: {type(self.config.unzip_dir)})")
            
            # Initialize schema generator
            schema_generator = SchemaGenerator(self.config.unzip_dir)
            
            # Generate and update schema
            schema = schema_generator.generate_and_update_schema()
            
            logger.info(f"Schema generated successfully with {len(schema['COLUMNS'])} columns")
            logger.info(f"Target column: {schema['TARGET_COLUMN']['name']}")
            logger.info(f"Data info: {schema['DATA_INFO']}")
            
            return schema
            
        except Exception as e:
            logger.error(f"Error generating schema: {e}")
            logger.info("Continuing without schema generation...")
            return None