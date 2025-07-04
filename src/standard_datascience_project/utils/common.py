import os
import yaml
import re
from src.standard_datascience_project import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
from box.exceptions import BoxValueError


def substitute_env_vars(content):
    """Substitute environment variables in string content
    
    Args:
        content: String content that may contain ${VAR:-default} patterns
        
    Returns:
        String with environment variables substituted
    """
    if isinstance(content, str):
        # Pattern to match ${VAR:-default} or ${VAR}
        pattern = r'\$\{([^:}]+)(?::-([^}]+))?\}'
        
        def replace(match):
            var_name = match.group(1)
            default_value = match.group(2)
            
            # Get environment variable value
            env_value = os.getenv(var_name)
            
            # Return environment value if exists, otherwise default value
            if env_value is not None:
                return env_value
            elif default_value is not None:
                return default_value
            else:
                # If no default and env var doesn't exist, return original
                return match.group(0)
        
        return re.sub(pattern, replace, content)
    return content


def substitute_env_vars_recursive(obj):
    """Recursively substitute environment variables in nested objects
    
    Args:
        obj: Object that may contain strings with env var patterns
        
    Returns:
        Object with environment variables substituted
    """
    if isinstance(obj, dict):
        return {key: substitute_env_vars_recursive(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [substitute_env_vars_recursive(item) for item in obj]
    elif isinstance(obj, str):
        return substitute_env_vars(obj)
    else:
        return obj


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            # Substitute environment variables
            content = substitute_env_vars_recursive(content)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
        


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")

@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)

@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")

@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data