"""
Configuration file for the Automated Assessment API
Copy this file and rename it to .env to set your environment variables
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your-openai-api-key-here')
    
    # Token Limits
    MAX_TOKENS_PER_DAY = int(os.getenv('MAX_TOKENS_PER_DAY', '25000'))
    MAX_TOKENS_PER_REQUEST = int(os.getenv('MAX_TOKENS_PER_REQUEST', '4000'))
    
    # Model Configuration
    MODEL = os.getenv('MODEL', 'gpt-3.5-turbo')
    
    # File Configuration
    TOKEN_USAGE_FILE = os.getenv('TOKEN_USAGE_FILE', 'token_usage.json')
    
    # Server Configuration
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', '5001'))
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'

# Instructions for setup:
"""
To set up your environment:

1. Create a .env file in the root directory
2. Add your OpenAI API key:
   OPENAI_API_KEY=your-actual-api-key-here

3. Optional: Override default settings:
   MAX_TOKENS_PER_DAY=25000
   MODEL=gpt-3.5-turbo
   DEBUG=True

4. Make sure to add .env to your .gitignore file to keep your API key secure
""" 