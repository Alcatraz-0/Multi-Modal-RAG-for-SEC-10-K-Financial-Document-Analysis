"""
Configuration file for Multi-Modal RAG system
"""
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
PARSED_DATA_DIR = DATA_DIR / 'parsed'

# Index and model directories
INDICES_DIR = PROJECT_ROOT / 'indices'
MODEL_DIR = PROJECT_ROOT / 'models'

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, PARSED_DATA_DIR, INDICES_DIR, MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API Configuration
SEC_API_BASE_URL = "https://data.sec.gov"
SEC_EDGAR_ARCHIVES = "https://www.sec.gov/cgi-bin/browse-edgar"

# Model configurations
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# Retrieval parameters
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_TOP_K_SECTIONS = 5
DEFAULT_TOP_K_CONTENT = 10

# QA parameters
DEFAULT_MAX_ANSWER_LENGTH = 512
DEFAULT_TEMPERATURE = 0.1

# Evaluation parameters
RANDOM_SEED = 42
