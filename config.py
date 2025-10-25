import os
from pathlib import Path
from dotenv import load_dotenv  # ADD this

# Load .env file if it exists
load_dotenv()  # ADD this

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
EMBEDDINGS_DIR = MODELS_DIR / "embeddings"

# API Configuration
API_BASE_URL = "https://match4research.com/api/network/machine-learning"
API_TOKEN = os.getenv('M4R_API_TOKEN', '')  # Now reads from .env file

# Model configuration
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
EMBEDDING_DIMENSION = int(os.getenv('EMBEDDING_DIMENSION', '384'))

# Matching parameters
SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.7'))
MAX_RECOMMENDATIONS = int(os.getenv('MAX_RECOMMENDATIONS', '20'))
TOP_K_SEARCH = int(os.getenv('TOP_K_SEARCH', '50'))

# Vector store settings
VECTOR_STORE_TYPE = "faiss"
INDEX_TYPE = "IndexFlatIP"

# Optional LLM re-ranking
USE_LLM_RERANKING = os.getenv('USE_LLM_RERANKING', 'False').lower() == 'true'
LLM_MODEL = os.getenv('LLM_MODEL', 'gemini-flash')
LLM_TOP_K = int(os.getenv('LLM_TOP_K', '10'))
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')

# Create directories
for directory in [SAMPLE_DATA_DIR, PROCESSED_DATA_DIR, EMBEDDINGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
