import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
EMBEDDINGS_DIR = MODELS_DIR / "embeddings"

# Model configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Start with this
# EMBEDDING_MODEL = "BAAI/bge-m3"  # Upgrade later
EMBEDDING_DIMENSION = 384  # 384 for MiniLM, 1024 for BGE-M3

# Matching parameters
SIMILARITY_THRESHOLD = 0.7
MAX_RECOMMENDATIONS = 20
TOP_K_SEARCH = 50

# Vector store settings
VECTOR_STORE_TYPE = "faiss"
INDEX_TYPE = "IndexFlatIP"  # Inner Product for cosine similarity

# Optional LLM re-ranking
USE_LLM_RERANKING = False
LLM_MODEL = "gemini-flash"
LLM_TOP_K = 10

# Create directories
for directory in [SAMPLE_DATA_DIR, PROCESSED_DATA_DIR, EMBEDDINGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
