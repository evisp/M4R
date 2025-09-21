import time
print("=== Performance Debug ===")

start = time.time()
print("1. Importing libraries...")
from src.embedding_service import EmbeddingService
from src.vector_store import VectorStore  
from src.matching_engine import MatchingEngine
print(f"   Import time: {time.time() - start:.2f}s")

start = time.time()
print("2. Initializing components...")
embedding_service = EmbeddingService()
vector_store = VectorStore()
matching_engine = MatchingEngine(vector_store)
print(f"   Init time: {time.time() - start:.2f}s")

start = time.time()
print("3. Loading FAISS index...")
loaded = vector_store.load_index()
print(f"   FAISS load time: {time.time() - start:.2f}s")

if loaded:
    start = time.time()
    print("4. Loading embeddings...")
    embeddings_data = embedding_service.load_embeddings()
    print(f"   Embeddings load time: {time.time() - start:.2f}s")
    
    start = time.time()  
    print("5. Setting up matching engine...")
    matching_engine.set_embeddings_data(embeddings_data)
    print(f"   Setup time: {time.time() - start:.2f}s")

print("=== Debug Complete ===")
