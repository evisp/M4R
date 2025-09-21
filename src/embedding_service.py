"""
Embedding service for generating vector embeddings from text
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import config
import json
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    print("✅ sentence-transformers imported successfully")
except ImportError as e:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print(f"❌ sentence-transformers import failed: {e}")
    print("⚠️  Try: pip install sentence-transformers")


class EmbeddingService:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.model = None
        self.embedding_dimension = config.EMBEDDING_DIMENSION
        self.embeddings_cache = {}
        
    def load_model(self) -> bool:
        """Load the embedding model"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("❌ Cannot load model: sentence-transformers not installed")
            return False
        
        try:
            print(f"Loading embedding model: {self.model_name}")
            print("(This may take a few minutes on first run - downloading model...)")
            
            self.model = SentenceTransformer(self.model_name)
            actual_dim = self.model.get_sentence_embedding_dimension()
            
            print(f"✅ Model loaded successfully!")
            print(f"   Model: {self.model_name}")
            print(f"   Embedding dimension: {actual_dim}")
            
            # Update config if dimension differs
            if actual_dim != self.embedding_dimension:
                print(f"   (Updated dimension from {self.embedding_dimension} to {actual_dim})")
                self.embedding_dimension = actual_dim
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for a single text"""
        if self.model is None:
            print("❌ Model not loaded. Call load_model() first.")
            return None
        
        try:
            # Generate embedding
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)  # Ensure float32 for efficiency
        except Exception as e:
            print(f"❌ Error generating embedding for text: {e}")
            return None
    
    def generate_embeddings_batch(self, texts: List[str], show_progress: bool = True) -> List[np.ndarray]:
        """Generate embeddings for multiple texts (more efficient)"""
        if self.model is None:
            print("❌ Model not loaded. Call load_model() first.")
            return []
        
        try:
            print(f"Generating embeddings for {len(texts)} texts...")
            
            # Batch encoding is more efficient than one-by-one
            embeddings = self.model.encode(
                texts, 
                convert_to_numpy=True,
                show_progress_bar=show_progress,
                batch_size=32  # Process in batches of 32
            )
            
            # Convert to list of numpy arrays with consistent dtype
            return [emb.astype(np.float32) for emb in embeddings]
            
        except Exception as e:
            print(f"❌ Error generating batch embeddings: {e}")
            return []
    
    def process_text_representations(self, text_data: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """Process text representations and add embeddings"""
        
        if self.model is None:
            if not self.load_model():
                return {}
        
        results = {}
        
        for entity_type, items in text_data.items():
            print(f"\nProcessing {entity_type}...")
            
            if not items:
                results[entity_type] = []
                continue
            
            # Extract texts for batch processing
            texts = [item['text_representation'] for item in items]
            
            # Generate embeddings in batch
            embeddings = self.generate_embeddings_batch(texts)
            
            if len(embeddings) != len(items):
                print(f"❌ Embedding count mismatch for {entity_type}")
                results[entity_type] = []
                continue
            
            # Combine original data with embeddings
            processed_items = []
            for i, item in enumerate(items):
                processed_item = item.copy()
                processed_item['embedding'] = embeddings[i]
                processed_item['embedding_dimension'] = len(embeddings[i])
                processed_items.append(processed_item)
            
            results[entity_type] = processed_items
            print(f"✅ Generated {len(embeddings)} embeddings for {entity_type}")
        
        return results
    
    def save_embeddings(self, embeddings_data: Dict[str, List[Dict]], filename: str = "embeddings_cache.json"):
        """Save embeddings to disk (for caching)"""
        save_path = config.PROCESSED_DATA_DIR / filename
        
        try:
            # Prepare data for JSON serialization (convert numpy arrays to lists)
            serializable_data = {}
            
            for entity_type, items in embeddings_data.items():
                serializable_items = []
                for item in items:
                    serializable_item = item.copy()
                    if 'embedding' in serializable_item:
                        # Convert numpy array to list for JSON
                        serializable_item['embedding'] = serializable_item['embedding'].tolist()
                    serializable_items.append(serializable_item)
                serializable_data[entity_type] = serializable_items
            
            # Add metadata
            metadata = {
                'model_name': self.model_name,
                'embedding_dimension': self.embedding_dimension,
                'total_embeddings': sum(len(items) for items in embeddings_data.values())
            }
            
            save_data = {
                'metadata': metadata,
                'embeddings': serializable_data
            }
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2)
            
            print(f"💾 Saved embeddings to: {save_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving embeddings: {e}")
            return False
    
    def load_embeddings(self, filename: str = "embeddings_cache.json") -> Optional[Dict[str, List[Dict]]]:
        """Load embeddings from disk"""
        load_path = config.PROCESSED_DATA_DIR / filename
        
        if not load_path.exists():
            print(f"📂 No cached embeddings found at {load_path}")
            return None
        
        try:
            with open(load_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert embedding lists back to numpy arrays
            embeddings_data = {}
            for entity_type, items in data['embeddings'].items():
                converted_items = []
                for item in items:
                    converted_item = item.copy()
                    if 'embedding' in converted_item:
                        converted_item['embedding'] = np.array(converted_item['embedding'], dtype=np.float32)
                    converted_items.append(converted_item)
                embeddings_data[entity_type] = converted_items
            
            metadata = data.get('metadata', {})
            print(f"📂 Loaded cached embeddings:")
            print(f"   Model: {metadata.get('model_name', 'Unknown')}")
            print(f"   Dimension: {metadata.get('embedding_dimension', 'Unknown')}")
            print(f"   Total embeddings: {metadata.get('total_embeddings', 'Unknown')}")
            
            return embeddings_data
            
        except Exception as e:
            print(f"❌ Error loading embeddings: {e}")
            return None
    
    def get_embedding_statistics(self, embeddings_data: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Get statistics about generated embeddings"""
        
        stats = {}
        
        for entity_type, items in embeddings_data.items():
            if not items:
                stats[entity_type] = {'count': 0}
                continue
            
            # Extract embeddings
            embeddings = [item['embedding'] for item in items if 'embedding' in item]
            
            if not embeddings:
                stats[entity_type] = {'count': 0}
                continue
            
            # Calculate statistics
            embeddings_array = np.array(embeddings)
            
            stats[entity_type] = {
                'count': len(embeddings),
                'dimension': embeddings_array.shape[1],
                'mean_norm': float(np.mean(np.linalg.norm(embeddings_array, axis=1))),
                'std_norm': float(np.std(np.linalg.norm(embeddings_array, axis=1))),
                'sample_values': embeddings[0][:5].tolist() if embeddings else []  # First 5 values of first embedding
            }
        
        return stats
    
    def find_similar_texts(self, query_embedding: np.ndarray, 
                          embeddings_data: Dict[str, List[Dict]], 
                          entity_type: str, 
                          top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Find most similar texts to a query embedding (simple cosine similarity)"""
        
        if entity_type not in embeddings_data:
            return []
        
        items = embeddings_data[entity_type]
        if not items:
            return []
        
        similarities = []
        
        for item in items:
            if 'embedding' not in item:
                continue
            
            # Calculate cosine similarity
            item_embedding = item['embedding']
            
            # Normalize vectors
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            item_norm = item_embedding / np.linalg.norm(item_embedding)
            
            # Cosine similarity
            similarity = np.dot(query_norm, item_norm)
            similarities.append((item, float(similarity)))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
