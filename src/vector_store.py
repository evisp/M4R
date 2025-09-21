"""
Vector store for managing FAISS index and similarity search
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import config
import pickle
from pathlib import Path

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️  faiss-cpu not installed. Run: pip install faiss-cpu")


class VectorStore:
    def __init__(self, embedding_dimension: int = None):
        self.embedding_dimension = embedding_dimension or config.EMBEDDING_DIMENSION
        self.index = None
        self.metadata = []  # Store metadata for each vector
        self.id_to_index = {}  # Map entity IDs to index positions
        self.is_built = False
        
    def create_index(self, index_type: str = "IndexFlatIP") -> bool:
        """Create FAISS index"""
        if not FAISS_AVAILABLE:
            print("❌ Cannot create index: faiss-cpu not installed")
            return False
        
        try:
            if index_type == "IndexFlatIP":
                # Inner Product (good for normalized vectors/cosine similarity)
                self.index = faiss.IndexFlatIP(self.embedding_dimension)
            elif index_type == "IndexFlatL2":
                # L2 distance
                self.index = faiss.IndexFlatL2(self.embedding_dimension)
            else:
                print(f"❌ Unsupported index type: {index_type}")
                return False
            
            print(f"✅ Created FAISS index: {index_type}")
            print(f"   Dimension: {self.embedding_dimension}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating index: {e}")
            return False
    
    def save_index(self, filepath: str = None) -> bool:
        """Save FAISS index and metadata to disk"""
        if not self.is_built:
            print("❌ No index to save")
            return False
        
        if filepath is None:
            filepath = config.PROCESSED_DATA_DIR / "faiss_index"
        
        try:
            # Ensure directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            index_file = f"{filepath}.index"
            faiss.write_index(self.index, index_file)
            
            # Save metadata
            metadata_file = f"{filepath}.metadata"
            with open(metadata_file, 'wb') as f:
                pickle.dump({
                    'metadata': self.metadata,
                    'id_to_index': self.id_to_index,
                    'embedding_dimension': self.embedding_dimension
                }, f)
            
            print(f"💾 Saved FAISS index to: {index_file}")
            print(f"💾 Saved metadata to: {metadata_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving index: {e}")
            return False
    
    def load_index(self, filepath: str = None) -> bool:
        """Load FAISS index and metadata from disk"""
        if filepath is None:
            filepath = config.PROCESSED_DATA_DIR / "faiss_index"
        
        index_file = f"{filepath}.index"
        metadata_file = f"{filepath}.metadata"
        
        if not Path(index_file).exists():
            print(f"📂 No saved index found at {index_file}")
            return False
        
        if not Path(metadata_file).exists():
            print(f"📂 No saved metadata found at {metadata_file}")
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(index_file)
            
            # Load metadata
            with open(metadata_file, 'rb') as f:
                saved_data = pickle.load(f)
            
            self.metadata = saved_data['metadata']
            self.id_to_index = saved_data['id_to_index']
            self.embedding_dimension = saved_data['embedding_dimension']
            self.is_built = True
            
            print(f"📂 Loaded FAISS index from: {index_file}")
            print(f"📂 Loaded metadata from: {metadata_file}")
            print(f"   Total vectors: {self.index.ntotal}")
            print(f"   Dimension: {self.embedding_dimension}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading index: {e}")
            return False
    
    def load_or_build_index(self, embeddings_data: Dict[str, List[Dict]] = None) -> bool:
        """Try to load existing index, or build from embeddings if not found"""
        
        # First try to load existing index
        if self.load_index():
            return True
        
        # If no saved index, build from embeddings
        if embeddings_data:
            print("Building new index from embeddings...")
            success = self.add_embeddings(embeddings_data)
            if success:
                # Auto-save after building
                self.save_index()
            return success
        else:
            print("❌ No saved index found and no embeddings provided to build one")
            return False
    
    def add_embeddings(self, embeddings_data: Dict[str, List[Dict]]) -> bool:
        """Add embeddings to the index"""
        if self.index is None:
            if not self.create_index():
                return False
        
        try:
            all_embeddings = []
            current_index = 0
            
            # Process all entity types
            for entity_type, items in embeddings_data.items():
                print(f"Adding {len(items)} {entity_type} to index...")
                
                for item in items:
                    if 'embedding' not in item:
                        continue
                    
                    embedding = item['embedding']
                    entity_id = item.get('id', f"{entity_type}_{current_index}")
                    
                    # Normalize embedding for cosine similarity
                    embedding_norm = embedding / np.linalg.norm(embedding)
                    all_embeddings.append(embedding_norm)
                    
                    # Store metadata
                    metadata = {
                        'entity_id': entity_id,
                        'entity_type': entity_type,
                        'index_position': current_index,
                        'original_data': item.get('original_data', {}),
                        'text_representation': item.get('text_representation', '')
                    }
                    self.metadata.append(metadata)
                    self.id_to_index[entity_id] = current_index
                    
                    current_index += 1
            
            if all_embeddings:
                # Add to FAISS index
                embeddings_array = np.array(all_embeddings, dtype=np.float32)
                self.index.add(embeddings_array)
                self.is_built = True
                
                print(f"✅ Added {len(all_embeddings)} embeddings to FAISS index")
                return True
            else:
                print("❌ No embeddings found to add to index")
                return False
                
        except Exception as e:
            print(f"❌ Error adding embeddings to index: {e}")
            return False
    
    def search(self, query_embedding: np.ndarray, k: int = 10, entity_types: List[str] = None) -> List[Dict]:
        """Search for similar vectors"""
        if not self.is_built:
            print("❌ Index not built. Add embeddings first.")
            return []
        
        try:
            # Normalize query embedding
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            query_array = np.array([query_norm], dtype=np.float32)
            
            # Search FAISS index
            similarities, indices = self.index.search(query_array, k * 2)  # Get extra to allow filtering
            
            results = []
            for similarity, idx in zip(similarities[0], indices[0]):
                if idx < len(self.metadata):
                    metadata = self.metadata[idx]
                    
                    # Filter by entity type if specified
                    if entity_types and metadata['entity_type'] not in entity_types:
                        continue
                    
                    result = {
                        'entity_id': metadata['entity_id'],
                        'entity_type': metadata['entity_type'],
                        'similarity_score': float(similarity),
                        'original_data': metadata['original_data'],
                        'text_representation': metadata['text_representation']
                    }
                    results.append(result)
                    
                    if len(results) >= k:
                        break
            
            return results
            
        except Exception as e:
            print(f"❌ Error searching index: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        if not self.is_built:
            return {'built': False, 'total_vectors': 0}
        
        entity_type_counts = {}
        for metadata in self.metadata:
            entity_type = metadata['entity_type']
            entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1
        
        return {
            'built': True,
            'total_vectors': self.index.ntotal,
            'dimension': self.embedding_dimension,
            'entity_type_counts': entity_type_counts
        }
    
    def clear_cache(self, filepath: str = None) -> bool:
        """Delete saved index files"""
        if filepath is None:
            filepath = config.PROCESSED_DATA_DIR / "faiss_index"
        
        index_file = Path(f"{filepath}.index")
        metadata_file = Path(f"{filepath}.metadata")
        
        deleted_files = []
        
        if index_file.exists():
            index_file.unlink()
            deleted_files.append(str(index_file))
        
        if metadata_file.exists():
            metadata_file.unlink()
            deleted_files.append(str(metadata_file))
        
        if deleted_files:
            print(f"🗑️ Deleted cached files: {deleted_files}")
            return True
        else:
            print("📂 No cached files found to delete")
            return False
