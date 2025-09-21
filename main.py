"""
Main entry point for testing the matching pipeline
"""
from src.data_processor import DataProcessor
from src.text_generator import TextGenerator
from src.embedding_service import EmbeddingService
from src.vector_store import VectorStore
from src.matching_engine import MatchingEngine
import config

def main():
    print("Match4Research AI Matching System")
    print("=" * 40)
    
    # Initialize all components
    processor = DataProcessor()
    text_gen = TextGenerator()
    embedding_service = EmbeddingService()
    vector_store = VectorStore()
    matching_engine = MatchingEngine(vector_store)
    
    # Step 4: Smart Loading - Try cache first, build if needed
    print("\nStep 4: Loading/Building Vector Store & Matching Engine")
    print("=" * 60)
    
    # Try to load complete cached system first (FASTEST PATH)
    if vector_store.load_index():
        print("✅ Loaded complete FAISS index from cache!")
        
        # Still need embeddings data for matching engine
        embeddings_data = embedding_service.load_embeddings()
        if embeddings_data:
            print("✅ Loaded embeddings data from cache!")
            matching_engine.set_embeddings_data(embeddings_data)
        else:
            print("❌ FAISS index found but no embeddings data - rebuilding...")
            rebuild_everything()
            return
    
    else:
        print("📂 No cached index found - building from scratch...")
        
        # Steps 1-3: Full pipeline
        print("\nStep 1: Loading Data...")
        data = processor.load_all_data()
        
        print("\nStep 2: Generating Text Representations...")
        text_representations = text_gen.generate_text_for_dataset(data)
        
        print("\nStep 3: Loading/Generating Embeddings...")
        cached_embeddings = embedding_service.load_embeddings()
        
        if cached_embeddings:
            embeddings_data = cached_embeddings
            print("✅ Using cached embeddings")
        else:
            print("🔄 Generating new embeddings...")
            embeddings_data = embedding_service.process_text_representations(text_representations)
            if embeddings_data:
                embedding_service.save_embeddings(embeddings_data)
        
        # Build vector store with auto-save
        print("\n🔄 Building FAISS index...")
        success = vector_store.load_or_build_index(embeddings_data)
        if not success:
            print("❌ Failed to build vector store")
            return
        
        # Set up matching engine
        matching_engine.set_embeddings_data(embeddings_data)
    
    # Show system stats
    stats = vector_store.get_stats()
    print(f"\nSystem Ready! 🎉")
    print(f"  Total vectors: {stats['total_vectors']}")
    print(f"  Dimension: {stats['dimension']}")
    print(f"  Entity counts: {stats['entity_type_counts']}")
    
    # Get embeddings data if we don't have it yet
    if 'embeddings_data' not in locals():
        embeddings_data = embedding_service.load_embeddings()
    
    # Test recommendations for different users
    print("\n" + "=" * 60)
    print("TESTING SMART MATCHING")
    print("=" * 60)
    
    # Test individuals
    individuals = embeddings_data.get('individuals', [])
    if individuals:
        print(f"\n👤 Testing Individual Recommendations:")
        print("-" * 40)
        
        # Test first 2 individuals
        for i, individual in enumerate(individuals[:2]):
            user_id = individual['id']
            user_name = individual['original_data']['fullName']
            user_title = individual['original_data']['title']
            
            print(f"\n{i+1}. {user_name} ({user_title})")
            print("-" * 50)
            
            recommendations = matching_engine.find_recommendations(
                user_id=user_id,
                entity_type='individuals',
                top_k=3
            )
            
            if recommendations:
                for j, rec in enumerate(recommendations, 1):
                    print(f"   {j}. {rec['project_title'][:45]}...")
                    print(f"      Score: {rec['match_score']:.3f} ({rec['confidence']})")
                    print(f"      Type: {rec['project_type']} | Org: {rec['organization_name']}")
                    print(f"      Reasons: {', '.join(rec['match_reasons'])}")
                    print()
            else:
                print("   No recommendations found")
    
    # Test organizations
    organizations = embeddings_data.get('organizations', [])
    if organizations:
        print(f"\n🏢 Testing Organization Recommendations:")
        print("-" * 45)
        
        # Test first organization
        org = organizations[0]
        org_id = org['id']
        org_name = org['original_data']['name']
        org_type = org['original_data']['type']
        
        print(f"\n{org_name} ({org_type})")
        print("-" * 50)
        
        recommendations = matching_engine.find_recommendations(
            user_id=org_id,
            entity_type='organizations',
            top_k=3
        )
        
        if recommendations:
            for j, rec in enumerate(recommendations, 1):
                print(f"   {j}. {rec['project_title'][:45]}...")
                print(f"      Score: {rec['match_score']:.3f} ({rec['confidence']})")
                print(f"      Type: {rec['project_type']} | Org: {rec['organization_name']}")
                print(f"      Reasons: {', '.join(rec['match_reasons'])}")
                print()
        else:
            print("   No recommendations found")
    
    print(f"\n" + "=" * 60)
    print("🎉 Testing Complete!")
    print("💡 Next time you run this, it will load much faster from cache!")
    print("🔄 To rebuild from scratch, delete files in data/processed/")
    print("⚡ For fast testing only, use: python3 test_matching.py")

def rebuild_everything():
    """Rebuild everything from scratch"""
    print("🔄 Rebuilding entire system from scratch...")
    
    # Clear all caches
    embedding_service = EmbeddingService()
    vector_store = VectorStore()
    
    # Clear cached files
    vector_store.clear_cache()
    
    import os
    embeddings_cache = config.PROCESSED_DATA_DIR / "embeddings_cache.json"
    if embeddings_cache.exists():
        os.remove(embeddings_cache)
        print("🗑️ Cleared embeddings cache")
    
    print("✅ Caches cleared - restart to rebuild")

if __name__ == "__main__":
    main()
