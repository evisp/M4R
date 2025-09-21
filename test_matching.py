"""
Fast testing script - loads cached FAISS index directly
"""
from src.embedding_service import EmbeddingService
from src.vector_store import VectorStore
from src.matching_engine import MatchingEngine
import config

def main():
    print("Match4Research - Fast Matching Test")
    print("=" * 45)
    
    # Initialize components
    embedding_service = EmbeddingService()
    vector_store = VectorStore()
    matching_engine = MatchingEngine(vector_store)
    
    # Try to load complete index first (FASTEST)
    print("Loading cached FAISS index...")
    if vector_store.load_index():
        print("✅ Loaded complete index from cache!")
        
        # Still need embeddings data for matching engine
        embeddings_data = embedding_service.load_embeddings()
        if not embeddings_data:
            print("❌ Need embeddings data for matching engine")
            return
        
        matching_engine.set_embeddings_data(embeddings_data)
        
    else:
        # Fallback: build from embeddings
        print("No cached index, building from embeddings...")
        embeddings_data = embedding_service.load_embeddings()
        
        if not embeddings_data:
            print("❌ No cached data found!")
            print("Run 'python3 main.py' first to build the database.")
            return
        
        success = vector_store.load_or_build_index(embeddings_data)
        if not success:
            return
        
        matching_engine.set_embeddings_data(embeddings_data)
    
    # Quick stats
    stats = vector_store.get_stats()
    print(f"✅ Ready! {stats['total_vectors']} vectors loaded")
    print(f"   {stats['entity_type_counts']}")
    
    # Interactive testing
    print("\n" + "=" * 50)
    print("INTERACTIVE TESTING")
    print("=" * 50)
    
    # List available users
    individuals = embeddings_data.get('individuals', [])
    organizations = embeddings_data.get('organizations', [])
    
    print(f"\nAvailable Individuals ({len(individuals)}):")
    for i, ind in enumerate(individuals):
        name = ind['original_data']['fullName']
        title = ind['original_data']['title']
        location = ind['original_data'].get('location', 'Unknown')
        print(f"  {i+1}. {name} - {title} ({location})")
    
    print(f"\nAvailable Organizations ({len(organizations)}):")
    for i, org in enumerate(organizations):
        name = org['original_data']['name']
        org_type = org['original_data']['type']
        location = org['original_data'].get('location', 'Unknown')
        org_number = len(individuals) + i + 1
        print(f"  {org_number}. {name} - {org_type} ({location})")
    
    # Interactive testing loop
    while True:
        print("\n" + "-" * 50)
        total_entities = len(individuals) + len(organizations)
        choice = input(f"Enter number to test (1-{total_entities}, 'q' to quit): ").strip()
        
        if choice.lower() == 'q':
            print("👋 Goodbye!")
            break
        
        try:
            choice_num = int(choice)
            
            if 1 <= choice_num <= len(individuals):
                # Test individual
                user = individuals[choice_num - 1]
                user_id = user['id']
                user_name = user['original_data']['fullName']
                user_title = user['original_data']['title']
                
                print(f"\n🔍 Testing: {user_name} ({user_title})")
                recommendations = matching_engine.find_recommendations(
                    user_id=user_id,
                    entity_type='individuals',
                    top_k=5
                )
                
                display_recommendations(recommendations)
                
            elif len(individuals) + 1 <= choice_num <= len(individuals) + len(organizations):
                # Test organization
                org_index = choice_num - len(individuals) - 1
                org = organizations[org_index]
                org_id = org['id']
                org_name = org['original_data']['name']
                org_type = org['original_data']['type']
                
                print(f"\n🔍 Testing: {org_name} ({org_type})")
                recommendations = matching_engine.find_recommendations(
                    user_id=org_id,
                    entity_type='organizations',
                    top_k=5
                )
                
                display_recommendations(recommendations)
            
            else:
                print("❌ Invalid choice. Please try again.")
                
        except ValueError:
            print("❌ Please enter a valid number or 'q' to quit.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

def display_recommendations(recommendations):
    """Display recommendations in a nice format"""
    if not recommendations:
        print("   📭 No recommendations found")
        return
    
    print(f"   📋 Found {len(recommendations)} recommendations:")
    print()
    
    for i, rec in enumerate(recommendations, 1):
        # Header
        print(f"   {i}. 🎯 {rec['project_title']}")
        print(f"      📊 Score: {rec['match_score']:.3f} ({rec['confidence']} confidence)")
        
        # Project details
        print(f"      🏢 Organization: {rec['organization_name']}")
        print(f"      📝 Type: {rec['project_type']}")
        print(f"      💰 Budget: {rec['project_summary']['budget']}")
        print(f"      📍 Location: {rec['project_summary']['location']}")
        print(f"      ⏱️  Duration: {rec['project_summary']['duration']}")
        print(f"      🚀 Delivery: {rec['project_summary']['delivery']}")
        
        # Match reasons
        print(f"      ✨ Why it matches: {', '.join(rec['match_reasons'][:3])}")
        
        # Summary preview
        summary = rec['project_summary']['summary']
        if summary:
            preview = summary[:100] + "..." if len(summary) > 100 else summary
            print(f"      📄 Summary: {preview}")
        
        print()  

if __name__ == "__main__":
    main()
