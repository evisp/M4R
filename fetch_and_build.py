"""
Complete AI Matching Pipeline
Fetches data from API, builds embeddings, and generates recommendations

Usage:
    python3 fetch_and_build.py              # Full pipeline
    python3 fetch_and_build.py --fetch-only # Just fetch data
    python3 fetch_and_build.py --build-only # Just build (assumes data exists)
"""
import os
import json
import requests
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Import your existing modules
from src.data_processor import DataProcessor
from src.text_generator import TextGenerator
from src.embedding_service import EmbeddingService
from src.vector_store import VectorStore
from src.matching_engine import MatchingEngine
import config


def fetch_data_from_api():
    """Fetch data from Match4Research API"""
    print("=" * 60)
    print("STEP 1: FETCHING DATA FROM API")
    print("=" * 60)
    
    # Load environment variables
    load_dotenv()
    token = os.getenv('M4R_API_TOKEN')
    
    if not token:
        print("ERROR: M4R_API_TOKEN not found in .env file")
        print("Please add: M4R_API_TOKEN=your_token_here")
        return False
    
    # API endpoint
    api_url = "https://match4research.com/api/network/machine-learning"
    
    print(f"Fetching from: {api_url}")
    print(f"Using token: {token[:20]}...")
    
    try:
        headers = {
            'Authorization': f'Bearer {token}',
            'Accept': 'application/json'
        }
        
        response = requests.get(api_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Save raw response
        output_file = Path("data/api_response.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"SUCCESS: Saved API response to {output_file}")
        print(f"Keys found: {list(data.keys())}")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to fetch data from API: {e}")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def split_api_data():
    """Split API response into individuals, organizations, project_calls"""
    print("\n" + "=" * 60)
    print("STEP 2: SPLITTING DATA INTO FILES")
    print("=" * 60)
    
    api_file = Path("data/api_response.json")
    
    if not api_file.exists():
        print(f"ERROR: {api_file} not found")
        return False
    
    # Load API response
    with open(api_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded API response with keys: {list(data.keys())}")
    
    # Create output directory
    output_dir = Path("data/sample")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved = []
    
    # Save individuals
    individuals = data.get('individuals', [])
    if individuals:
        with open(output_dir / 'individuals.json', 'w', encoding='utf-8') as f:
            json.dump({"individuals": individuals}, f, indent=2, ensure_ascii=False)
        print(f"  Saved {len(individuals)} individuals")
        saved.append('individuals')
    
    # Save organizations
    organizations = data.get('organizations', [])
    if organizations:
        with open(output_dir / 'organizations.json', 'w', encoding='utf-8') as f:
            json.dump({"organizations": organizations}, f, indent=2, ensure_ascii=False)
        print(f"  Saved {len(organizations)} organizations")
        saved.append('organizations')
    
    # Save project calls (from openCalls)
    open_calls = data.get('openCalls', [])
    if open_calls:
        with open(output_dir / 'project_calls.json', 'w', encoding='utf-8') as f:
            json.dump({"project_calls": open_calls}, f, indent=2, ensure_ascii=False)
        print(f"  Saved {len(open_calls)} project calls")
        saved.append('project_calls')
    
    if len(saved) == 3:
        print(f"\nSUCCESS: All 3 files created in {output_dir}/")
        return True
    else:
        print(f"\nWARNING: Only {len(saved)}/3 files created")
        return False


def build_and_generate_recommendations(top_k=5):
    """Build embeddings and generate batch recommendations"""
    print("\n" + "=" * 60)
    print("STEP 3: BUILDING AI SYSTEM")
    print("=" * 60)
    
    # Initialize components
    print("Initializing AI components...")
    processor = DataProcessor()
    text_gen = TextGenerator()
    embedding_service = EmbeddingService()
    vector_store = VectorStore()
    matching_engine = MatchingEngine(vector_store)
    
    # Load or build embeddings
    print("\nLoading/building embeddings...")
    if vector_store.load_index():
        print("  Loaded from cache")
        embeddings_data = embedding_service.load_embeddings()
    else:
        print("  Building from scratch...")
        data = processor.load_all_data()
        text_representations = text_gen.generate_text_for_dataset(data)
        embeddings_data = embedding_service.process_text_representations(text_representations)
        if embeddings_data:
            embedding_service.save_embeddings(embeddings_data)
            vector_store.load_or_build_index(embeddings_data)
    
    if not embeddings_data:
        print("ERROR: Failed to load embeddings")
        return False
    
    matching_engine.set_embeddings_data(embeddings_data)
    
    # Generate recommendations
    print("\n" + "=" * 60)
    print("STEP 4: GENERATING RECOMMENDATIONS")
    print("=" * 60)
    
    individuals = embeddings_data.get('individuals', [])
    organizations = embeddings_data.get('organizations', [])
    projects = embeddings_data.get('project_calls', [])
    
    print(f"Entities: {len(individuals)} individuals, {len(organizations)} organizations, {len(projects)} projects")
    
    results = {
        "metadata": {
            "generated_at": None,  # Will be set by batch generator
            "total_individuals": len(individuals),
            "total_organizations": len(organizations),
            "total_projects": len(projects),
            "top_k_per_entity": top_k
        },
        "recommendations": {
            "individuals": [],
            "organizations": []
        }
    }
    
    # Generate for individuals
    print(f"\nGenerating recommendations for {len(individuals)} individuals...")
    for i, individual in enumerate(individuals, 1):
        user_id = individual['id']
        user_name = individual['original_data'].get('fullName', 'Unknown')
        
        if i % 5 == 0:
            print(f"  [{i}/{len(individuals)}] {user_name}")
        
        recommendations = matching_engine.find_recommendations(
            user_id=user_id,
            entity_type='individuals',
            top_k=top_k
        )
        
        results['recommendations']['individuals'].append({
            'individual_id': user_id,
            'individual_name': user_name,
            'recommendations': recommendations,
            'match_count': len(recommendations)
        })
    
    # Generate for organizations
    print(f"\nGenerating recommendations for {len(organizations)} organizations...")
    for i, org in enumerate(organizations, 1):
        org_id = org['id']
        org_name = org['original_data'].get('name', 'Unknown')
        
        if i % 5 == 0:
            print(f"  [{i}/{len(organizations)}] {org_name}")
        
        recommendations = matching_engine.find_recommendations(
            user_id=org_id,
            entity_type='organizations',
            top_k=top_k
        )
        
        results['recommendations']['organizations'].append({
            'organization_id': org_id,
            'organization_name': org_name,
            'recommendations': recommendations,
            'match_count': len(recommendations)
        })
    
    # Save results
    output_dir = Path("data/processed/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "recommendations.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n" + "=" * 60)
    print("SUCCESS: PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Recommendations saved to: {output_file}")
    print(f"Total individuals processed: {len(individuals)}")
    print(f"Total organizations processed: {len(organizations)}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Match4Research AI Matching Pipeline')
    parser.add_argument('--fetch-only', action='store_true', help='Only fetch data from API')
    parser.add_argument('--build-only', action='store_true', help='Only build embeddings (skip fetch)')
    parser.add_argument('--top-k', type=int, default=5, help='Number of recommendations per entity')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MATCH4RESEARCH AI MATCHING PIPELINE")
    print("=" * 60)
    
    if args.build_only:
        # Skip fetching, just build
        print("Mode: Build only (using existing data)")
        success = build_and_generate_recommendations(args.top_k)
    elif args.fetch_only:
        # Just fetch and split
        print("Mode: Fetch only")
        success = fetch_data_from_api()
        if success:
            success = split_api_data()
    else:
        # Full pipeline
        print("Mode: Full pipeline (fetch + build + generate)")
        
        # Step 1 & 2: Fetch and split
        success = fetch_data_from_api()
        if not success:
            print("\nERROR: Failed to fetch data. Aborting.")
            return
        
        success = split_api_data()
        if not success:
            print("\nERROR: Failed to split data. Aborting.")
            return
        
        # Step 3 & 4: Build and generate
        success = build_and_generate_recommendations(args.top_k)
    
    if success:
        print("\n" + "=" * 60)
        print("DONE! Recommendations ready for API consumption")
        print("=" * 60)
    else:
        print("\nERROR: Pipeline failed")


if __name__ == "__main__":
    main()
