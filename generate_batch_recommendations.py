"""
Batch Recommendation Generator
Generates complete recommendations for all individuals and organizations
Output: JSON file ready for API consumption
"""
from datetime import datetime
import json
from pathlib import Path
from src.data_processor import DataProcessor
from src.text_generator import TextGenerator
from src.embedding_service import EmbeddingService
from src.vector_store import VectorStore
from src.matching_engine import MatchingEngine
import config


def generate_batch_recommendations(top_k=5, output_file=None):
    """
    Generate recommendations for all individuals and organizations
    
    Args:
        top_k: Number of recommendations per entity
        output_file: Path to save JSON output (default: data/output/recommendations.json)
    
    Returns:
        Dictionary with complete recommendation data
    """
    print("=" * 60)
    print("BATCH RECOMMENDATION GENERATOR")
    print("=" * 60)
    
    # Initialize components
    print("\n1. Initializing AI matching system...")
    processor = DataProcessor()
    text_gen = TextGenerator()
    embedding_service = EmbeddingService()
    vector_store = VectorStore()
    matching_engine = MatchingEngine(vector_store)
    
    # Load or build system
    print("\n2. Loading/Building vector store...")
    if vector_store.load_index():
        print("   ✅ Loaded FAISS index from cache")
        embeddings_data = embedding_service.load_embeddings()
        if not embeddings_data:
            print("   ❌ Missing embeddings data")
            return None
    else:
        print("   📂 Building from scratch...")
        data = processor.load_all_data()
        text_representations = text_gen.generate_text_for_dataset(data)
        embeddings_data = embedding_service.process_text_representations(text_representations)
        if embeddings_data:
            embedding_service.save_embeddings(embeddings_data)
        vector_store.load_or_build_index(embeddings_data)
    
    matching_engine.set_embeddings_data(embeddings_data)
    
    # Get entity lists
    individuals = embeddings_data.get('individuals', [])
    organizations = embeddings_data.get('organizations', [])
    projects = embeddings_data.get('project_calls', [])
    
    print(f"\n3. Processing entities:")
    print(f"   👤 Individuals: {len(individuals)}")
    print(f"   🏢 Organizations: {len(organizations)}")
    print(f"   📋 Projects: {len(projects)}")
    
    # Initialize results structure
    results = {
        "metadata": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "model_version": "v1.0",
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_dimension": 384,
            "total_individuals": len(individuals),
            "total_organizations": len(organizations),
            "total_projects": len(projects),
            "top_k_per_entity": top_k
        },
        "recommendations": {
            "individuals": [],
            "organizations": []
        },
        "statistics": {
            "individuals_with_matches": 0,
            "organizations_with_matches": 0,
            "total_matches_generated": 0,
            "avg_match_score": 0.0,
            "high_confidence_matches": 0,
            "medium_confidence_matches": 0,
            "low_confidence_matches": 0
        }
    }
    
    all_scores = []
    
    # Process individuals
    print("\n4. Generating recommendations for individuals...")
    for i, individual in enumerate(individuals, 1):
        user_id = individual['id']
        # FIXED: camelCase with fallback
        user_name = individual['original_data'].get('fullName', individual['original_data'].get('full_name', 'Unknown'))
        user_type = individual['original_data'].get('type', 'Unknown')
        
        print(f"   [{i}/{len(individuals)}] Processing: {user_name}")
        
        # Get recommendations
        recommendations = matching_engine.find_recommendations(
            user_id=user_id,
            entity_type='individuals',
            top_k=top_k
        )
        
        # Format recommendations for API
        formatted_recs = []
        for rank, rec in enumerate(recommendations, 1):
            formatted_rec = {
                "project_id": rec['project_id'],
                "project_title": rec['project_title'],
                "project_type": rec['project_type'],
                "organization_name": rec['organization_name'],
                "match_score": round(rec['match_score'], 3),
                "confidence": rec['confidence'],
                "rank": rank,
                "match_reasons": rec['match_reasons'],
                "score_breakdown": {
                    "semantic_similarity": round(rec['score_breakdown']['semantic_similarity'], 3),
                    "applicant_type_match": round(rec['score_breakdown']['applicant_type_match'], 3),
                    "project_type_alignment": round(rec['score_breakdown']['project_type_alignment'], 3),
                    "delivery_compatibility": round(rec['score_breakdown']['delivery_compatibility'], 3),
                    "duration_fit": round(rec['score_breakdown']['duration_fit'], 3)
                },
                "project_details": {
                    "summary": rec['project_summary']['summary'],
                    "duration": rec['project_summary']['duration'],
                    "location": rec['project_summary']['location'],
                    "budget": rec['project_summary']['budget'],
                    "delivery": rec['project_summary']['delivery'],
                    "deadline": rec['project_summary']['deadline']
                }
            }
            formatted_recs.append(formatted_rec)
            all_scores.append(rec['match_score'])
            
            # Update statistics
            if rec['confidence'] == 'high':
                results['statistics']['high_confidence_matches'] += 1
            elif rec['confidence'] == 'medium':
                results['statistics']['medium_confidence_matches'] += 1
            else:
                results['statistics']['low_confidence_matches'] += 1
        
        # Add to results
        individual_result = {
            "individual_id": user_id,
            "individual_name": user_name,
            "individual_type": user_type,
            "location": individual['original_data'].get('location', 'Unknown'),
            "recommendations": formatted_recs,
            "match_count": len(formatted_recs),
            "has_matches": len(formatted_recs) > 0
        }
        
        results['recommendations']['individuals'].append(individual_result)
        
        if len(formatted_recs) > 0:
            results['statistics']['individuals_with_matches'] += 1
            results['statistics']['total_matches_generated'] += len(formatted_recs)
    
    # Process organizations
    print(f"\n5. Generating recommendations for organizations...")
    for i, organization in enumerate(organizations, 1):
        org_id = organization['id']
        org_name = organization['original_data'].get('name', 'Unknown')
        org_type = organization['original_data'].get('type', 'Unknown')
        
        print(f"   [{i}/{len(organizations)}] Processing: {org_name}")
        
        # Get recommendations
        recommendations = matching_engine.find_recommendations(
            user_id=org_id,
            entity_type='organizations',
            top_k=top_k
        )
        
        # Format recommendations for API
        formatted_recs = []
        for rank, rec in enumerate(recommendations, 1):
            formatted_rec = {
                "project_id": rec['project_id'],
                "project_title": rec['project_title'],
                "project_type": rec['project_type'],
                "organization_name": rec['organization_name'],
                "match_score": round(rec['match_score'], 3),
                "confidence": rec['confidence'],
                "rank": rank,
                "match_reasons": rec['match_reasons'],
                "score_breakdown": {
                    "semantic_similarity": round(rec['score_breakdown']['semantic_similarity'], 3),
                    "applicant_type_match": round(rec['score_breakdown']['applicant_type_match'], 3),
                    "project_type_alignment": round(rec['score_breakdown']['project_type_alignment'], 3),
                    "delivery_compatibility": round(rec['score_breakdown']['delivery_compatibility'], 3),
                    "duration_fit": round(rec['score_breakdown']['duration_fit'], 3)
                },
                "project_details": {
                    "summary": rec['project_summary']['summary'],
                    "duration": rec['project_summary']['duration'],
                    "location": rec['project_summary']['location'],
                    "budget": rec['project_summary']['budget'],
                    "delivery": rec['project_summary']['delivery'],
                    "deadline": rec['project_summary']['deadline']
                }
            }
            formatted_recs.append(formatted_rec)
            all_scores.append(rec['match_score'])
            
            # Update statistics
            if rec['confidence'] == 'high':
                results['statistics']['high_confidence_matches'] += 1
            elif rec['confidence'] == 'medium':
                results['statistics']['medium_confidence_matches'] += 1
            else:
                results['statistics']['low_confidence_matches'] += 1
        
        # Add to results
        org_result = {
            "organization_id": org_id,
            "organization_name": org_name,
            "organization_type": org_type,
            "location": organization['original_data'].get('location', 'Unknown'),
            "recommendations": formatted_recs,
            "match_count": len(formatted_recs),
            "has_matches": len(formatted_recs) > 0
        }
        
        results['recommendations']['organizations'].append(org_result)
        
        if len(formatted_recs) > 0:
            results['statistics']['organizations_with_matches'] += 1
            results['statistics']['total_matches_generated'] += len(formatted_recs)
    
    # Calculate average score
    if all_scores:
        results['statistics']['avg_match_score'] = round(sum(all_scores) / len(all_scores), 3)
    
    # Save to file
    if output_file is None:
        output_dir = config.PROCESSED_DATA_DIR / "output"
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "recommendations.json"
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n6. ✅ Recommendations saved to: {output_path}")
    print(f"\n{'=' * 60}")
    print("GENERATION SUMMARY")
    print("=" * 60)
    print(f"Total entities processed: {len(individuals) + len(organizations)}")
    print(f"Individuals with matches: {results['statistics']['individuals_with_matches']}/{len(individuals)}")
    print(f"Organizations with matches: {results['statistics']['organizations_with_matches']}/{len(organizations)}")
    print(f"Total matches generated: {results['statistics']['total_matches_generated']}")
    print(f"Average match score: {results['statistics']['avg_match_score']}")
    print(f"High confidence: {results['statistics']['high_confidence_matches']}")
    print(f"Medium confidence: {results['statistics']['medium_confidence_matches']}")
    print(f"Low confidence: {results['statistics']['low_confidence_matches']}")
    print("=" * 60)
    
    return results


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate batch recommendations')
    parser.add_argument('--top-k', type=int, default=5, help='Number of recommendations per entity')
    parser.add_argument('--output', type=str, default=None, help='Output file path')
    
    args = parser.parse_args()
    
    results = generate_batch_recommendations(
        top_k=args.top_k,
        output_file=args.output
    )
    
    if results:
        print("\n✅ Batch generation complete!")
        print(f"📄 Results ready for API consumption")
    else:
        print("\n❌ Batch generation failed")


if __name__ == "__main__":
    main()
