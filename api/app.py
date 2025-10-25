"""
FastAPI REST API for AI Recommendations
Serves pre-computed recommendations from JSON file
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json
from typing import Optional, List
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
import config

app = FastAPI(
    title="Match4Research AI API",
    description="AI-powered recommendation system for researchers and organizations",
    version="1.0.0"
)

# CORS middleware (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load recommendations on startup
RECOMMENDATIONS_FILE = config.PROCESSED_DATA_DIR / "output" / "recommendations.json"
recommendations_data = {}

@app.on_event("startup")
async def load_recommendations():
    """Load recommendations from JSON file"""
    global recommendations_data
    try:
        with open(RECOMMENDATIONS_FILE, 'r', encoding='utf-8') as f:
            recommendations_data = json.load(f)
        print(f"✅ Loaded recommendations: {recommendations_data['metadata']['total_individuals']} individuals, "
              f"{recommendations_data['metadata']['total_organizations']} organizations")
    except FileNotFoundError:
        print(f"⚠️  Recommendations file not found at {RECOMMENDATIONS_FILE}")
        print("   Run: python3 generate_batch_recommendations.py")
    except Exception as e:
        print(f"❌ Error loading recommendations: {e}")


@app.get("/")
async def root():
    """API health check"""
    return {
        "status": "healthy",
        "service": "Match4Research AI API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "individual_recommendations": "/api/recommendations/individual/{individual_id}",
            "organization_recommendations": "/api/recommendations/organization/{organization_id}",
            "statistics": "/api/statistics",
            "all_recommendations": "/api/recommendations/all"
        }
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "recommendations_loaded": bool(recommendations_data),
        "total_individuals": recommendations_data.get('metadata', {}).get('total_individuals', 0),
        "total_organizations": recommendations_data.get('metadata', {}).get('total_organizations', 0),
        "total_projects": recommendations_data.get('metadata', {}).get('total_projects', 0),
        "last_generated": recommendations_data.get('metadata', {}).get('generated_at', 'unknown')
    }


@app.get("/api/recommendations/individual/{individual_id}")
async def get_individual_recommendations(
    individual_id: str,
    top_k: Optional[int] = Query(None, description="Limit number of recommendations")
):
    """Get recommendations for a specific individual"""
    if not recommendations_data:
        raise HTTPException(status_code=503, detail="Recommendations not loaded")
    
    # Find individual in recommendations
    individuals = recommendations_data.get('recommendations', {}).get('individuals', [])
    individual_recs = next((ind for ind in individuals if ind['individual_id'] == individual_id), None)
    
    if not individual_recs:
        raise HTTPException(status_code=404, detail=f"Individual {individual_id} not found")
    
    # Apply top_k filter if requested
    if top_k and top_k > 0:
        individual_recs = individual_recs.copy()
        individual_recs['recommendations'] = individual_recs['recommendations'][:top_k]
        individual_recs['match_count'] = len(individual_recs['recommendations'])
    
    return {
        "success": True,
        "data": individual_recs
    }


@app.get("/api/recommendations/organization/{organization_id}")
async def get_organization_recommendations(
    organization_id: str,
    top_k: Optional[int] = Query(None, description="Limit number of recommendations")
):
    """Get recommendations for a specific organization"""
    if not recommendations_data:
        raise HTTPException(status_code=503, detail="Recommendations not loaded")
    
    # Find organization in recommendations
    organizations = recommendations_data.get('recommendations', {}).get('organizations', [])
    org_recs = next((org for org in organizations if org['organization_id'] == organization_id), None)
    
    if not org_recs:
        raise HTTPException(status_code=404, detail=f"Organization {organization_id} not found")
    
    # Apply top_k filter if requested
    if top_k and top_k > 0:
        org_recs = org_recs.copy()
        org_recs['recommendations'] = org_recs['recommendations'][:top_k]
        org_recs['match_count'] = len(org_recs['recommendations'])
    
    return {
        "success": True,
        "data": org_recs
    }


@app.get("/api/recommendations/all")
async def get_all_recommendations(
    entity_type: Optional[str] = Query(None, description="Filter by 'individual' or 'organization'")
):
    """Get all recommendations"""
    if not recommendations_data:
        raise HTTPException(status_code=503, detail="Recommendations not loaded")
    
    if entity_type:
        if entity_type == 'individual':
            return {
                "success": True,
                "entity_type": "individuals",
                "count": len(recommendations_data['recommendations']['individuals']),
                "data": recommendations_data['recommendations']['individuals']
            }
        elif entity_type == 'organization':
            return {
                "success": True,
                "entity_type": "organizations",
                "count": len(recommendations_data['recommendations']['organizations']),
                "data": recommendations_data['recommendations']['organizations']
            }
        else:
            raise HTTPException(status_code=400, detail="entity_type must be 'individual' or 'organization'")
    
    return {
        "success": True,
        "data": recommendations_data['recommendations'],
        "metadata": recommendations_data['metadata']
    }


@app.get("/api/statistics")
async def get_statistics():
    """Get recommendation statistics"""
    if not recommendations_data:
        raise HTTPException(status_code=503, detail="Recommendations not loaded")
    
    return {
        "success": True,
        "metadata": recommendations_data.get('metadata', {}),
        "statistics": recommendations_data.get('statistics', {})
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
