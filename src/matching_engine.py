"""
Core matching engine for smart recommendations
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from src.vector_store import VectorStore
import config


class MatchingEngine:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.embeddings_data = {}
        
    def set_embeddings_data(self, embeddings_data: Dict[str, List[Dict]]):
        """Set the embeddings data for reference"""
        self.embeddings_data = embeddings_data
    
    def get_user_preferences(self, user_data: Dict) -> Dict[str, Any]:
        """Extract user preferences for filtering"""
        preferences = user_data.get('preferences', {})
        
        # Determine what user is seeking
        seeking = []
        if preferences.get('collaborations'):
            seeking.append('collaboration')
        if preferences.get('grantFundedProjects'):
            seeking.append('research')
            seeking.append('grant')
        if preferences.get('funding'):
            seeking.append('funding')
        if preferences.get('consulting'):
            seeking.append('consulting')
        if preferences.get('opportunities'):
            seeking.append('opportunity')
        
        return {
            'seeking': seeking,
            'location': user_data.get('location', 'any'),
            'user_type': user_data.get('type', 'unknown'),
            'skills': [skill.get('skill', '') for skill in user_data.get('skills', [])],
            'expertise': [area.get('industry', '') for area in user_data.get('areasOfExpertise', [])]
        }
    
    def apply_compatibility_filters(self, projects: List[Dict], user_data: Dict) -> List[Dict]:
        """Filter projects based on compatibility criteria"""
        user_prefs = self.get_user_preferences(user_data)
        user_type = user_prefs['user_type']
        user_location = user_prefs['location']
        
        compatible_projects = []
        
        for project in projects:
            project_data = project['original_data']
            
            # Check applicant type compatibility
            applicant_types = project_data.get('applicantTypes', [])
            
            # Map user types to project applicant types (UPDATED WITH ORGANIZATIONS)
            type_mapping = {
                # Individual types
                'student/early-career': ['student'],
                'student': ['student'],
                'professional': ['professional', 'research'],
                'entrepreneur': ['entrepreneur', 'startup'],
                
                # Organization types (EXPANDED)
                'research_institute': ['research', 'research_center', 'academic_institution'],
                'startup': ['startup', 'company', 'entrepreneur'],
                'company': ['company', 'startup', 'professional'],
                'non_profit': ['non_profit', 'research', 'academic_institution'],
                'publicInstitution': ['research', 'academic_institution', 'government_agency'],
                'accelerator': ['startup', 'entrepreneur', 'company'],
                'academic_institution': ['academic_institution', 'research', 'student'],
                
                # Fallback for unknown types
                'unknown': ['professional', 'research']
            }
            
            user_compatible_types = type_mapping.get(user_type, [user_type])
            
            # Check if user type is compatible with project
            if applicant_types:  # Only filter if applicant types are specified
                if not any(utype in applicant_types for utype in user_compatible_types):
                    continue
            
            # Check location compatibility (flexible matching)
            project_location = project_data.get('location', 'any')
            delivery = project_data.get('delivery', 'in-person')
            
            location_compatible = (
                user_location == project_location or
                user_location == 'other' or  # 'other' users are flexible
                project_location == 'other' or
                delivery in ['online-virtual', 'hybrid']  # Remote work possible
            )
            
            if not location_compatible:
                continue
            
            # Check if project status is active
            if project_data.get('status') != 'Active':
                continue
            
            compatible_projects.append(project)
        
        return compatible_projects
    
    def calculate_relevance_score(self, project: Dict, user_data: Dict, similarity_score: float) -> Dict[str, Any]:
        """Calculate comprehensive relevance score"""
        user_prefs = self.get_user_preferences(user_data)
        project_data = project['original_data']
        
        # Start with semantic similarity (0-1)
        semantic_score = max(0, similarity_score)  # Ensure positive
        
        # Applicant type match score
        applicant_types = project_data.get('applicantTypes', [])
        user_type = user_prefs['user_type']
        
        type_score = 0.8  # Default decent match
        if user_type == 'student' and 'student' in applicant_types:
            type_score = 1.0
        elif user_type == 'professional' and 'professional' in applicant_types:
            type_score = 1.0
        elif user_type == 'entrepreneur' and 'entrepreneur' in applicant_types:
            type_score = 1.0
        elif user_type == 'research_institute' and any(t in applicant_types for t in ['research', 'research_center', 'academic_institution']):
            type_score = 1.0
        elif user_type == 'startup' and any(t in applicant_types for t in ['startup', 'company', 'entrepreneur']):
            type_score = 1.0
        
        # Delivery preference score
        delivery = project_data.get('delivery', 'in-person')
        delivery_score = 0.7  # Default
        if delivery in ['online-virtual', 'hybrid']:
            delivery_score = 0.9  # Slightly prefer flexible delivery
        
        # Duration preference (students might prefer shorter projects)
        duration = project_data.get('duration', 'unknown')
        duration_score = 0.8  # Default
        if user_type == 'student':
            if duration == 'short-term':
                duration_score = 1.0
            elif duration == 'medium-term':
                duration_score = 0.9
        
        # Project type alignment with user preferences
        project_type = project_data.get('type', 'unknown')
        type_alignment_score = 0.7  # Default
        
        seeking = user_prefs['seeking']
        if 'research' in seeking and 'research' in project_type:
            type_alignment_score = 1.0
        elif 'grant' in seeking and 'grant' in project_type:
            type_alignment_score = 1.0
        elif 'collaboration' in seeking and 'collaboration' in project_type:
            type_alignment_score = 1.0
        elif 'funding' in seeking and 'accelerator' in project_type:
            type_alignment_score = 0.9
        
        # Combine scores with weights
        final_score = (
            semantic_score * 0.4 +          # 40% semantic similarity
            type_score * 0.25 +             # 25% applicant type match
            type_alignment_score * 0.20 +   # 20% project type alignment
            delivery_score * 0.10 +         # 10% delivery flexibility
            duration_score * 0.05           # 5% duration fit
        )
        
        return {
            'final_score': final_score,
            'semantic_similarity': semantic_score,
            'applicant_type_match': type_score,
            'project_type_alignment': type_alignment_score,
            'delivery_compatibility': delivery_score,
            'duration_fit': duration_score
        }
    
    def generate_match_reasons(self, project: Dict, user_data: Dict, scores: Dict) -> List[str]:
        """Generate human-readable reasons for the match"""
        reasons = []
        project_data = project['original_data']
        user_prefs = self.get_user_preferences(user_data)
        
        # High semantic similarity
        if scores['semantic_similarity'] > 0.7:
            reasons.append("Strong skill and expertise alignment")
        elif scores['semantic_similarity'] > 0.5:
            reasons.append("Good skill compatibility")
        
        # Applicant type match
        if scores['applicant_type_match'] >= 1.0:
            user_type = user_prefs['user_type']
            if user_type in ['research_institute', 'startup', 'company', 'non_profit']:
                reasons.append(f"Perfect organizational fit")
            else:
                reasons.append(f"Perfect fit for {user_type} profile")
        elif scores['applicant_type_match'] > 0.8:
            reasons.append("Good applicant profile match")
        
        # Project type alignment
        if scores['project_type_alignment'] >= 1.0:
            reasons.append("Matches your stated preferences")
        elif scores['project_type_alignment'] > 0.8:
            reasons.append("Aligns with your interests")
        
        # Delivery method
        delivery = project_data.get('delivery', 'in-person')
        if delivery in ['online-virtual', 'hybrid']:
            reasons.append("Flexible delivery options available")
        
        # Duration appropriateness
        duration = project_data.get('duration', 'unknown')
        user_type = user_prefs['user_type']
        if user_type == 'student' and duration == 'short-term':
            reasons.append("Good duration for student schedule")
        
        # Budget availability
        budget = project_data.get('budget')
        if budget and budget != 'Not specified':
            reasons.append("Funded opportunity")
        
        return reasons[:4]  # Limit to top 4 reasons
    
    def find_recommendations(self, user_id: str, entity_type: str = 'individuals', 
                           top_k: int = 10) -> List[Dict]:
        """Find project recommendations for a user"""
        
        # Find user in embeddings data
        user_item = None
        for item in self.embeddings_data.get(entity_type, []):
            if item.get('id') == user_id:
                user_item = item
                break
        
        if not user_item:
            print(f"❌ User {user_id} not found in {entity_type}")
            return []
        
        user_embedding = user_item['embedding']
        user_data = user_item['original_data']
        
        print(f"Finding recommendations for: {user_data.get('fullName', user_data.get('name', user_id))}")
        
        # Get similar projects using vector search
        similar_projects = self.vector_store.search(
            user_embedding, 
            k=top_k * 2,  # Get extra for filtering
            entity_types=['project_calls']
        )
        
        if not similar_projects:
            print("❌ No similar projects found")
            return []
        
        # Apply compatibility filters
        compatible_projects = self.apply_compatibility_filters(similar_projects, user_data)
        
        print(f"Found {len(compatible_projects)} compatible projects (from {len(similar_projects)} similar)")
        
        # Score and rank projects
        recommendations = []
        for project in compatible_projects[:top_k]:  # Limit to requested amount
            
            # Calculate comprehensive scores
            scores = self.calculate_relevance_score(
                project, user_data, project['similarity_score']
            )
            
            # Generate match reasons
            reasons = self.generate_match_reasons(project, user_data, scores)
            
            recommendation = {
                'project_id': project['entity_id'],
                'project_title': project['original_data'].get('title', 'Unknown'),
                'project_type': project['original_data'].get('type', 'Unknown'),
                'organization_name': project['original_data'].get('organization', {}).get('name', 'Unknown'),
                'match_score': scores['final_score'],
                'confidence': 'high' if scores['final_score'] > 0.8 else 'medium' if scores['final_score'] > 0.6 else 'low',
                'match_reasons': reasons,
                'score_breakdown': scores,
                'project_summary': {
                    'summary': project['original_data'].get('summary', '')[:200],
                    'duration': project['original_data'].get('duration', 'Unknown'),
                    'location': project['original_data'].get('location', 'Unknown'),
                    'budget': project['original_data'].get('budget', 'Not specified'),
                    'delivery': project['original_data'].get('delivery', 'Unknown'),
                    'deadline': project['original_data'].get('deadline', 'Not specified')
                }
            }
            recommendations.append(recommendation)
        
        # Sort by final match score
        recommendations.sort(key=lambda x: x['match_score'], reverse=True)
        
        return recommendations
