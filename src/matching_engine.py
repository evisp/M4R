"""
Core matching engine for smart recommendations
"""
import numpy as np
from typing import Dict, List, Any
import json  # JSON parsing for applicantTypes field
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
        seeking = []
        if preferences.get('collaborations'):
            seeking.append('collaboration')
        if preferences.get('grantFundedProjects'):
            seeking.extend(['research', 'grant'])
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
            'expertise': [area.get('industry', '') for area in user_data.get('areasOfExpertise', [])],
        }

    def apply_compatibility_filters(self, projects: List[Dict], user_data: Dict) -> List[Dict]:
        """Filter projects based on compatibility criteria"""
        user_prefs = self.get_user_preferences(user_data)
        user_type = user_prefs['user_type']
        user_location = user_prefs['location']

        # FIXED type mappings - bridge the gap between user types and project applicant types
        type_mapping = {
            # Individual types - map to what projects expect
            'student/early-career': ['student', 'student/early-career'],
            'professional/consultant': ['professional', 'professional/consultant'],

            # Organization types - map to what projects expect  
            'publicInstitution': ['professional', 'research', 'academic_institution'],
            'startup': ['startup', 'company', 'entrepreneur'],

            # Project applicant types (keep for reverse compatibility)
            'professional': ['professional'],
            'student': ['student'],
            'research_center': ['research_center'],
            'company': ['company'],
            'entrepreneur': ['entrepreneur'],
            'research': ['research'],
            'non_profit': ['non_profit'],
            'government_agency': ['government_agency'],
            'academic_institution': ['academic_institution'],

            # Fallback
            'unknown': ['professional', 'student']
        }

        user_compatible_types = type_mapping.get(user_type, [user_type])
        compatible_projects = []

        for project in projects:
            project_data = project.get('original_data', {})

            # Parse applicantTypes (handle string or list)
            applicant_types_raw = project_data.get('applicantTypes', [])
            applicant_types = []
            if isinstance(applicant_types_raw, str):
                try:
                    applicant_types = json.loads(applicant_types_raw)
                except Exception:
                    applicant_types = []
            elif isinstance(applicant_types_raw, list):
                applicant_types = applicant_types_raw

            if applicant_types:
                if not any(utype in applicant_types for utype in user_compatible_types):
                    continue

            # Location compatibility check
            project_location = project_data.get('location', 'any')
            delivery = project_data.get('delivery', 'in-person')
            location_compatible = (
                user_location == project_location or
                user_location == 'other' or
                project_location == 'other' or
                delivery in ['online-virtual', 'hybrid']
            )
            if not location_compatible:
                continue

            if project_data.get('status', '').lower() != 'active':
                continue

            compatible_projects.append(project)

        return compatible_projects

    def calculate_relevance_score(self, project: Dict, user_data: Dict, similarity_score: float) -> Dict[str, Any]:
        """Calculate comprehensive relevance score"""
        user_prefs = self.get_user_preferences(user_data)
        project_data = project.get('original_data', {})

        semantic_score = max(0, similarity_score)

        applicant_types = project_data.get('applicantTypes', [])
        user_type = user_prefs['user_type']

        # Enhanced type scoring for better matching
        type_score = 0.8  # Default
        
        if user_type == 'professional/consultant' and 'professional' in applicant_types:
            type_score = 1.0
        elif user_type == 'student/early-career' and 'student' in applicant_types:
            type_score = 1.0
        elif user_type == 'publicInstitution' and any(t in applicant_types for t in ['research', 'academic_institution', 'professional']):
            type_score = 1.0
        elif user_type == 'startup' and any(t in applicant_types for t in ['startup', 'company', 'entrepreneur']):
            type_score = 1.0

        delivery = project_data.get('delivery', 'in-person')
        delivery_score = 0.7
        if delivery in ['online-virtual', 'hybrid']:
            delivery_score = 0.9

        duration = project_data.get('duration', 'unknown')
        duration_score = 0.8
        if 'student' in user_type and duration == 'short-term':
            duration_score = 1.0
        elif 'student' in user_type and duration == 'medium-term':
            duration_score = 0.9

        project_type = project_data.get('type', 'unknown')
        type_alignment_score = 0.7
        seeking = user_prefs['seeking']
        if 'research' in seeking and 'funding-opportunity' in project_type:
            type_alignment_score = 1.0
        elif 'grant' in seeking and ('fellowship' in project_type or 'funding' in project_type):
            type_alignment_score = 1.0
        elif 'collaboration' in seeking:
            type_alignment_score = 0.9
        elif 'funding' in seeking and 'funding' in project_type:
            type_alignment_score = 0.9

        final_score = (
            semantic_score * 0.4 +
            type_score * 0.25 +
            type_alignment_score * 0.20 +
            delivery_score * 0.10 +
            duration_score * 0.05
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
        project_data = project.get('original_data', {})
        user_prefs = self.get_user_preferences(user_data)

        if scores['semantic_similarity'] > 0.7:
            reasons.append("Strong skill and expertise alignment")
        elif scores['semantic_similarity'] > 0.5:
            reasons.append("Good skill compatibility")

        if scores['applicant_type_match'] >= 1.0:
            user_type = user_prefs['user_type']
            if user_type in ['publicInstitution', 'startup']:
                reasons.append("Perfect organizational fit")
            else:
                reasons.append(f"Perfect fit for your profile")
        elif scores['applicant_type_match'] > 0.8:
            reasons.append("Good applicant profile match")

        if scores['project_type_alignment'] >= 1.0:
            reasons.append("Matches your stated preferences")
        elif scores['project_type_alignment'] > 0.8:
            reasons.append("Aligns with your interests")

        delivery = project_data.get('delivery', 'in-person')
        if delivery in ['online-virtual', 'hybrid']:
            reasons.append("Flexible delivery options available")

        duration = project_data.get('duration', 'unknown')
        user_type = user_prefs['user_type']
        if 'student' in user_type and duration == 'short-term':
            reasons.append("Good duration for student schedule")

        budget = project_data.get('budget')
        if budget and budget != 'Not specified':
            reasons.append("Funded opportunity")

        return reasons[:4]

    def find_recommendations(self, user_id: str, entity_type: str = 'individuals',
                             top_k: int = 10) -> List[Dict]:
        """Find project recommendations for a user"""
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

        similar_projects = self.vector_store.search(
            user_embedding,
            k=top_k * 2,
            entity_types=['project_calls']
        )

        if not similar_projects:
            print("❌ No similar projects found")
            return []

        compatible_projects = self.apply_compatibility_filters(similar_projects, user_data)

        print(f"Found {len(compatible_projects)} compatible projects (from {len(similar_projects)} similar)")

        recommendations = []
        for project in compatible_projects[:top_k]:
            scores = self.calculate_relevance_score(
                project, user_data, project.get('similarity_score', 0)
            )
            reasons = self.generate_match_reasons(project, user_data, scores)

            recommendation = {
                'project_id': project.get('entity_id'),
                'project_title': project.get('original_data', {}).get('title', 'Unknown'),
                'project_type': project.get('original_data', {}).get('type', 'Unknown'),
                'organization_name': project.get('original_data', {}).get('organization', {}).get('name', 'Unknown'),
                'match_score': scores['final_score'],
                'confidence': 'high' if scores['final_score'] > 0.8 else 'medium' if scores['final_score'] > 0.6 else 'low',
                'match_reasons': reasons,
                'score_breakdown': scores,
                'project_summary': {
                    'summary': project.get('original_data', {}).get('summary', '')[:200],
                    'duration': project.get('original_data', {}).get('duration', 'Unknown'),
                    'location': project.get('original_data', {}).get('location', 'Unknown'),
                    'budget': project.get('original_data', {}).get('budget', 'Not specified'),
                    'delivery': project.get('original_data', {}).get('delivery', 'Unknown'),
                    'deadline': project.get('original_data', {}).get('deadline', 'Not specified'),
                }
            }
            recommendations.append(recommendation)

        recommendations.sort(key=lambda x: x['match_score'], reverse=True)

        return recommendations
