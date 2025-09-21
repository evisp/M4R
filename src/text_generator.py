"""
Text generator for creating embedding-ready text representations
"""
from typing import Dict, List, Any, Optional


class TextGenerator:
    def __init__(self):
        pass
    
    def generate_individual_text(self, individual_data: Dict) -> str:
        """Generate text representation for an individual"""
        
        # Extract key information
        name = individual_data.get('fullName', 'Unknown')
        title = individual_data.get('title', 'Unknown')
        bio = individual_data.get('bio', '').replace('\n', ' ').strip()
        location = individual_data.get('location', 'Unknown')
        availability = individual_data.get('availability', 'Unknown')
        individual_type = individual_data.get('type', 'Unknown')
        
        # Extract skills
        skills_raw = individual_data.get('skills', [])
        skills = [skill.get('skill', '') for skill in skills_raw if skill.get('skill')]
        skills_text = ', '.join(skills) if skills else 'Not specified'
        
        # Extract areas of expertise
        expertise_raw = individual_data.get('areasOfExpertise', [])
        expertise = [area.get('industry', '') for area in expertise_raw if area.get('industry')]
        expertise_text = ', '.join(expertise) if expertise else 'Not specified'
        
        # Extract preferences (what they're seeking)
        preferences = individual_data.get('preferences', {})
        seeking = []
        if preferences.get('collaborations'):
            seeking.append('collaborations')
        if preferences.get('grantFundedProjects'):
            seeking.append('grant-funded projects')
        if preferences.get('funding'):
            seeking.append('funding opportunities')
        if preferences.get('consulting'):
            seeking.append('consulting work')
        if preferences.get('opportunities'):
            seeking.append('career opportunities')
        
        seeking_text = ', '.join(seeking) if seeking else 'open to opportunities'
        
        # Extract education info if available
        education = individual_data.get('education', {})
        degree = education.get('degree', '') if education else ''
        
        # Construct text representation
        text_parts = [
            f"Profile: {name}",
            f"Role: {title}",
            f"Type: {individual_type}"
        ]
        
        if bio and len(bio) > 10:  # Only include bio if it's meaningful
            text_parts.append(f"Bio: {bio[:200]}")  # Limit bio length
        
        text_parts.extend([
            f"Skills: {skills_text}",
            f"Expertise: {expertise_text}",
            f"Location: {location}",
            f"Availability: {availability}",
            f"Seeking: {seeking_text}"
        ])
        
        if degree:
            text_parts.append(f"Education: {degree}")
        
        return " | ".join(text_parts)
    
    def generate_organization_text(self, org_data: Dict) -> str:
        """Generate text representation for an organization"""
        
        # Extract key information
        name = org_data.get('name', 'Unknown')
        org_type = org_data.get('type', 'Unknown')
        industry = org_data.get('industry', 'Unknown')
        description = org_data.get('description', '').strip()
        location = org_data.get('location', 'Unknown')
        team_size = org_data.get('teamSize', 'Unknown')
        
        # Extract areas of interest
        interests_raw = org_data.get('areasOfInterests', [])
        interests = [area.get('industry', '') for area in interests_raw if area.get('industry')]
        interests_text = ', '.join(interests) if interests else 'Not specified'
        
        # Extract preferences (what they're seeking)
        preferences = org_data.get('preferences', {})
        seeking = []
        if preferences.get('collaborations'):
            seeking.append('collaborations')
        if preferences.get('grantFundedProjects'):
            seeking.append('grant-funded projects')
        if preferences.get('funding'):
            seeking.append('funding opportunities')
        if preferences.get('consulting'):
            seeking.append('consulting services')
        if preferences.get('opportunities'):
            seeking.append('partnership opportunities')
        
        seeking_text = ', '.join(seeking) if seeking else 'open to partnerships'
        
        # Construct text representation
        text_parts = [
            f"Organization: {name}",
            f"Type: {org_type}",
            f"Industry: {industry}",
            f"Size: {team_size}",
            f"Location: {location}"
        ]
        
        if description and len(description) > 10:
            text_parts.append(f"Description: {description[:300]}")  # Limit description length
        
        text_parts.extend([
            f"Interests: {interests_text}",
            f"Seeking: {seeking_text}"
        ])
        
        return " | ".join(text_parts)
    
    def generate_project_text(self, project_data: Dict) -> str:
        """Generate text representation for a project call"""
        
        # Extract key information
        title = project_data.get('title', 'Unknown')
        project_type = project_data.get('type', 'Unknown')
        summary = project_data.get('summary', '').strip()
        requirements = project_data.get('requirements', '').strip()
        location = project_data.get('location', 'Unknown')
        duration = project_data.get('duration', 'Unknown')
        delivery = project_data.get('delivery', 'Unknown')
        budget = project_data.get('budget', 'Not specified')
        
        # Extract target applicant types
        applicant_types = project_data.get('applicantTypes', [])
        applicants_text = ', '.join(applicant_types) if applicant_types else 'All applicants'
        
        # Extract organization info
        organization = project_data.get('organization', {})
        org_name = organization.get('name', 'Unknown Organization')
        org_industry = organization.get('industry', 'Unknown')
        
        # Get who should apply info
        who_should_apply = project_data.get('whoShouldApply', '').strip()
        
        # Construct text representation
        text_parts = [
            f"Project: {title}",
            f"Type: {project_type}",
            f"Duration: {duration}",
            f"Location: {location}",
            f"Delivery: {delivery}"
        ]
        
        if budget != 'Not specified':
            text_parts.append(f"Budget: {budget}")
        
        if summary and len(summary) > 10:
            text_parts.append(f"Summary: {summary[:400]}")  # Limit summary length
        
        if requirements and len(requirements) > 10:
            text_parts.append(f"Requirements: {requirements[:300]}")  # Limit requirements length
        
        if who_should_apply and len(who_should_apply) > 10:
            text_parts.append(f"Target: {who_should_apply[:200]}")
        
        text_parts.extend([
            f"Applicant Types: {applicants_text}",
            f"Organization: {org_name}",
            f"Industry: {org_industry}"
        ])
        
        return " | ".join(text_parts)
    
    def generate_text_for_dataset(self, data: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """Generate text representations for entire dataset"""
        
        results = {
            'individuals': [],
            'organizations': [],
            'project_calls': []
        }
        
        # Process individuals
        for individual in data.get('individuals', []):
            text_repr = self.generate_individual_text(individual)
            results['individuals'].append({
                'id': individual.get('id'),
                'original_data': individual,
                'text_representation': text_repr,
                'entity_type': 'individual'
            })
        
        # Process organizations  
        for organization in data.get('organizations', []):
            text_repr = self.generate_organization_text(organization)
            results['organizations'].append({
                'id': organization.get('id'),
                'original_data': organization,
                'text_representation': text_repr,
                'entity_type': 'organization'
            })
        
        # Process project calls
        for project in data.get('project_calls', []):
            text_repr = self.generate_project_text(project)
            results['project_calls'].append({
                'id': project.get('id'),
                'original_data': project,
                'text_representation': text_repr,
                'entity_type': 'project_call'
            })
        
        return results
    
    def get_text_stats(self, text_representations: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Get statistics about generated text representations"""
        
        stats = {}
        
        for entity_type, items in text_representations.items():
            if items:
                texts = [item['text_representation'] for item in items]
                text_lengths = [len(text) for text in texts]
                
                stats[entity_type] = {
                    'count': len(items),
                    'avg_length': sum(text_lengths) / len(text_lengths),
                    'min_length': min(text_lengths),
                    'max_length': max(text_lengths)
                }
            else:
                stats[entity_type] = {
                    'count': 0,
                    'avg_length': 0,
                    'min_length': 0,
                    'max_length': 0
                }
        
        return stats
