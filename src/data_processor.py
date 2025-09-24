"""
Data processor for normalizing JSON input data
"""
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import config


class DataProcessor:
    def __init__(self):
        self.data_dir = config.SAMPLE_DATA_DIR
        self.loaded_data = {
            'individuals': [],
            'organizations': [],
            'project_calls': []
        }

    def load_sample_data(self, data_type: str) -> List[Dict]:
        """Load sample JSON data for specified type"""
        valid_types = ['individuals', 'organizations', 'project_calls']

        if data_type not in valid_types:
            raise ValueError(f"Invalid data type. Must be one of: {valid_types}")

        file_path = self.data_dir / f"{data_type}.json"

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
                # Parse applicantTypes JSON strings in projects if needed
                if data_type == 'project_calls':
                    for item in data:
                        if 'applicantTypes' in item and isinstance(item['applicantTypes'], str):
                            try:
                                item['applicantTypes'] = json.loads(item['applicantTypes'])
                            except Exception:
                                item['applicantTypes'] = []

                self.loaded_data[data_type] = data
                print(f"✓ Loaded {len(data)} {data_type}")
                return data
        except FileNotFoundError:
            print(f"✗ File not found: {file_path}")
            return []
        except json.JSONDecodeError as e:
            print(f"✗ JSON decode error in {file_path}: {e}")
            return []
        except Exception as e:
            print(f"✗ Error loading {file_path}: {e}")
            return []

    def load_all_data(self) -> Dict[str, List[Dict]]:
        """Load all sample data types"""
        print("Loading sample data...")

        individuals = self.load_sample_data('individuals')
        organizations = self.load_sample_data('organizations')
        project_calls = self.load_sample_data('project_calls')

        return {
            'individuals': individuals,
            'organizations': organizations,
            'project_calls': project_calls
        }

    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary statistics of loaded data"""
        summary = {}

        for data_type, data_list in self.loaded_data.items():
            if data_list:
                summary[data_type] = {
                    'count': len(data_list),
                    'sample_names': self._extract_sample_names(data_type, data_list)
                }
            else:
                summary[data_type] = {'count': 0, 'sample_names': []}

        return summary

    def _extract_sample_names(self, data_type: str, data_list: List[Dict]) -> List[str]:
        """Extract sample names for display"""
        if not data_list:
            return []

        if data_type == 'individuals':
            return [item.get('fullName', 'Unknown') for item in data_list[:3]]
        elif data_type == 'organizations':
            return [item.get('name', 'Unknown') for item in data_list[:3]]
        elif data_type == 'project_calls':
            return [item.get('title', 'Unknown') for item in data_list[:3]]

        return []

    def get_sample_record(self, data_type: str) -> Optional[Dict]:
        """Get one sample record for inspection"""
        if data_type in self.loaded_data and self.loaded_data[data_type]:
            return self.loaded_data[data_type][0]
        return None

    def validate_data_structure(self) -> Dict[str, bool]:
        """Validate that loaded data has expected structure"""
        validation_results = {}

        # Check individuals structure
        individuals = self.loaded_data.get('individuals', [])
        if individuals:
            sample = individuals[0]
            required_fields = ['id', 'fullName', 'skills', 'areasOfExpertise', 'preferences']
            validation_results['individuals'] = all(field in sample for field in required_fields)
        else:
            validation_results['individuals'] = False

        # Check organizations structure (corrected: areasOfInterest spelling)
        organizations = self.loaded_data.get('organizations', [])
        if organizations:
            sample = organizations[0]
            required_fields = ['id', 'name', 'industry', 'areasOfInterest', 'preferences']
            validation_results['organizations'] = all(field in sample for field in required_fields)
        else:
            validation_results['organizations'] = False

        # Check project_calls structure
        project_calls = self.loaded_data.get('project_calls', [])
        if project_calls:
            sample = project_calls[0]
            required_fields = ['id', 'title', 'summary', 'requirements', 'applicantTypes']
            validation_results['project_calls'] = all(field in sample for field in required_fields)
        else:
            validation_results['project_calls'] = False

        return validation_results

    # Placeholder methods for future normalization (Step 2)
    def normalize_individual(self, individual_data: Dict) -> Dict:
        """Normalize individual JSON to standard format"""
        pass

    def normalize_organization(self, org_data: Dict) -> Dict:
        """Normalize organization JSON to standard format"""
        pass

    def normalize_project_call(self, project_data: Dict) -> Dict:
        """Normalize project call JSON to standard format"""
        pass
