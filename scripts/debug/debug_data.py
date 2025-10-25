# Save as debug_data.py
from src.data_processor import DataProcessor

processor = DataProcessor()
data = processor.load_all_data()

print("=" * 60)
print("DATA LOADED:")
print("=" * 60)
print(f"Individuals: {len(data['individuals'])}")
print(f"Projects: {len(data['project_calls'])}")

if data['individuals']:
    print("\n" + "=" * 60)
    print("SAMPLE INDIVIDUAL:")
    print("=" * 60)
    ind = data['individuals'][0]
    print(f"Name: {ind.get('fullName')}")
    print(f"Type: {ind.get('type')}")
    print(f"Location: {ind.get('location')}")
    print(f"Skills: {[s.get('skill') for s in ind.get('skills', [])]}")
    print(f"Areas of Expertise: {[a.get('industry') for a in ind.get('areasOfExpertise', [])]}")
    print(f"Preferences: {ind.get('preferences')}")

if data['project_calls']:
    print("\n" + "=" * 60)
    print("SAMPLE PROJECT:")
    print("=" * 60)
    proj = data['project_calls'][0]
    print(f"Title: {proj.get('title')}")
    print(f"Status: {proj.get('status')}")
    print(f"Location: {proj.get('location')}")
    print(f"Delivery: {proj.get('delivery')}")
    print(f"Applicant Types: {proj.get('applicantTypes')}")
    print(f"Type: {proj.get('type')}")
