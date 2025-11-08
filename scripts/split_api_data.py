"""
Split API response into separate JSON files
Reads: data/api_response.json
Creates: 
  - data/sample/individuals.json
  - data/sample/organizations.json
  - data/sample/project_calls.json

Usage: python3 scripts/split_api_data.py
"""
import json
from pathlib import Path


def split_api_data():
    """Split API response into three separate files"""
    
    # Input file
    api_file = Path("data/api_response.json")
    
    # Check if file exists
    if not api_file.exists():
        print(f"ERROR: {api_file} not found")
        print("\nFetch data first with:")
        print("  curl -H 'Authorization: Bearer YOUR_TOKEN' \\")
        print("       -H 'Accept: application/json' \\")
        print("       https://match4research.com/api/network/machine-learning > data/api_response.json")
        return False
    
    # Load the API response
    print(f"Reading {api_file}...")
    with open(api_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded. Keys found: {list(data.keys())}")
    
    # Create output directory
    output_dir = Path("data/sample")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved = []
    
    # 1. Save individuals
    individuals = data.get('individuals', [])
    if individuals:
        with open(output_dir / 'individuals.json', 'w', encoding='utf-8') as f:
            json.dump({"individuals": individuals}, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(individuals)} individuals")
        saved.append('individuals.json')
    else:
        print("WARNING: No individuals found")
    
    # 2. Save organizations
    organizations = data.get('organizations', [])
    if organizations:
        with open(output_dir / 'organizations.json', 'w', encoding='utf-8') as f:
            json.dump({"organizations": organizations}, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(organizations)} organizations")
        saved.append('organizations.json')
    else:
        print("WARNING: No organizations found")
    
    # 3. Save project_calls (from openCalls key)
    open_calls = data.get('openCalls', [])
    if open_calls:
        with open(output_dir / 'project_calls.json', 'w', encoding='utf-8') as f:
            json.dump({"project_calls": open_calls}, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(open_calls)} project calls")
        saved.append('project_calls.json')
    else:
        print("WARNING: No openCalls found")
    
    # Summary
    print(f"\nDone! Created {len(saved)}/3 files in {output_dir}/")
    for filename in saved:
        print(f"  - {filename}")
    
    if len(saved) == 3:
        print("\nNext steps:")
        print("  1. rm -rf data/processed/*")
        print("  2. python3 main.py")
        return True
    else:
        print("\nWARNING: Not all files were created")
        return False


if __name__ == "__main__":
    split_api_data()
