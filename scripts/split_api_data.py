"""
Split API response into separate JSON files
Usage: python3 scripts/split_api_data.py data/api_response.json
"""
import json
import sys
from pathlib import Path

def split_api_data(api_file_path):
    """Split API response into individuals, organizations, project_calls"""
    
    api_file = Path(api_file_path)
    
    if not api_file.exists():
        print(f"❌ File not found: {api_file_path}")
        print("\nFetch data first:")
        print("  curl -H 'Authorization: Bearer TOKEN' \\")
        print("       -H 'Accept: application/json' \\")
        print("       https://match4research.com/api/network/machine-learning \\")
        print("       -o data/api_response.json")
        return False
    
    print(f"📂 Reading {api_file_path}...")
    with open(api_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ Loaded API response")
    print(f"   Keys: {list(data.keys())}")
    
    # Create output directory
    output_dir = Path("data/sample")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_count = 0
    
    # Save individuals
    individuals = data.get('individuals', [])
    if individuals:
        individuals_file = output_dir / 'individuals.json'
        with open(individuals_file, 'w', encoding='utf-8') as f:
            json.dump({
                "individuals": individuals,
                "pagination": {"totalCount": len(individuals)}
            }, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved {len(individuals)} individuals")
        saved_count += 1
    else:
        print("⚠️  No individuals found")
    
    # Save organizations
    organizations = data.get('organizations', [])
    if organizations:
        orgs_file = output_dir / 'organizations.json'
        with open(orgs_file, 'w', encoding='utf-8') as f:
            json.dump({
                "organizations": organizations,
                "pagination": {"totalCount": len(organizations)}
            }, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved {len(organizations)} organizations")
        saved_count += 1
    else:
        print("⚠️  No organizations found")
    
    # Save project calls
    project_calls = data.get('openCalls', data.get('project_calls', []))
    if project_calls:
        projects_file = output_dir / 'project_calls.json'
        with open(projects_file, 'w', encoding='utf-8') as f:
            json.dump({
                "project_calls": project_calls,
                "pagination": {"totalCount": len(project_calls)}
            }, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved {len(project_calls)} project calls")
        saved_count += 1
    else:
        print("⚠️  No project calls found in API response")
        print("   Ask dev team to add 'openCalls' array to API")
    
    if saved_count > 0:
        print(f"\n✅ Done! {saved_count} files ready in data/sample/")
        print("\nNext steps:")
        print("  1. rm -rf data/processed/*")
        print("  2. python3 main.py")
        print("  3. python3 test_matching.py")
        return True
    else:
        print("\n❌ No data to save")
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1:
        api_file = sys.argv[1]
    else:
        api_file = "data/api_response.json"
    
    split_api_data(api_file)
