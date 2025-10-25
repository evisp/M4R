"""
Fetch data from Match4Research API and cache locally
"""
import requests
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


class APIDataFetcher:
    def __init__(self, bearer_token: str, cache_dir: str = None):
        self.base_url = "https://match4research.com/api/network/machine-learning"
        self.bearer_token = bearer_token
        
        # Set cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            project_root = Path(__file__).parent.parent
            self.cache_dir = project_root / "data" / "sample"
        
    def fetch_all_data(self):
        """Fetch all data from API - mimics browser request"""
        print("🔄 Fetching data from API...")
        print(f"   Endpoint: {self.base_url}")
        
        try:
            # Mimic browser headers to avoid 406 error
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
                'Authorization': f'Bearer {self.bearer_token}',
                'Accept': 'application/json',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Sec-Fetch-Dest': 'empty',
                'Sec-Fetch-Mode': 'cors',
                'Sec-Fetch-Site': 'same-origin'
            }
            
            response = requests.get(
                self.base_url,
                headers=headers,
                timeout=30
            )
            
            print(f"   Status Code: {response.status_code}")
            
            response.raise_for_status()
            
            data = response.json()
            print(f"✅ API Response received")
            
            # Show what keys we got
            if isinstance(data, dict):
                print(f"   Response keys: {list(data.keys())}")
                if 'individuals' in data:
                    print(f"   - {len(data['individuals'])} individuals")
                if 'organizations' in data:
                    print(f"   - {len(data['organizations'])} organizations")
                if 'openCalls' in data:
                    print(f"   - {len(data['openCalls'])} open calls")
                if 'project_calls' in data:
                    print(f"   - {len(data['project_calls'])} project calls")
            
            return data
            
        except requests.exceptions.HTTPError as e:
            print(f"❌ HTTP Error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"   Response text: {e.response.text[:500]}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"❌ Request Error: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"❌ JSON Parse Error: {e}")
            return None
    
    def save_to_cache(self, data: dict):
        """Save fetched data to local JSON files"""
        if not data:
            print("❌ No data to save")
            return False
        
        print("\n💾 Saving data to cache...")
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        saved_count = 0
        
        # Save individuals
        individuals = data.get('individuals', [])
        if individuals:
            individuals_file = self.cache_dir / 'individuals.json'
            with open(individuals_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "individuals": individuals,
                    "pagination": {"totalCount": len(individuals)}
                }, f, indent=2, ensure_ascii=False)
            print(f"   ✅ Saved {len(individuals)} individuals")
            saved_count += 1
        else:
            print(f"   ⚠️  No individuals found")
        
        # Save organizations
        organizations = data.get('organizations', [])
        if organizations:
            orgs_file = self.cache_dir / 'organizations.json'
            with open(orgs_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "organizations": organizations,
                    "pagination": {"totalCount": len(organizations)}
                }, f, indent=2, ensure_ascii=False)
            print(f"   ✅ Saved {len(organizations)} organizations")
            saved_count += 1
        else:
            print(f"   ⚠️  No organizations found")
        
        # Save project calls - handle both "openCalls" and "project_calls"
        project_calls = data.get('openCalls', data.get('project_calls', []))
        if project_calls:
            projects_file = self.cache_dir / 'project_calls.json'
            with open(projects_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "project_calls": project_calls,
                    "pagination": {"totalCount": len(project_calls)}
                }, f, indent=2, ensure_ascii=False)
            print(f"   ✅ Saved {len(project_calls)} project calls")
            saved_count += 1
        else:
            print(f"   ⚠️  No project calls found")
        
        # Save metadata
        metadata = {
            "fetched_at": datetime.utcnow().isoformat() + "Z",
            "api_endpoint": self.base_url,
            "total_individuals": len(individuals),
            "total_organizations": len(organizations),
            "total_projects": len(project_calls)
        }
        
        metadata_file = self.cache_dir / 'metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        print(f"   ✅ Saved metadata")
        
        return saved_count > 0
    
    def fetch_and_cache(self):
        """Main method: fetch from API and save to cache"""
        print("=" * 60)
        print("FETCHING DATA FROM API")
        print("=" * 60)
        
        data = self.fetch_all_data()
        
        if data:
            success = self.save_to_cache(data)
            if success:
                print("\n" + "=" * 60)
                print("✅ DATA SUCCESSFULLY CACHED")
                print("=" * 60)
                print(f"\n📂 Files saved to: {self.cache_dir}")
                print("\nNext steps:")
                print("  1. rm -rf data/processed/*")
                print("  2. python3 main.py")
                print("  3. python3 test_matching.py")
                return True
        
        print("\n❌ Failed to fetch and cache data")
        return False


def main():
    """Main execution"""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Fetch data from API')
    parser.add_argument('--token', type=str, help='Bearer token')
    parser.add_argument('--cache-dir', type=str, help='Cache directory')
    
    args = parser.parse_args()
    
    token = args.token or os.getenv('M4R_API_TOKEN')
    
    if not token:
        print("❌ Error: Bearer token required")
        print("\nUsage:")
        print("  python3 scripts/fetch_data_from_api.py --token YOUR_TOKEN")
        print("  OR")
        print("  export M4R_API_TOKEN=YOUR_TOKEN")
        print("  python3 scripts/fetch_data_from_api.py")
        return
    
    print(f"🔑 Token: {token[:20]}...")
    
    fetcher = APIDataFetcher(bearer_token=token, cache_dir=args.cache_dir)
    success = fetcher.fetch_and_cache()
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
