# Match4Research - AI Matching System

AI-guided recommendation system that connects individuals and organizations with relevant project opportunities using semantic similarity.

## Installation & Setup

### 1. Clone and Setup Environment
```bash
git clone https://github.com/yourusername/M4R.git
cd M4R
python3 -m venv venv
source venv/bin/activate  
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

This installs:
- `sentence-transformers` (AI model)
- `faiss-cpu` (vector search)
- `requests` (API calls)
- `python-dotenv` (config)
- `numpy` (numerical computing)

### 3. Configure API Token

Create a `.env` file in project root:
`M4R_API_TOKEN=your_token_here`

IMPORTANT: Never commit `.env` to git (already in `.gitignore`)

## Usage

### Generate Recommendations (One Command)

```bash
python3 fetch_and_build.py
```

What this does:
1. Fetches data from Match4Research API
2. Builds AI embeddings (semantic vectors)
3. Creates FAISS search index
4. Generates recommendations for all users/organizations
5. Saves to: data/processed/output/recommendations.json

Time: 2-5 minutes first run, 10-30 seconds after (uses cache)

### Alternative Commands
- `python3 fetch_and_build.py --fetch-only`    # Update data only
- `python3 fetch_and_build.py --build-only`    # Rebuild with existing data
- `python3 fetch_and_build.py --top-k 10`      # Generate 10 recommendations per entity
- `python3 test_matching.py`                   # Interactive testing interface

## How It Works

The system uses a 5-factor scoring algorithm:
- Semantic similarity: 55% (AI-based skill/expertise matching)
- Type match: 20% (profile compatibility)
- Preferences: 15% (stated interests)
- Delivery: 7% (location/remote fit)
- Duration: 3% (time commitment)

Confidence levels:
- High (>0.70): Strong matches
- Medium (0.50-0.70): Good matches
- Low (<0.50): Weak matches

## Integration

The output file (`data/processed/output/recommendations.json`) is ready for:
1. Direct frontend consumption
2. REST API integration (see api/app.py)
3. Database import

## Maintenance

Update recommendations (run daily/weekly):
`python3 fetch_and_build.py`

Clear cache if data structure changes:
`rm -rf data/processed/*`

