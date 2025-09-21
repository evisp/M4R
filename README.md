# Match4Research - Smart Matching System

## Project Goal

`Match4Research` connects individuals and organizations with relevant project opportunities using semantic understanding and intelligent filtering. The system analyzes user profiles and project requirements to provide ranked recommendations with detailed explanations.

## Core Components

**Data Processing** - Loads and validates user profiles, organization data, and project calls from JSON files

**Text Analysis** - Converts structured data into semantic text representations optimized for machine learning

**Embedding Generation** - Creates vector embeddings using sentence transformers for semantic similarity analysis

**Vector Storage** - FAISS-based vector database enabling fast similarity search across large datasets

**Matching Engine** - Combines semantic similarity with rule-based filtering to generate scored recommendations

## Data Flow

```
JSON Data → Text Representations → Vector Embeddings → FAISS Index → Similarity Search → Filtered Results → Scored Recommendations
```

## Installation and Usage

### Setup
```
git clone https://github.com/yourusername/M4R.git
cd M4R
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Data Preparation
Place JSON files in `data/sample/`:
- `individuals.json` - User profiles
- `organizations.json` - Organization profiles  
- `project_calls.json` - Project opportunities

### System Build
```
python3 main.py  # Initial build (2-5 minutes)
```

### Interactive Testing
```
python3 test_matching.py  # Fast testing interface (2-3 seconds)
```


