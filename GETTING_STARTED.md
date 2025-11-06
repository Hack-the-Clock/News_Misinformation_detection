# Getting Started with News Fact Validation Graph

This guide will help you get up and running with the Neurosymbolic AI fact validation system in under 10 minutes.

---

## Prerequisites

- **Python 3.8+** installed on your system
- **OpenAI API key** (get one at https://platform.openai.com/api-keys)
- **Basic command line knowledge**

---

## Installation Options

### Option 1: Quick Start Script (Recommended)

```bash
cd News_detector
./quick_start.sh
```

This script will:
1. Create a virtual environment
2. Install all dependencies
3. Set up configuration files
4. Run the demo (if API key is configured)

### Option 2: Manual Installation

```bash
# 1. Create virtual environment
python3 -m venv venv

# 2. Activate it
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API key
cp .env.example .env
# Edit .env and add your OpenAI API key

# 5. Run demo
python demo.py
```

---

## Configuration

### Step 1: Get OpenAI API Key

1. Visit https://platform.openai.com/api-keys
2. Create an account or log in
3. Click "Create new secret key"
4. Copy the key (starts with `sk-...`)

### Step 2: Configure Environment

Edit the `.env` file:

```bash
# .env file
OPENAI_API_KEY=sk-your-actual-key-here
OPENAI_MODEL=gpt-4-turbo-preview
```

**Models you can use:**
- `gpt-4-turbo-preview` - Best quality, slower, more expensive
- `gpt-3.5-turbo` - Fast, cheap, good for testing

---

## Running Your First Demo

### Basic Demo

```bash
python demo.py
```

**What happens:**
1. Processes the COVID contradiction example
2. Extracts entities and events
3. Builds knowledge graph
4. Detects contradictions
5. Generates corrected narrative
6. Creates visualizations

**Expected output:**
```
🔍 News Fact Validation Graph - Neurosymbolic AI Demo
========================================

📄 Input Article:
City X reported zero COVID cases yesterday...

✓ Extracted 2 entities
✓ Extracted 2 events
✓ Graph built with 5 nodes and 6 edges
✓ Detected 1 contradictions

✅ Demo Completed Successfully!
```

### View Results

Open the generated files in your browser:

```bash
# On Mac
open outputs/graphs/report.html
open outputs/graphs/knowledge_graph.html
open outputs/graphs/timeline.html

# On Linux
xdg-open outputs/graphs/report.html

# On Windows
start outputs/graphs/report.html
```

---

## Understanding the Output

### 1. Knowledge Graph (`knowledge_graph.html`)

**Interactive visualization showing:**
- 🟢 **Green nodes**: Entities (City X, Source)
- 🔵 **Blue nodes**: Events (reported zero cases, rise in cases)
- 🔴 **Red nodes/edges**: Contradictions
- **Hover** over nodes/edges for details
- **Drag** nodes to rearrange
- **Zoom** in/out with mouse wheel

### 2. Timeline (`timeline.html`)

**Chronological view of events:**
- Events ordered by timestamp
- Earlier events at the top
- Shows temporal relationships

### 3. Report (`report.html`)

**Comprehensive analysis including:**
- Graph statistics
- All detected contradictions
- Original vs corrected text
- Explanation of changes
- Confidence scores

---

## Try Your Own Examples

### Example 1: Custom Article

Create a file `my_article.txt`:

```
The company announced record profits in Q1.
However, the CEO stated they had significant
losses during the same quarter.
```

Process it:

```python
# custom_demo.py
from demo import *

article = open('my_article.txt').read()

# Run same pipeline
llm_client = LLMClient()
extractor = EntityExtractor(llm_client)
result = extractor.extract_all(article)

# ... continue with rest of pipeline
```

### Example 2: Using Sample Articles

```python
import json
from demo import *

# Load sample articles
with open('data/sample_articles.json') as f:
    articles = json.load(f)

# Process each article
for article in articles:
    print(f"\nProcessing: {article['title']}")
    # Run pipeline...
```

---

## Common Issues and Solutions

### Issue 1: API Key Not Found

**Error:**
```
ValueError: OPENAI_API_KEY not found
```

**Solution:**
1. Make sure `.env` file exists
2. Check that API key is correctly set
3. Restart your terminal/IDE after setting it

### Issue 2: Module Not Found

**Error:**
```
ModuleNotFoundError: No module named 'openai'
```

**Solution:**
```bash
pip install -r requirements.txt
```

### Issue 3: Rate Limit Error

**Error:**
```
RateLimitError: You exceeded your current quota
```

**Solutions:**
- **Option 1**: Add billing info to your OpenAI account
- **Option 2**: Use `gpt-3.5-turbo` (cheaper model)
- **Option 3**: Wait and retry (free tier has limits)

### Issue 4: Torch/PyTorch Geometric Installation

**Error:**
```
Could not find torch-geometric
```

**Solution:**
```bash
pip install torch torchvision
pip install torch-geometric
```

If still failing, check: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

---

## Next Steps

### 1. Understand the Code

Read the key modules in this order:

1. **[src/models/fact.py](src/models/fact.py)** - Data structures
2. **[src/llm/llm_client.py](src/llm/llm_client.py)** - LLM integration
3. **[src/extraction/entity_extractor.py](src/extraction/entity_extractor.py)** - Extraction logic
4. **[src/graph/temporal_graph.py](src/graph/temporal_graph.py)** - Graph operations
5. **[src/reasoning/logic_rules.py](src/reasoning/logic_rules.py)** - Symbolic rules

### 2. Customize Logic Rules

Add your own contradiction detection rules:

```python
# In src/reasoning/logic_rules.py

class MyCustomRule(LogicRule):
    def __init__(self):
        super().__init__(
            "My Rule Name",
            "What this rule checks"
        )

    def check(self, event1, event2):
        # Your logic here
        if my_condition:
            return True, 0.8, "Explanation"
        return False, 0.0, "No violation"
```

Register it in the engine:

```python
# In src/reasoning/logic_rules.py
class LogicRuleEngine:
    def __init__(self):
        self.rules = [
            TemporalConsistencyRule(),
            MutualExclusivityRule(),
            SourceConsistencyRule(),
            EventOrderingRule(),
            MyCustomRule(),  # Add here
        ]
```

### 3. Train the GNN (Advanced)

```python
from src.gnn.gnn_model import FactVerificationGNN
from src.gnn.graph_embeddings import GraphEmbedding

# Convert graph to PyTorch format
embedding = GraphEmbedding(feature_dim=32)
pyg_data = embedding.networkx_to_pyg(knowledge_graph.graph)

# Initialize model
model = FactVerificationGNN(
    input_dim=32,
    hidden_dim=64,
    num_layers=2
)

# Training requires labeled data
# See ARCHITECTURE.md for details
```

### 4. Integrate with Your Application

```python
from src.llm.llm_client import LLMClient
from src.extraction.entity_extractor import EntityExtractor
from src.graph.graph_builder import GraphBuilder
from src.reasoning.contradiction_detector import ContradictionDetector

def validate_article(article_text):
    """Main validation function"""

    # Initialize components
    llm = LLMClient()
    extractor = EntityExtractor(llm)
    detector = ContradictionDetector()

    # Extract information
    extraction = extractor.extract_all(article_text)

    # Build graph
    builder = GraphBuilder()
    graph = builder.build_from_extraction(extraction)

    # Detect contradictions
    contradictions = detector.detect_contradictions(graph)

    return {
        'has_contradictions': len(contradictions) > 0,
        'contradictions': contradictions,
        'graph': graph
    }

# Use it
result = validate_article("Your article text here...")
print(f"Found {len(result['contradictions'])} contradictions")
```

---

## Learning Resources

### Documentation
- **[README.md](README.md)** - Project overview
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Detailed architecture
- **This file** - Getting started guide

### Code Examples
- **[demo.py](demo.py)** - Complete working example
- **[data/sample_articles.json](data/sample_articles.json)** - Test cases

### External Resources
- **OpenAI API**: https://platform.openai.com/docs
- **NetworkX**: https://networkx.org/documentation/
- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/
- **Neurosymbolic AI**: https://arxiv.org/abs/2105.05330

---

## Project Structure Quick Reference

```
News_detector/
├── demo.py                     ← Start here
├── requirements.txt            ← Dependencies
├── .env                        ← Your API key (create this)
├── README.md                   ← Overview
├── GETTING_STARTED.md          ← This file
├── ARCHITECTURE.md             ← Deep dive
│
├── src/
│   ├── llm/                    ← LLM integration
│   ├── extraction/             ← Extract entities/events
│   ├── graph/                  ← Knowledge graph
│   ├── reasoning/              ← Logic rules
│   ├── gnn/                    ← Neural networks
│   ├── correction/             ← Generate corrections
│   ├── visualization/          ← Create visualizations
│   └── utils/                  ← Helpers
│
├── data/
│   └── sample_articles.json    ← Test cases
│
└── outputs/
    └── graphs/                 ← Generated files
```

---

## Support

### Having Issues?

1. **Check** common issues section above
2. **Read** error messages carefully
3. **Search** for similar issues online
4. **Ask** on the project's issue tracker

### Want to Contribute?

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

## Summary

You should now be able to:

- ✅ Install and configure the system
- ✅ Run the demo with the COVID example
- ✅ View and understand the outputs
- ✅ Try your own news articles
- ✅ Understand the code structure
- ✅ Customize logic rules
- ✅ Integrate into your own projects

**Next:** Try processing different articles from `data/sample_articles.json` to see how the system handles various types of contradictions!

---

**Happy fact-checking! 🔍**
