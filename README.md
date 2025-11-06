# 🔍 News Fact Validation Graph

**Neurosymbolic AI for Misinformation Detection**

A Track 1 project combining Large Language Models (LLMs) with symbolic reasoning to detect contradictions in news articles using temporal knowledge graphs.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5%2FGPT--4-green.svg)](https://openai.com/)
[![NetworkX](https://img.shields.io/badge/NetworkX-3.1+-orange.svg)](https://networkx.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## ✅ Project Status: **FULLY IMPLEMENTED & TESTED**

- ✅ All core features implemented
- ✅ Demo successfully runs with GPT-3.5-turbo
- ✅ Contradiction detection working
- ✅ Interactive visualizations generated
- ✅ Comprehensive documentation complete

**Last tested:** Successfully ran on macOS with Python 3.12.4

---

## 🎯 Project Overview

This system addresses the challenge of misinformation detection by combining:

1. **Neural (LLM)**: Extract entities, events, and temporal facts from text
2. **Symbolic (Logic Rules)**: Apply formal reasoning rules to detect contradictions
3. **Graph Neural Networks**: Learn patterns in fact relationships
4. **Knowledge Graphs**: Represent facts with temporal and semantic structure

### Example Use Case

**Input:**
```
"City X reported zero COVID cases yesterday.
The same source confirmed a rise in cases earlier that day."
```

**Output:**
- ✅ Temporal knowledge graph with events and entities
- ⚠️ Detected contradiction: "zero cases" vs "rise in cases" on the same day
- ✍️ Corrected narrative with explanation
- 📊 Interactive visualization highlighting the inconsistency

---

## 🏗️ Architecture

```
Input Article → LLM Extraction → Knowledge Graph → Symbolic Logic Rules
                                                   ↓
Output: Corrected Narrative ← Narrative Generation ← GNN Analysis
```

**Components:**
1. **LLM Extraction (OpenAI)**: Entities, events, temporal facts
2. **Temporal Knowledge Graph (NetworkX)**: Nodes (entities/events) + Edges (relationships)
3. **Symbolic Logic Engine**: Contradiction detection rules
4. **GNN (PyTorch Geometric)**: Pattern learning and anomaly detection
5. **Narrative Correction**: Generate fixed version with explanation

---

## 🚀 Quick Start

### 1. Installation

```bash
# Navigate to project directory
cd News_detector

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-your-api-key-here
```

### 3. Run the Demo

```bash
# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run the demo
python demo.py
```

**Note:** The system uses GPT-3.5-turbo by default. If you have access to GPT-4, you can change the model in `.env`:
```
OPENAI_MODEL=gpt-4-turbo-preview
```

This will:
- Process the COVID example article
- Extract entities and events using LLM
- Build temporal knowledge graph
- Apply symbolic logic rules
- Detect contradictions
- Generate corrected narrative
- Create visualizations in `outputs/graphs/`

**Output files will automatically open in your browser:**
- `knowledge_graph.html` - Interactive graph
- `timeline.html` - Event timeline
- `report.html` - Full analysis report

---

## 📁 Project Structure

```
News_detector/
├── demo.py                    # Main demonstration script
├── requirements.txt           # Python dependencies
├── README.md                 # Project documentation
├── .env.example              # Environment template
│
├── src/
│   ├── llm/                  # LLM integration
│   │   └── llm_client.py
│   ├── extraction/           # Entity/event extraction
│   │   └── entity_extractor.py
│   ├── models/               # Data models
│   │   └── fact.py
│   ├── graph/                # Knowledge graph
│   │   ├── temporal_graph.py
│   │   └── graph_builder.py
│   ├── reasoning/            # Symbolic logic
│   │   ├── logic_rules.py
│   │   └── contradiction_detector.py
│   ├── gnn/                  # Graph Neural Networks
│   │   ├── graph_embeddings.py
│   │   └── gnn_model.py
│   ├── correction/           # Narrative correction
│   │   └── narrative_generator.py
│   ├── visualization/        # Graph visualization
│   │   └── graph_visualizer.py
│   └── utils/                # Utilities
│       ├── config.py
│       └── time_utils.py
│
└── outputs/
    └── graphs/               # Generated visualizations
```

---

## 🧠 Neurosymbolic AI Features

### Neural Component (LLM)
- Entity recognition (locations, sources, people)
- Event extraction with temporal information
- Relationship extraction
- Temporal expression parsing ("yesterday" → timestamp)

### Symbolic Component (Logic Rules)

#### 1. Temporal Consistency Rule
Events should follow logical temporal ordering

#### 2. Mutual Exclusivity Rule
Contradictory facts cannot both be true

#### 3. Source Consistency Rule
Same source should not contradict itself

#### 4. Event Ordering Rule
Effects cannot occur before causes

### Graph Neural Network
- Learn patterns in contradiction structures
- Anomaly detection in fact graphs
- Credibility scoring
- Combine with symbolic rules for final verdict

---

## 📊 Example Output (Actual Results)

### Knowledge Graph
```
Nodes: 3 (1 entity, 2 events)
Edges: 1 relationship
Contradictions: 1 HIGH severity
```

### Detected Contradiction
```
Type: Mutual Exclusivity
Severity: 1.0

Fact 1: "City X reported zero COVID cases" (yesterday 14:00)
Fact 2: "a rise in cases" (earlier that day)

Explanation: Mutually exclusive: 'zero' conflicts with 'rise'
```

### Corrected Narrative
```
Original:
"City X reported zero COVID cases yesterday.
The same source confirmed a rise in cases earlier that day."

Corrected:
"City X reported zero new COVID cases yesterday.
The same source confirmed a rise in cases earlier in the week."

Explanation:
Changed "zero COVID cases" to "zero new COVID cases" to clarify
that no new cases were reported, but there may still be existing cases.
Also specified that the rise in cases occurred earlier in the week,
rather than on the same day, to resolve the contradiction.

Confidence: 0.90
```

### Visualizations Generated
- **Interactive Knowledge Graph**: Color-coded nodes with contradictions in red
- **Timeline View**: Chronological event ordering
- **Comprehensive Report**: Full analysis with statistics

---

## 🔧 Customization

### Add Custom Logic Rules

Edit `src/reasoning/logic_rules.py`:

```python
class CustomRule(LogicRule):
    def __init__(self):
        super().__init__("Rule Name", "Description")

    def check(self, event1: Dict, event2: Dict):
        # Your logic here
        if violation:
            return True, severity, "Explanation"
        return False, 0.0, "No violation"
```

### Train GNN Model

```python
from src.gnn.gnn_model import FactVerificationGNN
from src.gnn.graph_embeddings import GraphEmbedding

# Convert graph
embedding = GraphEmbedding(feature_dim=32)
pyg_data = embedding.networkx_to_pyg(graph)

# Train model
model = FactVerificationGNN(input_dim=32)
# Training loop...
```

---

## 🎓 Key Technologies

- **OpenAI GPT-3.5-turbo/GPT-4**: Entity and event extraction, narrative generation
- **NetworkX**: Graph manipulation and symbolic reasoning
- **PyTorch Geometric**: Graph neural networks for pattern learning
- **PyVis**: Interactive graph visualization
- **dateparser**: Temporal expression parsing
- **Python 3.8+**: Core implementation language

---

## 📈 Future Extensions

- [ ] Neo4j integration for large-scale graphs
- [ ] Multi-article cross-referencing
- [ ] Source credibility scoring
- [ ] Real-time fact-checking API
- [ ] Fine-tuned GNN on misinformation datasets
- [ ] Multi-language support

---

## 🧪 Testing & Troubleshooting

### Run the Demo
```bash
# Activate virtual environment
source venv/bin/activate

# Run demo with default COVID example
python demo.py

# View outputs
open outputs/graphs/report.html
open outputs/graphs/knowledge_graph.html
open outputs/graphs/timeline.html
```

### Common Issues

#### Issue: "Model not found" or "No access to gpt-4"
**Solution:** Use GPT-3.5-turbo instead (works with all OpenAI accounts)
```bash
# Edit .env file
OPENAI_MODEL=gpt-3.5-turbo
```

#### Issue: "Invalid API key"
**Solution:** Get a new API key from https://platform.openai.com/api-keys
```bash
# Update .env file
OPENAI_API_KEY=sk-proj-your-new-key-here
```

#### Issue: No visualizations generated
**Solution:** Check that outputs directory exists
```bash
mkdir -p outputs/graphs
python demo.py
```

### Testing with Custom Articles

Edit `demo.py` line 23 to test with your own news:
```python
article_text = """Your custom article text here..."""
```

Or use sample articles from the dataset:
```python
import json
with open('data/sample_articles.json') as f:
    articles = json.load(f)
    # Test with article #2
    article_text = articles[1]['text']
```

---

## 📚 Core Concepts

**Temporal Knowledge Graph**: Nodes (entities/events) with timestamps, edges (relationships) with temporal constraints

**Neurosymbolic AI**: Combines neural networks (pattern learning) with symbolic logic (rule-based reasoning)

**Contradiction Detection**: Multi-level approach using temporal logic, semantic analysis, and source consistency

**Narrative Correction**: LLM-based rewriting that preserves facts while removing contradictions

---

## 📄 License

MIT License

---

## 🙏 Acknowledgments

Built for **Track 1: Neurosymbolic AI**
Theme: *Bring logic back into machine learning*

Technologies: OpenAI, PyTorch Geometric, NetworkX, PyVis

---

**⭐ Star this repo if you find it useful!**
