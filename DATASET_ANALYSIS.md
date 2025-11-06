# 📊 Ideal Dataset Analysis for GNN Training

## Executive Summary

Based on analysis of your News Fact Validation Graph codebase, here are the **ideal datasets** for training your GNN model, ranked by suitability:

### 🏆 Top Recommendations

1. **FEVER Dataset** ⭐⭐⭐⭐⭐ (BEST FIT)
2. **LIAR Dataset** ⭐⭐⭐⭐ (GOOD ALTERNATIVE)
3. **FakeNewsNet** ⭐⭐⭐⭐ (NEWS-SPECIFIC)
4. **PolitiFact** ⭐⭐⭐ (DOMAIN-SPECIFIC)
5. **Custom Dataset** ⭐⭐⭐⭐⭐ (IDEAL BUT REQUIRES WORK)

---

## 🔍 Your System Requirements

### What Your GNN Needs:

1. **Input Format**: 
   - Raw text articles/claims → Converted to knowledge graphs
   - Your system already handles: `Article Text → LLM Extraction → NetworkX Graph → PyTorch Geometric`

2. **Label Format**:
   - Binary classification: `0` = No Contradiction, `1` = Contradiction
   - Your GNN model: `FactVerificationGNN(output_dim=2)`

3. **Graph Structure**:
   - Nodes: Entities (locations, organizations, people) + Events (with timestamps)
   - Edges: Relations (reported, contradicts, supports, temporal_before, etc.)
   - Features: Node type, confidence, timestamps, entity types

4. **Minimum Dataset Size**:
   - Minimum: 1,000 labeled examples
   - Recommended: 5,000+ examples
   - Ideal: 10,000+ examples

5. **Balance**:
   - 50% contradictory examples
   - 50% non-contradictory examples

---

## 📚 Dataset Analysis

### 1. FEVER Dataset ⭐⭐⭐⭐⭐

**Why It's Perfect for Your System:**

✅ **Direct Match**: Designed for fact verification and contradiction detection  
✅ **Large Size**: 185,445 labeled claims  
✅ **English Language**: Matches your current system  
✅ **Label Format**: SUPPORTS, REFUTES, NOT ENOUGH INFO (maps perfectly to your binary classification)  
✅ **Evidence-Based**: Includes evidence sentences (useful for graph construction)  
✅ **Accessibility**: Available on HuggingFace (`datasets` library)  
✅ **Quality**: Human-annotated by Stanford NLP researchers  

**Dataset Structure:**
```python
{
    'claim': "The Rodney King riots took place in the most populous county in the USA.",
    'label': 'REFUTES',  # or 'SUPPORTS' or 'NOT ENOUGH INFO'
    'evidence': [
        ['Rodney_King', 0, 'The 1992 Los Angeles riots...'],
        ...
    ]
}
```

**Mapping to Your System:**
- `REFUTES` → `has_contradiction = True` (label = 1)
- `SUPPORTS` → `has_contradiction = False` (label = 0)
- `NOT ENOUGH INFO` → Can be filtered out or treated as negative

**How to Use:**
```python
from datasets import load_dataset

dataset = load_dataset('fever', 'v1.0', split='train')
# Your system converts: claim + evidence → graph → GNN input
```

**Pros:**
- ✅ Perfect alignment with your contradiction detection task
- ✅ Large dataset (185K examples)
- ✅ High quality annotations
- ✅ Easy to load via HuggingFace
- ✅ Evidence sentences help build richer graphs

**Cons:**
- ⚠️ Claims are shorter than full news articles (but still workable)
- ⚠️ Need to combine claim + evidence for graph construction

**Integration Effort**: ⭐ Low (2-3 hours) - Already planned in `DATASET_INTEGRATION_PLAN.md`

---

### 2. LIAR Dataset ⭐⭐⭐⭐

**Why It's Good:**

✅ **News-Specific**: Real-world political statements from PolitiFact  
✅ **Size**: 12,836 labeled statements  
✅ **Labels**: 6 categories (can be binarized to true/false)  
✅ **Context**: Includes speaker, context, and statement text  
✅ **Temporal**: Includes dates (useful for temporal graphs)  

**Dataset Structure:**
```python
{
    'statement': "Hillary Clinton said that, at this point, what difference does it make.",
    'label': 'false',  # true, mostly-true, half-true, mostly-false, false, pants-fire
    'subject': 'hillary-clinton',
    'speaker': 'Hillary Clinton',
    'job_title': 'Former U.S. Senator',
    'state_info': 'New York',
    'party_affiliation': 'Democrat',
    'barely_true_counts': 0,
    'false_counts': 1,
    'half_true_counts': 0,
    'mostly_true_counts': 0,
    'pants_fire_counts': 0,
    'context': 'A hearing on Benghazi.'
}
```

**Mapping to Your System:**
- `false`, `pants-fire` → `has_contradiction = True` (label = 1)
- `true`, `mostly-true` → `has_contradiction = False` (label = 0)
- `half-true`, `mostly-false` → Can be filtered or treated as contradictions

**Pros:**
- ✅ Real-world political statements (news domain)
- ✅ Includes temporal information (dates)
- ✅ Rich metadata (speaker, context)
- ✅ Good for temporal contradiction detection

**Cons:**
- ⚠️ Smaller than FEVER (12K vs 185K)
- ⚠️ Statements are shorter than full articles
- ⚠️ Political bias (may not generalize to all news)

**Integration Effort**: ⭐⭐ Medium (3-4 hours)

---

### 3. FakeNewsNet ⭐⭐⭐⭐

**Why It's Relevant:**

✅ **News Articles**: Full news articles (not just claims)  
✅ **Size**: ~20,000 articles  
✅ **Labels**: Real vs Fake news  
✅ **Sources**: Includes source credibility  
✅ **Temporal**: Publication dates included  

**Dataset Structure:**
```python
{
    'id': 'article_123',
    'title': 'Article Title',
    'text': 'Full article text...',
    'label': 'fake',  # or 'real'
    'source': 'source_url',
    'date': '2020-01-15',
    'author': 'Author Name'
}
```

**Mapping to Your System:**
- `fake` → `has_contradiction = True` (label = 1) - if article contains contradictions
- `real` → `has_contradiction = False` (label = 0)

**Note**: This requires additional processing to identify specific contradictions within articles, not just fake vs real.

**Pros:**
- ✅ Full news articles (matches your use case)
- ✅ Temporal information
- ✅ Source metadata
- ✅ Real-world news domain

**Cons:**
- ⚠️ Labels are "fake vs real" not "contradiction vs no contradiction"
- ⚠️ Need to identify contradictions within fake articles
- ⚠️ May require additional annotation

**Integration Effort**: ⭐⭐⭐ High (5-6 hours) - Requires contradiction extraction

---

### 4. PolitiFact Dataset ⭐⭐⭐

**Why It's Useful:**

✅ **Fact-Checking**: Direct fact-checking labels  
✅ **Size**: ~10,000+ statements  
✅ **Rich Metadata**: Speaker, context, dates  
✅ **Verification**: Includes verification details  

**Similar to LIAR but from PolitiFact directly.**

**Pros:**
- ✅ High-quality fact-checking labels
- ✅ Temporal information
- ✅ Rich context

**Cons:**
- ⚠️ Political domain only
- ⚠️ Statements, not full articles
- ⚠️ May need additional processing

**Integration Effort**: ⭐⭐ Medium (3-4 hours)

---

### 5. Custom Dataset ⭐⭐⭐⭐⭐

**Why It's Ideal (But Requires Work):**

✅ **Perfect Fit**: Tailored to your exact needs  
✅ **Full Articles**: Real news articles with contradictions  
✅ **Domain-Specific**: Can focus on your target domain  
✅ **Quality Control**: You control annotation quality  

**How to Create:**

1. **Collect Articles** (1,000+):
   - News articles from various sources
   - Mix of contradictory and non-contradictory articles

2. **Annotation**:
   - Hire annotators (Amazon Mechanical Turk, Prolific)
   - Expert annotation (domain experts)
   - Self-annotation (if you have expertise)

3. **Format**:
```json
[
  {
    "text": "Full article text...",
    "has_contradiction": true,
    "contradiction_type": "temporal",
    "contradiction_severity": 1.0,
    "contradiction_facts": [
      {"fact1": "...", "fact2": "...", "type": "mutual_exclusivity"}
    ],
    "source": "news_source",
    "date": "2024-01-15"
  }
]
```

**Pros:**
- ✅ Perfect alignment with your system
- ✅ Full control over quality
- ✅ Domain-specific
- ✅ Can include temporal contradictions

**Cons:**
- ❌ Requires significant time and effort
- ❌ Cost (if hiring annotators)
- ❌ Need annotation guidelines

**Integration Effort**: ⭐⭐⭐⭐ Very High (20+ hours) - But best long-term solution

---

## 🎯 Recommendation Matrix

| Dataset | Size | Quality | Integration | Fit | Overall |
|---------|------|---------|-------------|-----|---------|
| **FEVER** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **LIAR** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **FakeNewsNet** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **PolitiFact** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Custom** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 🚀 Recommended Approach

### Phase 1: Quick Start (FEVER) - **RECOMMENDED**

**Timeline**: 2-3 hours  
**Why**: Fastest path to working GNN model

1. Use FEVER dataset (already planned in your `DATASET_INTEGRATION_PLAN.md`)
2. Implement `FEVERDatasetLoader` (code already provided)
3. Train GNN on 5,000-10,000 examples
4. Evaluate and iterate

**Expected Results:**
- Validation Accuracy: 75-85%
- Working GNN model
- Proof of concept

### Phase 2: Enhancement (Custom Dataset) - **LONG-TERM**

**Timeline**: 2-4 weeks  
**Why**: Best performance and domain fit

1. Collect 1,000-5,000 news articles
2. Annotate contradictions (or use your symbolic rules to pre-label)
3. Create custom dataset
4. Fine-tune GNN on custom data

**Expected Results:**
- Validation Accuracy: 80-90%+
- Domain-specific performance
- Production-ready model

---

## 📋 Dataset Requirements Checklist

For any dataset you choose, ensure it has:

- [ ] **Text Input**: Articles, claims, or statements
- [ ] **Binary Labels**: Contradiction (1) vs No Contradiction (0)
- [ ] **Minimum Size**: 1,000+ examples (5,000+ recommended)
- [ ] **Balance**: ~50/50 split between classes
- [ ] **English Language**: Matches your current system
- [ ] **Temporal Info** (Optional but helpful): Dates, timestamps
- [ ] **Metadata** (Optional): Sources, authors, contexts

---

## 🔧 Integration Steps (Using FEVER as Example)

### Step 1: Install Dependencies
```bash
pip install datasets scikit-learn
```

### Step 2: Create Dataset Loader
```python
# src/data/fever_loader.py (already planned)
from datasets import load_dataset

class FEVERDatasetLoader:
    def load(self, split='train', limit=None):
        dataset = load_dataset('fever', 'v1.0', split=split)
        # Process and return
```

### Step 3: Convert to Graphs
```python
# Your existing pipeline
extractor = EntityExtractor(llm_client)
extraction = extractor.extract_all(claim_text)
graph = GraphBuilder().build_from_extraction(extraction)
pyg_data = GraphEmbedding().networkx_to_pyg(graph.graph)
```

### Step 4: Train GNN
```python
# Your existing train_gnn.py
model = FactVerificationGNN(input_dim=32, output_dim=2)
# Training loop...
```

---

## 📊 Expected Performance

Based on similar GNN fact-checking systems:

| Dataset | Expected Accuracy | Training Time | Notes |
|---------|-------------------|---------------|-------|
| **FEVER** | 75-85% | 2-4 hours | Good baseline |
| **LIAR** | 70-80% | 1-2 hours | Smaller dataset |
| **FakeNewsNet** | 80-90% | 4-6 hours | Requires preprocessing |
| **Custom** | 85-95% | 10-20 hours | Best if well-annotated |

---

## 🎓 Key Takeaways

1. **FEVER is your best starting point** - Already planned, easy integration, perfect fit
2. **Custom dataset is ideal long-term** - But requires significant effort
3. **Your system is ready** - Graph conversion pipeline is complete
4. **Start small** - Train on 1,000 examples first, then scale up
5. **Combine approaches** - Use FEVER for initial training, custom for fine-tuning

---

## 📚 Resources

- **FEVER Dataset**: https://fever.ai/
- **HuggingFace FEVER**: https://huggingface.co/datasets/fever
- **LIAR Dataset**: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
- **FakeNewsNet**: https://github.com/KaiDMML/FakeNewsNet
- **Your Integration Plan**: `DATASET_INTEGRATION_PLAN.md`

---

## ✅ Next Steps

1. **Review this analysis** - Understand your options
2. **Choose FEVER for quick start** - Implement the loader
3. **Train initial model** - Get baseline performance
4. **Evaluate results** - Check if accuracy meets needs
5. **Consider custom dataset** - If you need domain-specific performance

---

**Recommendation**: Start with **FEVER Dataset** - it's the fastest path to a working GNN model that fits your system perfectly! 🚀

