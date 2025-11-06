# System Architecture

## Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                          INPUT ARTICLE                               │
│  "City X reported zero COVID cases yesterday.                        │
│   The same source confirmed a rise in cases earlier that day."       │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: LLM EXTRACTION                           │
│                         (OpenAI GPT-4)                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │   ENTITIES   │  │    EVENTS    │  │  RELATIONS   │              │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤              │
│  │ • City X     │  │ • zero cases │  │ • reported   │              │
│  │ • Source     │  │ • rise       │  │ • confirmed  │              │
│  │              │  │ • yesterday  │  │ • same_source│              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│                                                                       │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│              PHASE 2: TEMPORAL KNOWLEDGE GRAPH                       │
│                        (NetworkX)                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│       ┌─────────┐        reported        ┌─────────┐                │
│       │ City X  │◄──────────────────────│ Event1  │                │
│       │(entity) │                        │"zero    │                │
│       └────┬────┘                        │cases"   │                │
│            │                             │14:00    │                │
│            │                             └─────────┘                │
│            │ involves                         │                     │
│            │                             contradicts                │
│            │                                  │                     │
│            │                                  ▼                     │
│            │                             ┌─────────┐                │
│            └────────────────────────────►│ Event2  │                │
│                     involves             │"rise in │                │
│                                          │cases"   │                │
│       ┌─────────┐  same_source          │08:00    │                │
│       │ Source  ├─────────────────────► └─────────┘                │
│       │(entity) │                                                   │
│       └─────────┘                                                   │
│                                                                       │
│  Nodes: 4 (2 entities, 2 events)                                    │
│  Edges: 5 (relationships + temporal links)                           │
│                                                                       │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│          PHASE 3: SYMBOLIC LOGIC REASONING                           │
│                    (Rule-Based Engine)                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Rule 1: TEMPORAL CONSISTENCY                                        │
│  ├─ Check: Events on same day?         ✓ YES (Jan 15, 2024)        │
│  ├─ Check: Contradictory content?      ✓ YES ("zero" vs "rise")    │
│  └─ Result: VIOLATION (severity: 0.9)                               │
│                                                                       │
│  Rule 2: MUTUAL EXCLUSIVITY                                          │
│  ├─ Check: "zero" + "rise" compatible? ✗ NO                         │
│  └─ Result: VIOLATION (severity: 1.0)                               │
│                                                                       │
│  Rule 3: SOURCE CONSISTENCY                                          │
│  ├─ Check: Same source?                ✓ YES                        │
│  ├─- Check: Contradictory statements?  ✓ YES                        │
│  └─ Result: VIOLATION (severity: 0.95)                              │
│                                                                       │
│  Rule 4: EVENT ORDERING                                              │
│  ├─ Check: Temporal order valid?       ✓ YES (08:00 < 14:00)       │
│  └─ Result: NO VIOLATION                                             │
│                                                                       │
│  ╔═══════════════════════════════════════════════════════════╗      │
│  ║  CONTRADICTIONS DETECTED: 3                               ║      │
│  ║  Highest Severity: 1.0 (CRITICAL)                         ║      │
│  ╚═══════════════════════════════════════════════════════════╝      │
│                                                                       │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│               PHASE 4: GNN ANALYSIS (Optional)                       │
│                  (PyTorch Geometric)                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Graph → Node Features → GNN Layers → Embeddings                    │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │  Graph Attention Network (GAT)                          │        │
│  │  • Learn contradiction patterns                          │        │
│  │  • Detect anomalous graph structures                     │        │
│  │  • Score node credibility                                │        │
│  └─────────────────────────────────────────────────────────┘        │
│                                                                       │
│  Neurosymbolic Fusion:                                               │
│  Final_Score = 0.5 × Symbolic_Score + 0.5 × GNN_Score               │
│                                                                       │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│            PHASE 5: NARRATIVE CORRECTION                             │
│                    (LLM Generation)                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Input to LLM:                                                       │
│  • Original text                                                     │
│  • Detected contradictions with explanations                         │
│  • Timeline of events                                                │
│  • Confidence scores                                                 │
│                                                                       │
│  LLM Task:                                                           │
│  • Remove contradictory statements                                   │
│  • Retain higher-confidence facts                                    │
│  • Generate coherent corrected narrative                             │
│  • Provide explanation of changes                                    │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │  CORRECTED NARRATIVE                                     │        │
│  │  "City X confirmed a rise in cases earlier that day."   │        │
│  │                                                          │        │
│  │  EXPLANATION                                             │        │
│  │  The statement about "zero cases" contradicts the       │        │
│  │  earlier report of rising cases. The chronologically    │        │
│  │  earlier statement is retained as more reliable.        │        │
│  │                                                          │        │
│  │  CONFIDENCE: 0.92                                        │        │
│  └─────────────────────────────────────────────────────────┘        │
│                                                                       │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│               PHASE 6: VISUALIZATION & REPORTING                     │
│                      (PyVis, HTML)                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Output Files:                                                       │
│  ┌────────────────────────────────────────────┐                     │
│  │  1. knowledge_graph.html                   │                     │
│  │     • Interactive graph with nodes/edges   │                     │
│  │     • Red highlights for contradictions    │                     │
│  │     • Hover for details                     │                     │
│  └────────────────────────────────────────────┘                     │
│                                                                       │
│  ┌────────────────────────────────────────────┐                     │
│  │  2. timeline.html                          │                     │
│  │     • Chronological event view              │                     │
│  │     • Time-ordered facts                    │                     │
│  └────────────────────────────────────────────┘                     │
│                                                                       │
│  ┌────────────────────────────────────────────┐                     │
│  │  3. report.html                            │                     │
│  │     • Complete analysis report              │                     │
│  │     • Statistics                            │                     │
│  │     • Contradictions list                   │                     │
│  │     • Original vs Corrected text            │                     │
│  └────────────────────────────────────────────┘                     │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Module Interactions

### Data Flow

```
Article Text
    │
    ├─→ LLM Client (llm/llm_client.py)
    │       │
    │       ├─→ Entity Extractor (extraction/entity_extractor.py)
    │       │       │
    │       │       └─→ Entities, Events, Relations
    │       │
    │       └─→ Temporal Parser (utils/time_utils.py)
    │               │
    │               └─→ Normalized Timestamps
    │
    ├─→ Graph Builder (graph/graph_builder.py)
    │       │
    │       └─→ Temporal Knowledge Graph (graph/temporal_graph.py)
    │               │
    │               └─→ NetworkX MultiDiGraph
    │
    ├─→ Contradiction Detector (reasoning/contradiction_detector.py)
    │       │
    │       ├─→ Logic Rules (reasoning/logic_rules.py)
    │       │       │
    │       │       └─→ Contradiction List
    │       │
    │       └─→ GNN Model (gnn/gnn_model.py) [Optional]
    │               │
    │               ├─→ Graph Embeddings (gnn/graph_embeddings.py)
    │               │
    │               └─→ Credibility Scores
    │
    ├─→ Narrative Generator (correction/narrative_generator.py)
    │       │
    │       └─→ Corrected Narrative Object
    │
    └─→ Graph Visualizer (visualization/graph_visualizer.py)
            │
            ├─→ Interactive HTML Graph
            ├─→ Timeline View
            └─→ Full Report
```

---

## Key Design Patterns

### 1. Separation of Concerns
- **Extraction**: Pure LLM-based extraction
- **Representation**: Graph-based storage
- **Reasoning**: Symbolic logic rules
- **Learning**: GNN pattern recognition
- **Generation**: LLM-based correction

### 2. Modular Architecture
Each component can be:
- Tested independently
- Swapped with alternatives
- Extended with new features

### 3. Neurosymbolic Integration
```
Symbolic Rules (Interpretable) + Neural Networks (Learnable)
                    ↓
            Hybrid Reasoning System
                    ↓
        Better than either alone
```

---

## Data Models

### Entity
```python
{
    id: "entity_abc123",
    name: "City X",
    entity_type: EntityType.LOCATION,
    properties: {},
    confidence: 0.95
}
```

### Event
```python
{
    id: "event_def456",
    description: "reported zero COVID cases",
    timestamp: datetime(2024, 1, 15, 14, 0),
    temporal_expression: "yesterday",
    entities_involved: ["City X", "Source"],
    confidence: 0.90
}
```

### Relation
```python
{
    id: "relation_ghi789",
    source_id: "event_def456",
    target_id: "entity_abc123",
    relation_type: RelationType.REPORTED,
    timestamp: datetime(2024, 1, 15, 14, 0),
    confidence: 0.85
}
```

### Contradiction
```python
{
    id: "contradiction_jkl012",
    fact1_id: "event_def456",
    fact2_id: "event_mno345",
    contradiction_type: "Mutual Exclusivity",
    severity: 1.0,
    explanation: "Cannot have zero and rise simultaneously",
    temporal_conflict: True,
    semantic_conflict: True
}
```

---

## Extensibility Points

### 1. Add New Logic Rules
Extend `LogicRule` class in `src/reasoning/logic_rules.py`

### 2. Custom Entity Types
Add to `EntityType` enum in `src/models/fact.py`

### 3. Alternative LLMs
Implement new client in `src/llm/`

### 4. Graph Storage Backend
Create Neo4j adapter implementing same interface as `TemporalKnowledgeGraph`

### 5. Custom Visualizations
Add methods to `GraphVisualizer` class

---

## Performance Considerations

### Time Complexity
- **Extraction**: O(1) per article (API call)
- **Graph Building**: O(n + m) where n=nodes, m=edges
- **Contradiction Detection**: O(n²) for pairwise comparison
- **GNN**: O(n × d × L) where d=hidden_dim, L=num_layers
- **Visualization**: O(n + m)

### Space Complexity
- **Graph Storage**: O(n + m)
- **GNN Features**: O(n × feature_dim)
- **Embeddings**: O(n × hidden_dim)

### Optimization Strategies
1. **Batch Processing**: Process multiple articles in parallel
2. **Caching**: Cache LLM responses for identical queries
3. **Pruning**: Remove low-confidence nodes/edges
4. **Incremental Updates**: Update graph without rebuilding

---

## Future Architecture Enhancements

### 1. Microservices Architecture
```
API Gateway → [Extraction Service]
           → [Graph Service]
           → [Reasoning Service]
           → [Visualization Service]
```

### 2. Real-time Processing
- Kafka/RabbitMQ for streaming
- Redis for caching
- WebSocket for live updates

### 3. Scalability
- Neo4j for large-scale graphs
- Distributed GNN training
- Cloud deployment (AWS/GCP)

### 4. Multi-modal
- Image analysis integration
- Video fact-checking
- Audio transcription + analysis
