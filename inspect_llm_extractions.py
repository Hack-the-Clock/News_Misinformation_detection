"""
Script to inspect LLM extraction quality from logs
"""
import json
import networkx as nx
from src.data.llm_log_graph_loader import LLMLogGraphLoader
from src.gnn.graph_embeddings import GraphEmbedding
import matplotlib.pyplot as plt

def analyze_llm_log():
    """Analyze the quality of LLM extractions"""

    print("="*80)
    print("LLM EXTRACTION QUALITY ANALYSIS")
    print("="*80)
    print()

    # Load and parse log file
    log_path = 'logs/llm_calls.log'
    extractions = []
    successful = 0
    failed = 0

    print("📂 Loading LLM extraction logs...")
    with open(log_path, 'r') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            entry = json.loads(line)
            if 'error' in entry['response']:
                failed += 1
            else:
                successful += 1
                extractions.append({
                    'text': entry['payload']['text'],
                    'response': entry['response'],
                    'timestamp': entry['timestamp']
                })

    print(f"✓ Found {successful} successful extractions, {failed} failed")
    print()

    if not extractions:
        print("❌ No successful extractions to analyze!")
        return

    # Analyze each extraction
    print("="*80)
    print("DETAILED EXTRACTION ANALYSIS")
    print("="*80)
    print()

    for i, ext in enumerate(extractions, 1):
        print(f"\n{'='*80}")
        print(f"EXTRACTION #{i}")
        print('='*80)
        print(f"\n📝 Input Text:")
        print(f"   {ext['text']}")
        print()

        response = ext['response']
        entities = response.get('entities', [])
        events = response.get('events', [])
        relations = response.get('relations', [])

        print(f"🎯 Extracted Entities: {len(entities)}")
        for ent in entities:
            print(f"   - {ent['name']} ({ent['type']})")
            if ent.get('properties'):
                print(f"     Properties: {ent['properties']}")

        print()
        print(f"📅 Extracted Events: {len(events)}")
        for evt in events:
            print(f"   - {evt['description']}")
            if evt.get('temporal_expression'):
                print(f"     Time: {evt['temporal_expression']}")
            if evt.get('entities_involved'):
                print(f"     Entities: {', '.join(evt['entities_involved'])}")

        print()
        print(f"🔗 Extracted Relations: {len(relations)}")
        for rel in relations:
            print(f"   - {rel['source']} --[{rel['relation']}]--> {rel['target']}")
            if rel.get('temporal_expression'):
                print(f"     Time: {rel['temporal_expression']}")

        print()
        print(f"📊 Stats:")
        print(f"   Total nodes: {len(entities) + len(events)}")
        print(f"   Total edges: {len(relations) + sum(len(e.get('entities_involved', [])) for e in events)}")

    # Build graph from logs
    print()
    print("="*80)
    print("GRAPH CONSTRUCTION ANALYSIS")
    print("="*80)
    print()

    loader = LLMLogGraphLoader()
    G = loader.load_graph(max_datapoints=10)

    print(f"📊 Complete Graph Statistics:")
    print(f"   Total nodes: {G.number_of_nodes()}")
    print(f"   Total edges: {G.number_of_edges()}")
    print()

    if G.number_of_nodes() > 0:
        print(f"🔍 Node Types:")
        node_types = {}
        for node, data in G.nodes(data=True):
            node_type = data.get('node_type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        for ntype, count in node_types.items():
            print(f"   {ntype}: {count}")

        print()
        print(f"🔗 Edge Types:")
        edge_types = {}
        for u, v, data in G.edges(data=True):
            edge_type = data.get('relation', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        for etype, count in edge_types.items():
            print(f"   {etype}: {count}")

        print()
        print(f"📈 Graph Connectivity:")
        if G.number_of_nodes() > 0:
            # Convert to undirected for connectivity analysis
            G_undirected = G.to_undirected()
            num_components = nx.number_connected_components(G_undirected)
            print(f"   Connected components: {num_components}")

            if num_components > 0:
                component_sizes = [len(c) for c in nx.connected_components(G_undirected)]
                print(f"   Largest component size: {max(component_sizes)}")
                print(f"   Smallest component size: {min(component_sizes)}")
                print(f"   Average component size: {sum(component_sizes)/len(component_sizes):.2f}")

        print()
        print(f"📋 Sample Nodes:")
        for i, (node, data) in enumerate(list(G.nodes(data=True))[:5]):
            print(f"   {i+1}. {node} (type: {data.get('node_type', 'unknown')})")

        if G.number_of_edges() > 0:
            print()
            print(f"📋 Sample Edges:")
            for i, (u, v, data) in enumerate(list(G.edges(data=True))[:5]):
                relation = data.get('relation', 'unknown')
                print(f"   {i+1}. {u} --[{relation}]--> {v}")

    # PyG conversion test
    print()
    print("="*80)
    print("PYTORCH GEOMETRIC CONVERSION TEST")
    print("="*80)
    print()

    if G.number_of_nodes() > 0:
        embedding = GraphEmbedding(feature_dim=32)

        # Get connected components
        G_undirected = G.to_undirected()
        components = list(nx.connected_components(G_undirected))

        print(f"🔄 Converting {len(components)} connected components to PyG format...")
        print()

        valid_graphs = 0
        for i, component in enumerate(components[:3], 1):  # Show first 3
            subgraph = G.subgraph(component).copy()
            try:
                pyg_data = embedding.networkx_to_pyg(subgraph)
                valid_graphs += 1

                print(f"Component #{i}:")
                print(f"   Nodes: {pyg_data.num_nodes}")
                print(f"   Edges: {pyg_data.num_edges}")
                print(f"   Node features shape: {pyg_data.x.shape}")
                if hasattr(pyg_data, 'edge_attr') and pyg_data.edge_attr is not None:
                    print(f"   Edge features shape: {pyg_data.edge_attr.shape}")
                print()
            except Exception as e:
                print(f"Component #{i}: ❌ Failed to convert - {e}")
                print()

        print(f"✓ Successfully converted {valid_graphs}/{len(components)} components to PyG format")
    else:
        print("❌ No nodes in graph - cannot test PyG conversion")

    # Assessment
    print()
    print("="*80)
    print("QUALITY ASSESSMENT")
    print("="*80)
    print()

    if not extractions:
        print("❌ POOR: No successful extractions")
        return

    avg_entities = sum(len(e['response'].get('entities', [])) for e in extractions) / len(extractions)
    avg_events = sum(len(e['response'].get('events', [])) for e in extractions) / len(extractions)
    avg_relations = sum(len(e['response'].get('relations', [])) for e in extractions) / len(extractions)

    print(f"📊 Average per extraction:")
    print(f"   Entities: {avg_entities:.2f}")
    print(f"   Events: {avg_events:.2f}")
    print(f"   Relations: {avg_relations:.2f}")
    print()

    # Quality scoring
    score = 0
    issues = []

    if avg_entities >= 2:
        score += 1
        print("✓ Good: Extracting multiple entities per text")
    else:
        issues.append("Low entity extraction (avg < 2)")

    if avg_relations >= 1:
        score += 1
        print("✓ Good: Extracting relationships")
    else:
        issues.append("Few relationships extracted (avg < 1)")

    if G.number_of_nodes() >= 2:
        score += 1
        print("✓ Good: Building connected graph")
    else:
        issues.append("Graph too small")

    if G.number_of_edges() >= 1:
        score += 1
        print("✓ Good: Graph has edges/connections")
    else:
        issues.append("Graph has no edges")

    print()
    print(f"Overall Score: {score}/4")
    print()

    if issues:
        print("⚠️  Issues detected:")
        for issue in issues:
            print(f"   - {issue}")
        print()

    if score >= 3:
        print("✅ RECOMMENDATION: Data quality is GOOD - proceed with 100 samples")
    elif score >= 2:
        print("⚠️  RECOMMENDATION: Data quality is FAIR - may need prompt improvements")
    else:
        print("❌ RECOMMENDATION: Data quality is POOR - need better extraction prompts")

    print()
    print("="*80)

if __name__ == "__main__":
    analyze_llm_log()
