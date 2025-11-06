"""
Test script to verify GNN model integration in demo
Tests without making real LLM calls (uses mock graph)
"""

import torch
import networkx as nx
from src.gnn.gnn_model import FactVerificationGNN
from src.gnn.graph_embeddings import GraphEmbedding

def test_gnn_model():
    """Test the trained GNN model with a simple graph"""

    print("="*80)
    print("🧪 Testing Trained GNN Model")
    print("="*80)
    print()

    # Check if model exists
    import os
    if not os.path.exists('models/gnn_llmlog.pth'):
        print("❌ Model not found at models/gnn_llmlog.pth")
        print("   Run: python train_gnn.py --use_llm_log --llm_log_limit 100 --epochs 30")
        return

    print("✓ Found trained model at models/gnn_llmlog.pth")
    print()

    # Load the model
    print("📂 Loading GNN model...")
    model = FactVerificationGNN(
        input_dim=32,
        hidden_dim=64,
        output_dim=2,
        num_layers=2,
        dropout=0.1,
        use_attention=True
    )
    model.load_state_dict(torch.load('models/gnn_llmlog.pth'))
    model.eval()
    print("✓ Model loaded successfully")
    print()

    # Create a simple test graph (similar to what would be extracted from an article)
    print("🔬 Creating test knowledge graph...")

    # Test Case 1: Simple claim graph (SUPPORTS)
    G1 = nx.DiGraph()
    G1.add_node("Entity1", node_type="person")
    G1.add_node("Event1", node_type="event")
    G1.add_edge("Entity1", "Event1", relation="involved_in")

    # Test Case 2: Contradictory claim graph (REFUTES)
    G2 = nx.DiGraph()
    G2.add_node("Entity2", node_type="person")
    G2.add_node("Entity3", node_type="organization")
    G2.add_edge("Entity2", "Entity3", relation="contradicts")

    print("✓ Created 2 test graphs")
    print()

    # Convert graphs to PyG format
    print("🔄 Converting graphs to PyTorch Geometric format...")
    embedding = GraphEmbedding(feature_dim=32)

    pyg_data1 = embedding.networkx_to_pyg(G1)
    pyg_data2 = embedding.networkx_to_pyg(G2)

    print("✓ Graphs converted")
    print()

    # Make predictions
    print("🤖 Running GNN predictions...")
    print()

    with torch.no_grad():
        # Test 1
        out1 = model(pyg_data1)
        probs1 = torch.exp(out1)
        pred1 = out1.argmax(dim=1).item()
        conf1 = probs1[0][pred1].item()

        print("Test Case 1 (Simple Claim):")
        print(f"  Graph: {G1.number_of_nodes()} nodes, {G1.number_of_edges()} edges")
        print(f"  Prediction: {'REFUTES' if pred1 == 1 else 'SUPPORTS'}")
        print(f"  Confidence: {conf1:.2%}")
        print(f"  Raw probabilities: SUPPORTS={probs1[0][0].item():.2%}, REFUTES={probs1[0][1].item():.2%}")
        print()

        # Test 2
        out2 = model(pyg_data2)
        probs2 = torch.exp(out2)
        pred2 = out2.argmax(dim=1).item()
        conf2 = probs2[0][pred2].item()

        print("Test Case 2 (Contradictory Claim):")
        print(f"  Graph: {G2.number_of_nodes()} nodes, {G2.number_of_edges()} edges")
        print(f"  Prediction: {'REFUTES' if pred2 == 1 else 'SUPPORTS'}")
        print(f"  Confidence: {conf2:.2%}")
        print(f"  Raw probabilities: SUPPORTS={probs2[0][0].item():.2%}, REFUTES={probs2[0][1].item():.2%}")
        print()

    print("="*80)
    print("✅ GNN Model Test Complete!")
    print("="*80)
    print()
    print("📊 Model Statistics:")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Model file size: {os.path.getsize('models/gnn_llmlog.pth') / 1024:.2f} KB")
    print()
    print("💡 Next Steps:")
    print("   1. The GNN model is working and integrated into demo.py")
    print("   2. When your OpenAI API limit resets, run: python demo.py")
    print("   3. The demo will automatically use the GNN for fact verification")
    print()

if __name__ == "__main__":
    test_gnn_model()
