"""
Test demo.py on FEVER dataset examples
Evaluates GNN predictions vs ground truth labels
"""

import torch
import time
from src.llm.llm_client import LLMClient
from src.extraction.entity_extractor import EntityExtractor
from src.graph.graph_builder import GraphBuilder
from src.gnn.gnn_model import FactVerificationGNN
from src.gnn.graph_embeddings import GraphEmbedding
from src.data.fever_loader import FEVERDatasetLoader
from src.utils.config import Config
import os
from datetime import datetime

def test_on_fever_samples(num_samples=10):
    """
    Test the complete pipeline on FEVER dataset samples

    Args:
        num_samples: Number of FEVER samples to test
    """
    print("="*80)
    print("🧪 Testing Demo Pipeline on FEVER Dataset")
    print("="*80)
    print()

    # Validate configuration
    try:
        Config.validate()
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        return

    # Check if GNN model exists
    if not os.path.exists('models/gnn_llmlog.pth'):
        print("❌ GNN model not found at models/gnn_llmlog.pth")
        print("   Run: python train_gnn.py --use_llm_log --llm_log_limit 100 --epochs 30")
        return

    print("✓ Found GNN model")
    print()

    # Load FEVER dataset
    print(f"📂 Loading {num_samples} FEVER samples...")
    loader = FEVERDatasetLoader()
    samples = loader.get_balanced_dataset(n_samples=num_samples, split='train')
    print(f"✓ Loaded {len(samples)} samples")
    print()

    # Initialize components
    print("🤖 Initializing components...")
    llm_client = LLMClient()
    extractor = EntityExtractor(llm_client)
    graph_builder = GraphBuilder()
    embedding = GraphEmbedding(feature_dim=32)

    # Load GNN model
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
    print("✓ Components initialized")
    print()

    # Test each sample
    print("🔬 Testing samples...")
    print()

    results = []
    successful = 0
    failed = 0

    for i, sample in enumerate(samples, 1):
        print(f"Sample {i}/{len(samples)}: {sample['claim'][:60]}...")

        try:
            # Extract entities and build graph
            extraction = extractor.extract_all(sample['text'], datetime.now())
            graph = graph_builder.build_from_extraction(extraction)

            # Convert to PyG and get GNN prediction
            pyg_data = embedding.networkx_to_pyg(graph.graph)

            with torch.no_grad():
                out = model(pyg_data)
                probabilities = torch.exp(out)
                gnn_pred = out.argmax(dim=1).item()
                gnn_conf = probabilities[0][gnn_pred].item()

            # Ground truth
            true_label = sample['label_numeric']

            # Check if correct
            correct = (gnn_pred == true_label)

            result = {
                'text': sample['claim'],
                'true_label': 'REFUTES' if true_label == 1 else 'SUPPORTS',
                'gnn_pred': 'REFUTES' if gnn_pred == 1 else 'SUPPORTS',
                'confidence': gnn_conf,
                'correct': correct,
                'num_nodes': graph.graph.number_of_nodes(),
                'num_edges': graph.graph.number_of_edges()
            }
            results.append(result)
            successful += 1

            status = "✓" if correct else "✗"
            print(f"  {status} True: {result['true_label']}, Predicted: {result['gnn_pred']} ({gnn_conf:.1%})")
            print(f"  Graph: {result['num_nodes']} nodes, {result['num_edges']} edges")

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed += 1

        print()

        # Add 30 second delay between samples to avoid rate limits
        if i < len(samples):
            print(f"⏳ Waiting 30 seconds before next sample to avoid rate limits...")
            time.sleep(30)
            print()

    # Calculate metrics
    print("="*80)
    print("📊 Results Summary")
    print("="*80)
    print()

    if results:
        correct_predictions = sum(1 for r in results if r['correct'])
        accuracy = correct_predictions / len(results)

        print(f"Total Samples: {len(samples)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print()
        print(f"Correct Predictions: {correct_predictions}/{len(results)}")
        print(f"Accuracy: {accuracy:.2%}")
        print()

        # Average confidence
        avg_conf = sum(r['confidence'] for r in results) / len(results)
        print(f"Average Confidence: {avg_conf:.2%}")
        print()

        # By label
        supports_results = [r for r in results if r['true_label'] == 'SUPPORTS']
        refutes_results = [r for r in results if r['true_label'] == 'REFUTES']

        if supports_results:
            supports_acc = sum(1 for r in supports_results if r['correct']) / len(supports_results)
            print(f"SUPPORTS Accuracy: {supports_acc:.2%} ({len(supports_results)} samples)")

        if refutes_results:
            refutes_acc = sum(1 for r in refutes_results if r['correct']) / len(refutes_results)
            print(f"REFUTES Accuracy: {refutes_acc:.2%} ({len(refutes_results)} samples)")

        print()

        # Graph statistics
        avg_nodes = sum(r['num_nodes'] for r in results) / len(results)
        avg_edges = sum(r['num_edges'] for r in results) / len(results)
        print(f"Average Graph Size: {avg_nodes:.1f} nodes, {avg_edges:.1f} edges")
        print()

        # Show some examples
        print("="*80)
        print("📋 Sample Results")
        print("="*80)
        print()

        for i, r in enumerate(results[:5], 1):
            status = "✓ CORRECT" if r['correct'] else "✗ INCORRECT"
            print(f"{i}. {status}")
            print(f"   Text: {r['text'][:70]}...")
            print(f"   True: {r['true_label']}, Predicted: {r['gnn_pred']} ({r['confidence']:.1%})")
            print(f"   Graph: {r['num_nodes']} nodes, {r['num_edges']} edges")
            print()

    else:
        print("❌ No successful predictions")

    print("="*80)
    print("✅ Test Complete!")
    print("="*80)
    print()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test demo on FEVER dataset')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of FEVER samples to test (default: 10)')

    args = parser.parse_args()

    test_on_fever_samples(num_samples=args.num_samples)
