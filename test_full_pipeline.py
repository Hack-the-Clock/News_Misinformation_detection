"""
Test complete neurosymbolic pipeline on FEVER dataset
Evaluates both GNN predictions AND symbolic logic rules
"""

import torch
import time
from src.llm.llm_client import LLMClient
from src.extraction.entity_extractor import EntityExtractor
from src.graph.graph_builder import GraphBuilder
from src.gnn.gnn_model import FactVerificationGNN
from src.gnn.graph_embeddings import GraphEmbedding
from src.reasoning.contradiction_detector import ContradictionDetector
from src.data.fever_loader import FEVERDatasetLoader
from src.utils.config import Config
import os
from datetime import datetime

def test_full_pipeline(num_samples=5):
    """
    Test the COMPLETE neurosymbolic pipeline on FEVER dataset

    Tests:
    1. LLM extraction
    2. Knowledge graph building
    3. GNN predictions
    4. Symbolic logic contradiction detection
    5. Combined analysis

    Args:
        num_samples: Number of FEVER samples to test
    """
    print("="*80)
    print("🔬 COMPLETE NEUROSYMBOLIC PIPELINE TEST")
    print("="*80)
    print()
    print("Testing Components:")
    print("  ✓ LLM Entity/Event Extraction")
    print("  ✓ Knowledge Graph Construction")
    print("  ✓ GNN Fact Verification")
    print("  ✓ Symbolic Logic Rules")
    print("  ✓ Combined Analysis")
    print()
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
    print("🤖 Initializing pipeline components...")
    llm_client = LLMClient()
    extractor = EntityExtractor(llm_client)
    graph_builder = GraphBuilder()
    embedding = GraphEmbedding(feature_dim=32)
    detector = ContradictionDetector()

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
    print("✓ All components initialized")
    print()

    # Test each sample
    print("="*80)
    print("🔬 TESTING SAMPLES")
    print("="*80)
    print()

    results = []
    successful = 0
    failed = 0

    for i, sample in enumerate(samples, 1):
        print(f"\n{'='*80}")
        print(f"SAMPLE {i}/{len(samples)}")
        print('='*80)
        print(f"Claim: {sample['claim']}")
        print(f"Ground Truth: {sample['original_label']}")
        print()

        try:
            # Step 1: LLM Extraction
            print("Step 1: LLM Extraction...")
            extraction = extractor.extract_all(sample['text'], datetime.now())
            print(f"  ✓ Extracted: {len(extraction['entities'])} entities, "
                  f"{len(extraction['events'])} events, {len(extraction['relations'])} relations")

            # Step 2: Build Graph
            print("Step 2: Building Knowledge Graph...")
            graph = graph_builder.build_from_extraction(extraction)
            num_nodes = graph.graph.number_of_nodes()
            num_edges = graph.graph.number_of_edges()
            print(f"  ✓ Graph: {num_nodes} nodes, {num_edges} edges")

            # Step 3: GNN Prediction
            print("Step 3: GNN Analysis...")

            # Check if graph has edges (GNN needs edges to work)
            if num_edges == 0:
                print(f"  ⚠️  Warning: Graph has no edges, GNN cannot analyze")
                print(f"     This usually means LLM didn't extract relations")
                print(f"     Skipping GNN analysis for this sample")
                gnn_pred = None
                gnn_label = "N/A"
                gnn_correct = False
                gnn_conf = 0.0
            else:
                pyg_data = embedding.networkx_to_pyg(graph.graph)
                with torch.no_grad():
                    out = model(pyg_data)
                    probabilities = torch.exp(out)
                    gnn_pred = out.argmax(dim=1).item()
                    gnn_conf = probabilities[0][gnn_pred].item()

                gnn_label = 'REFUTES' if gnn_pred == 1 else 'SUPPORTS'
                gnn_correct = (gnn_pred == sample['label_numeric'])
                print(f"  ✓ GNN Prediction: {gnn_label} ({gnn_conf:.1%} confidence)")
                print(f"    {'✓ CORRECT' if gnn_correct else '✗ INCORRECT'}")

            # Step 4: Symbolic Logic
            print("Step 4: Symbolic Logic Rules...")
            contradictions = detector.detect_contradictions(graph)
            print(f"  ✓ Detected: {len(contradictions)} contradictions")

            if contradictions:
                for j, c in enumerate(contradictions[:3], 1):  # Show first 3
                    print(f"    {j}. {c.contradiction_type} (severity: {c.severity:.2f})")

            # Determine symbolic logic prediction
            # If contradictions found → REFUTES, else → SUPPORTS
            logic_pred = 1 if len(contradictions) > 0 else 0
            logic_label = 'REFUTES' if logic_pred == 1 else 'SUPPORTS'
            logic_correct = (logic_pred == sample['label_numeric'])
            print(f"  Logic Prediction: {logic_label}")
            print(f"    {'✓ CORRECT' if logic_correct else '✗ INCORRECT'}")

            # Step 5: Combined Decision
            print("Step 5: Combined Analysis...")
            # Simple voting: both must agree for high confidence
            if gnn_pred == logic_pred:
                combined_pred = gnn_pred
                combined_conf = "HIGH"
            else:
                # Disagreement - use GNN with lower confidence
                combined_pred = gnn_pred
                combined_conf = "LOW"

            combined_label = 'REFUTES' if combined_pred == 1 else 'SUPPORTS'
            combined_correct = (combined_pred == sample['label_numeric'])

            agreement = "AGREE" if gnn_pred == logic_pred else "DISAGREE"
            print(f"  GNN vs Logic: {agreement}")
            print(f"  Combined Prediction: {combined_label} (Confidence: {combined_conf})")
            print(f"    {'✓ CORRECT' if combined_correct else '✗ INCORRECT'}")

            # Store result
            result = {
                'claim': sample['claim'],
                'true_label': sample['original_label'],
                'true_numeric': sample['label_numeric'],
                'gnn_pred': gnn_label,
                'gnn_correct': gnn_correct,
                'gnn_conf': gnn_conf,
                'logic_pred': logic_label,
                'logic_correct': logic_correct,
                'num_contradictions': len(contradictions),
                'combined_pred': combined_label,
                'combined_correct': combined_correct,
                'agreement': agreement,
                'num_nodes': num_nodes,
                'num_edges': num_edges
            }
            results.append(result)
            successful += 1

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed += 1

        print()

        # Wait between samples
        if i < len(samples):
            print(f"⏳ Waiting 30 seconds before next sample...")
            time.sleep(30)

    # Calculate comprehensive metrics
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE RESULTS")
    print("="*80)
    print()

    if not results:
        print("❌ No successful tests")
        return

    # Overall metrics
    print(f"Total Samples: {len(samples)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print()

    # GNN Performance
    gnn_results = [r for r in results if r['gnn_pred'] != 'N/A']
    if gnn_results:
        gnn_correct = sum(1 for r in gnn_results if r['gnn_correct'])
        gnn_accuracy = gnn_correct / len(gnn_results)
        print("🤖 GNN Performance:")
        print(f"  Samples analyzed: {len(gnn_results)}/{len(results)}")
        print(f"  Accuracy: {gnn_accuracy:.1%} ({gnn_correct}/{len(gnn_results)})")
        print(f"  Avg Confidence: {sum(r['gnn_conf'] for r in gnn_results)/len(gnn_results):.1%}")
    else:
        print("🤖 GNN Performance:")
        print(f"  ⚠️  No graphs had edges - GNN could not analyze any samples")
        print(f"     (LLM extractions didn't include relations)")

    # Symbolic Logic Performance
    logic_correct = sum(1 for r in results if r['logic_correct'])
    logic_accuracy = logic_correct / len(results)
    print()
    print("🧠 Symbolic Logic Performance:")
    print(f"  Accuracy: {logic_accuracy:.1%} ({logic_correct}/{len(results)})")
    print(f"  Avg Contradictions: {sum(r['num_contradictions'] for r in results)/len(results):.1f}")

    # Combined Performance
    combined_correct = sum(1 for r in results if r['combined_correct'])
    combined_accuracy = combined_correct / len(results)
    print()
    print("🔮 Combined Performance:")
    print(f"  Accuracy: {combined_accuracy:.1%} ({combined_correct}/{len(results)})")

    # Agreement analysis
    agreements = sum(1 for r in results if r['agreement'] == 'AGREE')
    agreement_rate = agreements / len(results)
    print(f"  Agreement Rate: {agreement_rate:.1%} ({agreements}/{len(results)})")

    # When they agree, what's the accuracy?
    agree_results = [r for r in results if r['agreement'] == 'AGREE']
    if agree_results:
        agree_correct = sum(1 for r in agree_results if r['combined_correct'])
        agree_accuracy = agree_correct / len(agree_results)
        print(f"  Accuracy when agree: {agree_accuracy:.1%} ({agree_correct}/{len(agree_results)})")

    print()

    # By label breakdown
    print("📋 Performance by Label:")
    for label in ['SUPPORTS', 'REFUTES']:
        label_results = [r for r in results if r['true_label'] == label]
        if label_results:
            label_gnn_results = [r for r in label_results if r['gnn_pred'] != 'N/A']
            if label_gnn_results:
                gnn_acc = sum(1 for r in label_gnn_results if r['gnn_correct']) / len(label_gnn_results)
            else:
                gnn_acc = 0.0
            logic_acc = sum(1 for r in label_results if r['logic_correct']) / len(label_results)
            combined_acc = sum(1 for r in label_results if r['combined_correct']) / len(label_results)
            print(f"  {label}:")
            print(f"    GNN: {gnn_acc:.1%}, Logic: {logic_acc:.1%}, Combined: {combined_acc:.1%}")

    print()

    # Detailed examples
    print("="*80)
    print("📋 DETAILED EXAMPLES")
    print("="*80)
    print()

    for i, r in enumerate(results[:3], 1):  # Show first 3
        print(f"{i}. Claim: {r['claim'][:70]}...")
        print(f"   True: {r['true_label']}")
        print(f"   GNN: {r['gnn_pred']} ({r['gnn_conf']:.1%}) {'✓' if r['gnn_correct'] else '✗'}")
        print(f"   Logic: {r['logic_pred']} ({r['num_contradictions']} contradictions) {'✓' if r['logic_correct'] else '✗'}")
        print(f"   Combined: {r['combined_pred']} {'✓' if r['combined_correct'] else '✗'}")
        print(f"   Graph: {r['num_nodes']} nodes, {r['num_edges']} edges")
        print()

    print("="*80)
    print("✅ COMPLETE PIPELINE TEST FINISHED!")
    print("="*80)
    print()

    # Summary insights
    print("💡 Key Insights:")
    if gnn_results:
        if gnn_accuracy > logic_accuracy:
            print(f"  • GNN outperforms symbolic logic (+{(gnn_accuracy-logic_accuracy)*100:.1f}%)")
        elif logic_accuracy > gnn_accuracy:
            print(f"  • Symbolic logic outperforms GNN (+{(logic_accuracy-gnn_accuracy)*100:.1f}%)")
        else:
            print(f"  • GNN and symbolic logic perform equally")
    else:
        print(f"  • GNN could not analyze samples (no edges in graphs)")
        print(f"  • Only symbolic logic results available")

    if gnn_results and combined_accuracy > max(gnn_accuracy, logic_accuracy):
        print(f"  • Combined approach improves accuracy! 🎉")

    if agreement_rate > 0.7:
        print(f"  • High agreement between methods ({agreement_rate:.1%})")
    else:
        print(f"  • Methods often disagree ({agreement_rate:.1%} agreement)")

    print()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test complete neurosymbolic pipeline')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of FEVER samples to test (default: 5)')

    args = parser.parse_args()

    test_full_pipeline(num_samples=args.num_samples)
