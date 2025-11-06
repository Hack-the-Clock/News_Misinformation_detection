"""
Demo script for News Fact Validation Graph
Demonstrates the complete Neurosymbolic AI pipeline
"""

from datetime import datetime
from src.utils.config import Config
from src.llm.llm_client import LLMClient
from src.extraction.entity_extractor import EntityExtractor
from src.graph.graph_builder import GraphBuilder
from src.reasoning.contradiction_detector import ContradictionDetector
from src.correction.narrative_generator import NarrativeGenerator
from src.visualization.graph_visualizer import GraphVisualizer


def main():
    """Run the complete fact validation pipeline"""

    print("=" * 80)
    print("🔍 News Fact Validation Graph - Neurosymbolic AI Demo")
    print("=" * 80)
    print()

    # Validate configuration
    try:
        Config.validate()
        Config.create_output_dirs()
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("\nPlease:")
        print("1. Copy .env.example to .env")
        print("2. Add your OpenAI API key to .env")
        return

    # COVID example from your requirements
    article_text = """City X reported zero COVID cases yesterday. The same source confirmed a rise in cases earlier that day."""

    reference_date = datetime(2024, 1, 15, 14, 0, 0)  # Jan 15, 2024, 2 PM

    print("📄 Input Article:")
    print("-" * 80)
    print(article_text)
    print("-" * 80)
    print()

    # Step 1: Initialize LLM client
    print("🤖 Step 1: Initializing LLM Client...")
    llm_client = LLMClient()
    print("✓ LLM Client initialized")
    print()

    # Step 2: Extract entities, events, and relationships
    print("🔬 Step 2: Extracting Entities, Events, and Relationships...")
    extractor = EntityExtractor(llm_client)
    extraction_result = extractor.extract_all(article_text, reference_date)

    print(f"✓ Extracted {len(extraction_result['entities'])} entities")
    print(f"✓ Extracted {len(extraction_result['events'])} events")
    print(f"✓ Extracted {len(extraction_result['relations'])} relationships")
    print()

    # Print extracted information
    print("  Entities:")
    for entity in extraction_result['entities']:
        print(f"    - {entity.name} ({entity.entity_type.value})")
    print()

    print("  Events:")
    for event in extraction_result['events']:
        print(f"    - {event.description}")
        print(f"      Time: {event.temporal_expression} → {event.timestamp}")
    print()

    # Step 3: Build temporal knowledge graph
    print("🕸️  Step 3: Building Temporal Knowledge Graph...")
    graph_builder = GraphBuilder()
    knowledge_graph = graph_builder.build_from_extraction(extraction_result)

    stats = knowledge_graph.get_statistics()
    print(f"✓ Graph built with {stats['num_nodes']} nodes and {stats['num_edges']} edges")
    print()

    # Step 4: Apply symbolic logic rules
    print("🧠 Step 4: Applying Symbolic Logic Rules...")
    detector = ContradictionDetector()
    contradictions = detector.detect_contradictions(knowledge_graph)

    print(f"✓ Detected {len(contradictions)} contradictions")
    print()

    if contradictions:
        print("  Contradictions Found:")
        for i, c in enumerate(contradictions, 1):
            event1 = knowledge_graph.graph.nodes.get(c.fact1_id, {})
            event2 = knowledge_graph.graph.nodes.get(c.fact2_id, {})

            print(f"\n  {i}. {c.contradiction_type} (Severity: {c.severity:.2f})")
            print(f"     Fact 1: {event1.get('description', 'Unknown')}")
            print(f"     Fact 2: {event2.get('description', 'Unknown')}")
            print(f"     Explanation: {c.explanation}")
        print()

    # Step 5: Generate corrected narrative
    print("✍️  Step 5: Generating Corrected Narrative...")
    narrative_gen = NarrativeGenerator(llm_client)
    corrected_narrative = narrative_gen.generate_corrected_narrative(
        original_text=article_text,
        contradictions=contradictions,
        graph=knowledge_graph
    )

    print("✓ Corrected narrative generated")
    print()
    print("  Original Text:")
    print(f"  \"{corrected_narrative.original_text}\"")
    print()
    print("  Corrected Text:")
    print(f"  \"{corrected_narrative.corrected_text}\"")
    print()
    print("  Explanation:")
    print(f"  {corrected_narrative.explanation}")
    print()
    print(f"  Confidence Score: {corrected_narrative.confidence_score:.2f}")
    print()

    # Step 6: Generate visualizations
    print("📊 Step 6: Generating Visualizations...")
    visualizer = GraphVisualizer()

    # Interactive graph
    graph_file = visualizer.visualize(
        graph=knowledge_graph,
        contradictions=contradictions,
        output_file="outputs/graphs/knowledge_graph.html"
    )

    # Timeline
    timeline_file = visualizer.create_timeline_visualization(
        graph=knowledge_graph,
        output_file="outputs/graphs/timeline.html"
    )

    # Comprehensive report
    report_file = visualizer.generate_report(
        graph=knowledge_graph,
        contradictions=contradictions,
        corrected_narrative=corrected_narrative,
        output_file="outputs/graphs/report.html"
    )

    print()

    # Summary
    print("=" * 80)
    print("✅ Demo Completed Successfully!")
    print("=" * 80)
    print()
    print("📁 Output Files:")
    print(f"  • Interactive Graph: {graph_file}")
    print(f"  • Timeline View: {timeline_file}")
    print(f"  • Full Report: {report_file}")
    print()
    print("🎯 Neurosymbolic AI Features Demonstrated:")
    print("  ✓ LLM-based entity and temporal extraction")
    print("  ✓ Temporal knowledge graph construction")
    print("  ✓ Symbolic logic rules for contradiction detection")
    print("  ✓ Narrative correction with LLM")
    print("  ✓ Interactive visualization")
    print()
    print("💡 Next Steps:")
    print("  • Open the HTML files in your browser to explore the results")
    print("  • Try the system with different news articles")
    print("  • Add custom logic rules in src/reasoning/logic_rules.py")
    print("  • Train the GNN model for advanced pattern recognition")
    print()


if __name__ == "__main__":
    main()
