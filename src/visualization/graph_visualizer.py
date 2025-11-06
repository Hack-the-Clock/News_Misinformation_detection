"""Visualize temporal knowledge graphs"""

from pyvis.network import Network
import networkx as nx
from typing import List, Optional
import os

from ..models.fact import Contradiction
from ..graph.temporal_graph import TemporalKnowledgeGraph


class GraphVisualizer:
    """Visualize temporal knowledge graphs with contradiction highlighting"""

    def __init__(self, width: str = "100%", height: str = "800px"):
        """
        Initialize visualizer

        Args:
            width: Canvas width
            height: Canvas height
        """
        self.width = width
        self.height = height

    def visualize(
        self,
        graph: TemporalKnowledgeGraph,
        contradictions: Optional[List[Contradiction]] = None,
        output_file: str = "graph.html",
        notebook: bool = False
    ):
        """
        Create interactive visualization of the knowledge graph

        Args:
            graph: Temporal knowledge graph
            contradictions: Optional list of contradictions to highlight
            output_file: Output HTML file path
            notebook: Whether running in Jupyter notebook
        """
        # Create PyVis network
        net = Network(
            width=self.width,
            height=self.height,
            directed=True,
            notebook=notebook,
            bgcolor="#ffffff",
            font_color="#000000"
        )

        # Configure physics
        net.set_options("""
        {
          "physics": {
            "enabled": true,
            "barnesHut": {
              "gravitationalConstant": -8000,
              "centralGravity": 0.3,
              "springLength": 200,
              "springConstant": 0.04
            },
            "stabilization": {
              "enabled": true,
              "iterations": 100
            }
          },
          "nodes": {
            "font": {
              "size": 16
            }
          },
          "edges": {
            "smooth": {
              "enabled": true,
              "type": "continuous"
            },
            "arrows": {
              "to": {
                "enabled": true,
                "scaleFactor": 0.5
              }
            }
          }
        }
        """)

        # Build contradiction index
        contradiction_nodes = set()
        contradiction_edges = {}

        if contradictions:
            for c in contradictions:
                contradiction_nodes.add(c.fact1_id)
                contradiction_nodes.add(c.fact2_id)
                edge_key = (c.fact1_id, c.fact2_id)
                contradiction_edges[edge_key] = c.severity

        # Add nodes
        for node_id, node_data in graph.graph.nodes(data=True):
            node_type = node_data.get('node_type', 'entity')
            label = node_data.get('label', node_id)
            confidence = node_data.get('confidence', 1.0)

            # Determine node color
            if node_id in contradiction_nodes:
                color = '#ff4444'  # Red for contradictory nodes
            elif node_type == 'event':
                color = '#4444ff'  # Blue for events
            else:
                color = '#44ff44'  # Green for entities

            # Node size based on confidence
            size = 20 + (confidence * 20)

            # Build title (hover text)
            title = f"<b>{label}</b><br>"
            title += f"Type: {node_type}<br>"
            title += f"Confidence: {confidence:.2f}<br>"

            if node_type == 'event':
                desc = node_data.get('description', '')
                timestamp = node_data.get('timestamp', 'No timestamp')
                title += f"Description: {desc}<br>"
                title += f"Time: {timestamp}"
            else:
                entity_type = node_data.get('entity_type', '')
                title += f"Entity Type: {entity_type}"

            net.add_node(
                node_id,
                label=label,
                title=title,
                color=color,
                size=size,
                shape='dot'
            )

        # Add edges
        for source, target, edge_data in graph.graph.edges(data=True):
            relation_type = edge_data.get('relation_type', 'unknown')
            confidence = edge_data.get('confidence', 1.0)
            label = edge_data.get('label', relation_type)

            # Check if this edge represents a contradiction
            edge_key = (source, target)
            is_contradiction = edge_key in contradiction_edges

            if is_contradiction:
                color = '#ff0000'
                width = 5
                dashes = True
            else:
                color = '#888888'
                width = 2
                dashes = False

            # Build title
            title = f"Relation: {relation_type}<br>"
            title += f"Confidence: {confidence:.2f}"

            if edge_data.get('timestamp'):
                title += f"<br>Time: {edge_data['timestamp']}"

            net.add_edge(
                source,
                target,
                label=label,
                title=title,
                color=color,
                width=width,
                dashes=dashes
            )

        # Add contradiction edges (virtual edges between contradictory events)
        if contradictions:
            for c in contradictions:
                if c.fact1_id in graph.graph.nodes and c.fact2_id in graph.graph.nodes:
                    title = f"CONTRADICTION<br>"
                    title += f"Type: {c.contradiction_type}<br>"
                    title += f"Severity: {c.severity:.2f}<br>"
                    title += f"Explanation: {c.explanation}"

                    net.add_edge(
                        c.fact1_id,
                        c.fact2_id,
                        label="CONFLICT",
                        title=title,
                        color='#ff0000',
                        width=5,
                        dashes=True
                    )

        # Save
        net.save_graph(output_file)

        print(f"✓ Graph visualization saved to: {output_file}")

        return output_file

    def create_timeline_visualization(
        self,
        graph: TemporalKnowledgeGraph,
        output_file: str = "timeline.html"
    ):
        """
        Create a timeline visualization of events

        Args:
            graph: Temporal knowledge graph
            output_file: Output HTML file
        """
        timeline = graph.get_timeline()

        if not timeline:
            print("No temporal events to visualize")
            return

        # Create HTML timeline
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Event Timeline</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .timeline {
            position: relative;
            max-width: 1200px;
            margin: 0 auto;
        }
        .timeline::after {
            content: '';
            position: absolute;
            width: 6px;
            background-color: #4444ff;
            top: 0;
            bottom: 0;
            left: 50%;
            margin-left: -3px;
        }
        .container {
            padding: 10px 40px;
            position: relative;
            background-color: inherit;
            width: 50%;
        }
        .container.left {
            left: 0;
        }
        .container.right {
            left: 50%;
        }
        .content {
            padding: 20px 30px;
            background-color: white;
            position: relative;
            border-radius: 6px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .time {
            color: #4444ff;
            font-weight: bold;
            margin-bottom: 10px;
        }
        h1 {
            text-align: center;
            color: #333;
        }
    </style>
</head>
<body>
    <h1>Event Timeline</h1>
    <div class="timeline">
"""

        for i, (timestamp, event_id, description) in enumerate(timeline):
            side = "left" if i % 2 == 0 else "right"

            html_content += f"""
        <div class="container {side}">
            <div class="content">
                <div class="time">{timestamp}</div>
                <p>{description}</p>
                <small>Event ID: {event_id}</small>
            </div>
        </div>
"""

        html_content += """
    </div>
</body>
</html>
"""

        with open(output_file, 'w') as f:
            f.write(html_content)

        print(f"✓ Timeline visualization saved to: {output_file}")

        return output_file

    def generate_report(
        self,
        graph: TemporalKnowledgeGraph,
        contradictions: List[Contradiction],
        corrected_narrative,
        output_file: str = "report.html"
    ):
        """
        Generate comprehensive HTML report

        Args:
            graph: Knowledge graph
            contradictions: List of contradictions
            corrected_narrative: CorrectedNarrative object
            output_file: Output HTML file
        """
        stats = graph.get_statistics()

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Fact Validation Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .stat-box {{
            display: inline-block;
            background-color: #ecf0f1;
            padding: 15px;
            margin: 10px;
            border-radius: 5px;
            min-width: 150px;
        }}
        .stat-label {{
            font-size: 12px;
            color: #7f8c8d;
            text-transform: uppercase;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .contradiction {{
            background-color: #ffe6e6;
            border-left: 4px solid #e74c3c;
            padding: 15px;
            margin: 10px 0;
        }}
        .severity-high {{
            border-left-color: #e74c3c;
        }}
        .severity-medium {{
            border-left-color: #f39c12;
        }}
        .severity-low {{
            border-left-color: #f1c40f;
        }}
        .original-text {{
            background-color: #fff3cd;
            padding: 20px;
            border-radius: 5px;
            margin: 15px 0;
        }}
        .corrected-text {{
            background-color: #d4edda;
            padding: 20px;
            border-radius: 5px;
            margin: 15px 0;
        }}
        .explanation {{
            background-color: #d1ecf1;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 News Fact Validation Report</h1>

        <h2>📊 Graph Statistics</h2>
        <div class="stat-box">
            <div class="stat-label">Total Nodes</div>
            <div class="stat-value">{stats['num_nodes']}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Entities</div>
            <div class="stat-value">{stats['num_entities']}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Events</div>
            <div class="stat-value">{stats['num_events']}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Relations</div>
            <div class="stat-value">{stats['num_edges']}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Contradictions</div>
            <div class="stat-value">{len(contradictions)}</div>
        </div>

        <h2>⚠️ Detected Contradictions</h2>
"""

        if contradictions:
            for i, c in enumerate(contradictions, 1):
                severity_class = (
                    "severity-high" if c.severity >= 0.8 else
                    "severity-medium" if c.severity >= 0.5 else
                    "severity-low"
                )

                event1 = graph.graph.nodes.get(c.fact1_id, {})
                event2 = graph.graph.nodes.get(c.fact2_id, {})

                html_content += f"""
        <div class="contradiction {severity_class}">
            <h3>Contradiction #{i}: {c.contradiction_type}</h3>
            <p><strong>Severity:</strong> {c.severity:.2f}</p>
            <p><strong>Fact 1:</strong> {event1.get('description', 'Unknown')}</p>
            <p><strong>Fact 2:</strong> {event2.get('description', 'Unknown')}</p>
            <p><strong>Explanation:</strong> {c.explanation}</p>
        </div>
"""
        else:
            html_content += "<p>✓ No contradictions detected!</p>"

        html_content += f"""
        <h2>📝 Original Text</h2>
        <div class="original-text">
            {corrected_narrative.original_text}
        </div>

        <h2>✅ Corrected Narrative</h2>
        <div class="corrected-text">
            {corrected_narrative.corrected_text}
        </div>
        <p><strong>Confidence Score:</strong> {corrected_narrative.confidence_score:.2f}</p>

        <h2>💡 Explanation</h2>
        <div class="explanation">
            {corrected_narrative.explanation}
        </div>

    </div>
</body>
</html>
"""

        with open(output_file, 'w') as f:
            f.write(html_content)

        print(f"✓ Report saved to: {output_file}")

        return output_file
