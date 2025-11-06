"""Contradiction detection using symbolic logic rules"""

from typing import List, Dict
from .logic_rules import LogicRuleEngine
from ..models.fact import Contradiction
from ..graph.temporal_graph import TemporalKnowledgeGraph


class ContradictionDetector:
    """Detect contradictions in the knowledge graph using symbolic logic"""

    def __init__(self, rule_engine: LogicRuleEngine = None):
        """
        Initialize detector

        Args:
            rule_engine: Optional custom rule engine
        """
        self.rule_engine = rule_engine or LogicRuleEngine()

    def detect_contradictions(
        self,
        graph: TemporalKnowledgeGraph
    ) -> List[Contradiction]:
        """
        Detect all contradictions in the knowledge graph

        Args:
            graph: Temporal knowledge graph

        Returns:
            List of detected contradictions
        """
        all_contradictions = []

        # Get all event nodes
        event_nodes = [
            (node_id, data) for node_id, data in graph.graph.nodes(data=True)
            if data.get('node_type') == 'event'
        ]

        # Check all pairs of events
        for i, (id1, data1) in enumerate(event_nodes):
            for id2, data2 in event_nodes[i+1:]:
                # Apply all logic rules
                contradictions = self.rule_engine.check_all_rules(
                    id1, data1, id2, data2
                )

                all_contradictions.extend(contradictions)

        # Sort by severity (highest first)
        all_contradictions.sort(key=lambda c: c.severity, reverse=True)

        return all_contradictions

    def detect_in_timeline(
        self,
        graph: TemporalKnowledgeGraph
    ) -> Dict[str, List[Contradiction]]:
        """
        Detect contradictions organized by timeline

        Args:
            graph: Temporal knowledge graph

        Returns:
            Dictionary mapping time periods to contradictions
        """
        contradictions = self.detect_contradictions(graph)

        # Group by involved events' timestamps
        timeline_contradictions = {}

        for contradiction in contradictions:
            # Get event data
            event1_data = graph.graph.nodes.get(contradiction.fact1_id, {})
            event2_data = graph.graph.nodes.get(contradiction.fact2_id, {})

            time1 = event1_data.get('timestamp')
            time2 = event2_data.get('timestamp')

            if time1 and time2:
                time_key = f"{time1.date()} to {time2.date()}"
            else:
                time_key = "unknown_time"

            if time_key not in timeline_contradictions:
                timeline_contradictions[time_key] = []

            timeline_contradictions[time_key].append(contradiction)

        return timeline_contradictions

    def get_contradiction_summary(
        self,
        contradictions: List[Contradiction]
    ) -> Dict:
        """
        Get a summary of detected contradictions

        Args:
            contradictions: List of contradictions

        Returns:
            Summary dictionary
        """
        if not contradictions:
            return {
                'total': 0,
                'by_type': {},
                'high_severity': 0,
                'medium_severity': 0,
                'low_severity': 0
            }

        by_type = {}
        for c in contradictions:
            if c.contradiction_type not in by_type:
                by_type[c.contradiction_type] = 0
            by_type[c.contradiction_type] += 1

        high_severity = sum(1 for c in contradictions if c.severity >= 0.8)
        medium_severity = sum(1 for c in contradictions if 0.5 <= c.severity < 0.8)
        low_severity = sum(1 for c in contradictions if c.severity < 0.5)

        return {
            'total': len(contradictions),
            'by_type': by_type,
            'high_severity': high_severity,
            'medium_severity': medium_severity,
            'low_severity': low_severity,
            'temporal_conflicts': sum(1 for c in contradictions if c.temporal_conflict),
            'semantic_conflicts': sum(1 for c in contradictions if c.semantic_conflict)
        }
