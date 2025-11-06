"""Temporal Knowledge Graph using NetworkX"""

import networkx as nx
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import pickle

from ..models.fact import Entity, Event, Relation, EntityType, RelationType


class TemporalKnowledgeGraph:
    """
    Temporal knowledge graph for storing and querying facts with time information
    """

    def __init__(self):
        """Initialize an empty directed graph"""
        self.graph = nx.MultiDiGraph()  # Allows multiple edges between nodes
        self.entity_index = {}  # name -> node_id mapping
        self.event_index = {}   # description -> node_id mapping

    def add_entity(self, entity: Entity) -> str:
        """
        Add an entity node to the graph

        Args:
            entity: Entity object

        Returns:
            Node ID in the graph
        """
        node_id = entity.id

        self.graph.add_node(
            node_id,
            node_type='entity',
            name=entity.name,
            entity_type=entity.entity_type.value,
            properties=entity.properties,
            confidence=entity.confidence,
            label=entity.name  # For visualization
        )

        self.entity_index[entity.name.lower()] = node_id
        return node_id

    def add_event(self, event: Event) -> str:
        """
        Add an event node to the graph

        Args:
            event: Event object

        Returns:
            Node ID in the graph
        """
        node_id = event.id

        self.graph.add_node(
            node_id,
            node_type='event',
            description=event.description,
            timestamp=event.timestamp,
            temporal_expression=event.temporal_expression,
            entities_involved=event.entities_involved,
            properties=event.properties,
            confidence=event.confidence,
            label=event.description[:50]  # Truncate for visualization
        )

        self.event_index[event.description.lower()] = node_id
        return node_id

    def add_relation(self, relation: Relation) -> str:
        """
        Add a relation edge to the graph

        Args:
            relation: Relation object

        Returns:
            Edge key in the graph
        """
        # Resolve source and target IDs
        source_id = self._resolve_node_id(relation.source_id)
        target_id = self._resolve_node_id(relation.target_id)

        if not source_id or not target_id:
            print(f"Warning: Could not resolve nodes for relation {relation.id}")
            return None

        # Add edge with properties
        edge_key = self.graph.add_edge(
            source_id,
            target_id,
            relation_id=relation.id,
            relation_type=relation.relation_type.value,
            timestamp=relation.timestamp,
            properties=relation.properties,
            confidence=relation.confidence,
            label=relation.relation_type.value  # For visualization
        )

        return edge_key

    def _resolve_node_id(self, identifier: str) -> Optional[str]:
        """
        Resolve a node identifier (name or ID) to a node ID

        Args:
            identifier: Entity/event name or ID

        Returns:
            Node ID or None
        """
        # Check if it's already a node ID
        if identifier in self.graph.nodes:
            return identifier

        # Check entity index
        if identifier.lower() in self.entity_index:
            return self.entity_index[identifier.lower()]

        # Check event index
        if identifier.lower() in self.event_index:
            return self.event_index[identifier.lower()]

        return None

    def get_events_by_timerange(
        self,
        start: datetime,
        end: datetime
    ) -> List[Tuple[str, Dict]]:
        """
        Get all events within a time range

        Args:
            start: Start datetime
            end: End datetime

        Returns:
            List of (node_id, node_data) tuples
        """
        events = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get('node_type') == 'event':
                timestamp = data.get('timestamp')
                if timestamp and start <= timestamp <= end:
                    events.append((node_id, data))

        # Sort by timestamp
        events.sort(key=lambda x: x[1].get('timestamp'))
        return events

    def get_temporal_neighbors(
        self,
        node_id: str,
        relation_type: Optional[RelationType] = None
    ) -> List[Tuple[str, Dict, Dict]]:
        """
        Get neighbors of a node with temporal information

        Args:
            node_id: Node ID
            relation_type: Optional filter by relation type

        Returns:
            List of (neighbor_id, neighbor_data, edge_data) tuples
        """
        neighbors = []

        for neighbor_id in self.graph.neighbors(node_id):
            # Get all edges between node and neighbor
            edges = self.graph[node_id][neighbor_id]

            for edge_key, edge_data in edges.items():
                if relation_type is None or edge_data.get('relation_type') == relation_type.value:
                    neighbor_data = self.graph.nodes[neighbor_id]
                    neighbors.append((neighbor_id, neighbor_data, edge_data))

        return neighbors

    def find_contradictions_by_temporal_order(self) -> List[Tuple]:
        """
        Find contradictory events based on temporal ordering

        Returns:
            List of (event1_id, event2_id, reason) tuples
        """
        contradictions = []
        event_nodes = [
            (nid, data) for nid, data in self.graph.nodes(data=True)
            if data.get('node_type') == 'event'
        ]

        # Compare pairs of events
        for i, (id1, data1) in enumerate(event_nodes):
            for id2, data2 in event_nodes[i+1:]:
                # Check for temporal contradictions
                time1 = data1.get('timestamp')
                time2 = data2.get('timestamp')

                if time1 and time2:
                    # Check if events contradict each other
                    desc1 = data1.get('description', '').lower()
                    desc2 = data2.get('description', '').lower()

                    # Simple contradiction detection
                    if self._are_contradictory(desc1, desc2):
                        reason = f"Events at {time1} and {time2} contain contradictory information"
                        contradictions.append((id1, id2, reason))

        return contradictions

    def _are_contradictory(self, desc1: str, desc2: str) -> bool:
        """
        Simple heuristic to detect contradictory descriptions

        Args:
            desc1: First description
            desc2: Second description

        Returns:
            True if descriptions appear contradictory
        """
        contradiction_pairs = [
            ('zero', 'rise'),
            ('no', 'increase'),
            ('decrease', 'increase'),
            ('denied', 'confirmed'),
        ]

        for word1, word2 in contradiction_pairs:
            if (word1 in desc1 and word2 in desc2) or (word2 in desc1 and word1 in desc2):
                return True

        return False

    def get_timeline(self) -> List[Tuple[datetime, str, str]]:
        """
        Get a chronological timeline of all events

        Returns:
            List of (timestamp, event_id, description) tuples sorted by time
        """
        timeline = []

        for node_id, data in self.graph.nodes(data=True):
            if data.get('node_type') == 'event':
                timestamp = data.get('timestamp')
                if timestamp:
                    description = data.get('description', '')
                    timeline.append((timestamp, node_id, description))

        timeline.sort(key=lambda x: x[0])
        return timeline

    def save(self, filepath: str):
        """Save the graph to a file"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'graph': self.graph,
                'entity_index': self.entity_index,
                'event_index': self.event_index
            }, f)

    def load(self, filepath: str):
        """Load the graph from a file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.graph = data['graph']
            self.entity_index = data['entity_index']
            self.event_index = data['event_index']

    def get_statistics(self) -> Dict:
        """Get graph statistics"""
        num_entities = sum(
            1 for _, data in self.graph.nodes(data=True)
            if data.get('node_type') == 'entity'
        )
        num_events = sum(
            1 for _, data in self.graph.nodes(data=True)
            if data.get('node_type') == 'event'
        )

        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'num_entities': num_entities,
            'num_events': num_events,
            'density': nx.density(self.graph)
        }
