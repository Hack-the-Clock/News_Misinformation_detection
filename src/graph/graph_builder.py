"""Build knowledge graph from extracted facts"""

from typing import Dict, List
from .temporal_graph import TemporalKnowledgeGraph
from ..models.fact import Entity, Event, Relation


class GraphBuilder:
    """Build a temporal knowledge graph from extracted entities, events, and relations"""

    def __init__(self):
        self.graph = TemporalKnowledgeGraph()

    def build_from_extraction(self, extraction_result: Dict) -> TemporalKnowledgeGraph:
        """
        Build a knowledge graph from extraction results

        Args:
            extraction_result: Dictionary containing entities, events, and relations

        Returns:
            Populated TemporalKnowledgeGraph
        """
        # Add all entities
        entities: List[Entity] = extraction_result.get('entities', [])
        for entity in entities:
            self.graph.add_entity(entity)

        # Add all events
        events: List[Event] = extraction_result.get('events', [])
        for event in events:
            self.graph.add_event(event)

        # Add all relations
        relations: List[Relation] = extraction_result.get('relations', [])
        for relation in relations:
            self.graph.add_relation(relation)

        # Create implicit temporal relations between events
        self._create_temporal_relations(events)

        # Link events to involved entities
        self._link_events_to_entities(events, entities)

        return self.graph

    def _create_temporal_relations(self, events: List[Event]):
        """
        Create temporal ordering relations between events

        Args:
            events: List of events
        """
        from ..models.fact import Relation, RelationType
        import uuid

        # Sort events by timestamp
        timed_events = [(e, e.timestamp) for e in events if e.timestamp]
        timed_events.sort(key=lambda x: x[1])

        # Create TEMPORAL_BEFORE relations
        for i in range(len(timed_events) - 1):
            event1, time1 = timed_events[i]
            event2, time2 = timed_events[i + 1]

            relation = Relation(
                id=f"temporal_{uuid.uuid4().hex[:8]}",
                source_id=event1.id,
                target_id=event2.id,
                relation_type=RelationType.TEMPORAL_BEFORE,
                timestamp=time1,
                confidence=1.0
            )

            self.graph.add_relation(relation)

    def _link_events_to_entities(self, events: List[Event], entities: List[Entity]):
        """
        Create relations between events and their involved entities

        Args:
            events: List of events
            entities: List of entities
        """
        from ..models.fact import Relation, RelationType
        import uuid

        # Create entity name lookup
        entity_lookup = {e.name.lower(): e for e in entities}

        for event in events:
            for entity_name in event.entities_involved:
                entity = entity_lookup.get(entity_name.lower())
                if entity:
                    relation = Relation(
                        id=f"involves_{uuid.uuid4().hex[:8]}",
                        source_id=event.id,
                        target_id=entity.id,
                        relation_type=RelationType.INVOLVES,
                        timestamp=event.timestamp,
                        confidence=0.9
                    )

                    self.graph.add_relation(relation)

    def get_graph(self) -> TemporalKnowledgeGraph:
        """Get the built graph"""
        return self.graph
