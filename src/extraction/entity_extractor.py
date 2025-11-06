"""Entity extraction using LLM"""

from typing import List, Dict
from ..llm.llm_client import LLMClient
from ..models.fact import Entity, EntityType, Event, Relation, RelationType, ExtractedFact
from ..utils.time_utils import TemporalParser
from datetime import datetime
import uuid


class EntityExtractor:
    """Extract entities, events, and relationships from text"""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.temporal_parser = TemporalParser()

    def extract_all(self, text: str, reference_date: datetime = None) -> Dict:
        """
        Extract entities, events, and relationships from text

        Args:
            text: Input article or text snippet
            reference_date: Reference date for temporal parsing

        Returns:
            Dictionary containing entities, events, and relations
        """
        if reference_date is None:
            reference_date = datetime.now()

        # Define extraction prompt
        system_prompt = """You are an expert at extracting structured information from news articles.

Extract the following from the given text:
1. Entities (people, places, organizations, sources)
2. Events (what happened, when)
3. Relationships between entities and events

Return your response in JSON format with the following structure:
{
    "entities": [
        {
            "name": "entity name",
            "type": "location|organization|person|source|concept",
            "properties": {}
        }
    ],
    "events": [
        {
            "description": "what happened",
            "temporal_expression": "yesterday|earlier that day|etc",
            "entities_involved": ["entity names"],
            "properties": {"attribute": "value"}
        }
    ],
    "relations": [
        {
            "source": "entity or event",
            "relation": "reported|confirmed|contradicts|involves|happens_at",
            "target": "entity or event",
            "temporal_expression": "when this relation occurred"
        }
    ]
}

Be precise and extract ALL temporal information. Pay special attention to:
- Temporal expressions (yesterday, earlier that day, later, etc.)
- Contradictory statements
- Source attributions
"""

        # Call LLM for extraction
        response = self.llm_client.extract_structured_data(
            text=text,
            system_prompt=system_prompt,
            response_format="json"
        )

        # Parse and structure the results
        entities = self._parse_entities(response.get("entities", []))
        events = self._parse_events(
            response.get("events", []),
            reference_date
        )
        relations = self._parse_relations(
            response.get("relations", []),
            reference_date
        )

        return {
            "entities": entities,
            "events": events,
            "relations": relations,
            "raw_text": text
        }

    def _parse_entities(self, entity_data: List[Dict]) -> List[Entity]:
        """Parse entity data from LLM response"""
        entities = []

        for ent in entity_data:
            entity_type_str = ent.get("type", "concept").lower()

            # Map to EntityType enum
            try:
                entity_type = EntityType(entity_type_str)
            except ValueError:
                entity_type = EntityType.CONCEPT

            entity = Entity(
                id=f"entity_{uuid.uuid4().hex[:8]}",
                name=ent.get("name", ""),
                entity_type=entity_type,
                properties=ent.get("properties", {}),
                confidence=ent.get("confidence", 0.9)
            )
            entities.append(entity)

        return entities

    def _parse_events(
        self,
        event_data: List[Dict],
        reference_date: datetime
    ) -> List[Event]:
        """Parse event data from LLM response"""
        events = []

        for evt in event_data:
            temporal_expr = evt.get("temporal_expression", "")
            timestamp = None

            if temporal_expr:
                timestamp = self.temporal_parser.normalize_to_reference(
                    temporal_expr,
                    reference_date
                )

            event = Event(
                id=f"event_{uuid.uuid4().hex[:8]}",
                description=evt.get("description", ""),
                timestamp=timestamp,
                temporal_expression=temporal_expr,
                entities_involved=evt.get("entities_involved", []),
                properties=evt.get("properties", {}),
                confidence=evt.get("confidence", 0.9)
            )
            events.append(event)

        return events

    def _parse_relations(
        self,
        relation_data: List[Dict],
        reference_date: datetime
    ) -> List[Relation]:
        """Parse relation data from LLM response"""
        relations = []

        for rel in relation_data:
            relation_type_str = rel.get("relation", "involves").lower()

            # Map to RelationType enum
            try:
                relation_type = RelationType(relation_type_str)
            except ValueError:
                relation_type = RelationType.INVOLVES

            temporal_expr = rel.get("temporal_expression", "")
            timestamp = None

            if temporal_expr:
                timestamp = self.temporal_parser.parse_temporal_expression(
                    temporal_expr,
                    reference_date
                )

            relation = Relation(
                id=f"relation_{uuid.uuid4().hex[:8]}",
                source_id=rel.get("source", ""),
                target_id=rel.get("target", ""),
                relation_type=relation_type,
                timestamp=timestamp,
                properties=rel.get("properties", {}),
                confidence=rel.get("confidence", 0.9)
            )
            relations.append(relation)

        return relations

    def extract_facts(self, text: str) -> List[ExtractedFact]:
        """
        Extract facts in (subject, predicate, object, temporal) format

        Args:
            text: Input text

        Returns:
            List of extracted facts
        """
        system_prompt = """Extract facts from the text in the format: (subject, predicate, object, temporal_info).

Return JSON:
{
    "facts": [
        {
            "subject": "who or what",
            "predicate": "action or relation",
            "object": "target or value",
            "temporal_expression": "when",
            "confidence": 0.0-1.0
        }
    ]
}"""

        response = self.llm_client.extract_structured_data(
            text=text,
            system_prompt=system_prompt,
            response_format="json"
        )

        facts = []
        for fact_data in response.get("facts", []):
            fact = ExtractedFact(
                subject=fact_data.get("subject", ""),
                predicate=fact_data.get("predicate", ""),
                object=fact_data.get("object", ""),
                temporal_expression=fact_data.get("temporal_expression"),
                source_text=text,
                confidence=fact_data.get("confidence", 0.9)
            )
            facts.append(fact)

        return facts
