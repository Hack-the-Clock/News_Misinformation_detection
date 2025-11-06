"""Generate corrected narratives from validated facts"""

from typing import List, Dict
from ..llm.llm_client import LLMClient
from ..models.fact import Contradiction, CorrectedNarrative
from ..graph.temporal_graph import TemporalKnowledgeGraph


class NarrativeGenerator:
    """Generate corrected narratives after fact validation"""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def generate_corrected_narrative(
        self,
        original_text: str,
        contradictions: List[Contradiction],
        graph: TemporalKnowledgeGraph,
        removed_facts: List[str] = None
    ) -> CorrectedNarrative:
        """
        Generate a corrected version of the narrative

        Args:
            original_text: Original article text
            contradictions: List of detected contradictions
            graph: Knowledge graph
            removed_facts: List of fact IDs that were removed

        Returns:
            CorrectedNarrative object
        """
        if not contradictions:
            return CorrectedNarrative(
                original_text=original_text,
                contradictions=[],
                corrected_text=original_text,
                confidence_score=1.0,
                explanation="No contradictions detected. Original text is consistent.",
                facts_retained=[]
            )

        # Build contradiction explanation
        contradiction_details = self._format_contradictions(contradictions, graph)

        # Generate correction prompt
        system_prompt = """You are an expert fact-checker and editor.

Given an original text and a list of detected contradictions, generate a corrected version of the text that:
1. Removes or corrects contradictory information
2. Maintains the core factual information
3. Preserves the original writing style
4. Clearly indicates what was changed and why

Be objective and conservative in your corrections."""

        user_prompt = f"""Original Text:
{original_text}

Detected Contradictions:
{contradiction_details}

Please provide:
1. A corrected version of the text
2. An explanation of what was changed and why
3. A confidence score (0-1) for the correction

Format your response as:
CORRECTED TEXT:
[corrected version here]

EXPLANATION:
[explanation here]

CONFIDENCE:
[0.0-1.0]
"""

        # Generate correction
        response = self.llm_client.generate_text(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.3
        )

        # Parse response
        corrected_text, explanation, confidence = self._parse_correction_response(response)

        # Determine which facts were retained
        facts_retained = self._identify_retained_facts(graph, contradictions, removed_facts)

        return CorrectedNarrative(
            original_text=original_text,
            contradictions=contradictions,
            corrected_text=corrected_text,
            confidence_score=confidence,
            explanation=explanation,
            facts_removed=removed_facts or [],
            facts_retained=facts_retained
        )

    def _format_contradictions(
        self,
        contradictions: List[Contradiction],
        graph: TemporalKnowledgeGraph
    ) -> str:
        """Format contradictions for the LLM prompt"""
        formatted = []

        for i, contradiction in enumerate(contradictions, 1):
            # Get event details
            event1 = graph.graph.nodes.get(contradiction.fact1_id, {})
            event2 = graph.graph.nodes.get(contradiction.fact2_id, {})

            desc1 = event1.get('description', 'Unknown event')
            desc2 = event2.get('description', 'Unknown event')
            time1 = event1.get('timestamp', 'Unknown time')
            time2 = event2.get('timestamp', 'Unknown time')

            formatted.append(
                f"{i}. {contradiction.contradiction_type} (Severity: {contradiction.severity:.2f}):\n"
                f"   - Fact 1: \"{desc1}\" (Time: {time1})\n"
                f"   - Fact 2: \"{desc2}\" (Time: {time2})\n"
                f"   - Explanation: {contradiction.explanation}\n"
            )

        return "\n".join(formatted)

    def _parse_correction_response(self, response: str) -> tuple:
        """Parse the LLM correction response"""
        corrected_text = ""
        explanation = ""
        confidence = 0.8

        lines = response.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()

            if 'CORRECTED TEXT:' in line:
                current_section = 'corrected'
                continue
            elif 'EXPLANATION:' in line:
                current_section = 'explanation'
                continue
            elif 'CONFIDENCE:' in line:
                current_section = 'confidence'
                continue

            if current_section == 'corrected' and line:
                corrected_text += line + " "
            elif current_section == 'explanation' and line:
                explanation += line + " "
            elif current_section == 'confidence' and line:
                try:
                    confidence = float(line)
                except ValueError:
                    pass

        return corrected_text.strip(), explanation.strip(), confidence

    def _identify_retained_facts(
        self,
        graph: TemporalKnowledgeGraph,
        contradictions: List[Contradiction],
        removed_facts: List[str]
    ) -> List[str]:
        """Identify which facts were retained"""
        removed_facts = removed_facts or []

        # Get all event IDs
        all_events = [
            node_id for node_id, data in graph.graph.nodes(data=True)
            if data.get('node_type') == 'event'
        ]

        # Facts that were not removed
        retained = [
            event_id for event_id in all_events
            if event_id not in removed_facts
        ]

        return retained

    def suggest_alternative_narratives(
        self,
        original_text: str,
        contradictions: List[Contradiction],
        graph: TemporalKnowledgeGraph,
        num_alternatives: int = 2
    ) -> List[str]:
        """
        Generate multiple alternative corrected narratives

        Args:
            original_text: Original text
            contradictions: Detected contradictions
            graph: Knowledge graph
            num_alternatives: Number of alternatives to generate

        Returns:
            List of alternative narrative texts
        """
        alternatives = []

        for i in range(num_alternatives):
            system_prompt = f"""You are an expert fact-checker generating alternative interpretation #{i+1}.

Provide a different perspective on how to resolve the contradictions while staying factual."""

            contradiction_details = self._format_contradictions(contradictions, graph)

            user_prompt = f"""Original Text:
{original_text}

Contradictions:
{contradiction_details}

Provide an alternative corrected narrative:"""

            alternative = self.llm_client.generate_text(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.7  # Higher temperature for diversity
            )

            alternatives.append(alternative.strip())

        return alternatives
