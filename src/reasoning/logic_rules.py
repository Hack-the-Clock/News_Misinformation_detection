"""Symbolic logic rules for fact validation"""

from typing import List, Tuple, Dict
from datetime import datetime
from ..models.fact import Contradiction
import uuid


class LogicRule:
    """Base class for logic rules"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def check(self, event1: Dict, event2: Dict) -> Tuple[bool, float, str]:
        """
        Check if rule is violated

        Args:
            event1: First event node data
            event2: Second event node data

        Returns:
            (is_violated, severity, explanation)
        """
        raise NotImplementedError


class TemporalConsistencyRule(LogicRule):
    """Events should follow logical temporal ordering"""

    def __init__(self):
        super().__init__(
            "Temporal Consistency",
            "Events should occur in a logically consistent temporal order"
        )

    def check(self, event1: Dict, event2: Dict) -> Tuple[bool, float, str]:
        """Check temporal consistency"""
        time1 = event1.get('timestamp')
        time2 = event2.get('timestamp')

        if not time1 or not time2:
            return False, 0.0, "No temporal information"

        # Check if events contradict temporal logic
        # Example: "earlier that day" should come before "yesterday"
        desc1 = event1.get('description', '').lower()
        desc2 = event2.get('description', '').lower()

        # Check if same day
        same_day = (time1.date() == time2.date())

        if same_day:
            # Events on the same day with contradictory outcomes
            if self._has_contradictory_content(desc1, desc2):
                return True, 0.9, f"Events on same day ({time1.date()}) contain contradictory information"

        return False, 0.0, "Temporally consistent"

    def _has_contradictory_content(self, desc1: str, desc2: str) -> bool:
        """Check if descriptions have contradictory content"""
        opposites = [
            ('zero', 'rise'),
            ('zero', 'increase'),
            ('no', 'yes'),
            ('denied', 'confirmed'),
            ('decrease', 'increase'),
        ]

        for word1, word2 in opposites:
            if (word1 in desc1 and word2 in desc2) or (word2 in desc1 and word1 in desc2):
                return True

        return False


class MutualExclusivityRule(LogicRule):
    """Mutually exclusive facts cannot both be true"""

    def __init__(self):
        super().__init__(
            "Mutual Exclusivity",
            "Contradictory facts about the same entity cannot both be true"
        )

    def check(self, event1: Dict, event2: Dict) -> Tuple[bool, float, str]:
        """Check mutual exclusivity"""
        desc1 = event1.get('description', '').lower()
        desc2 = event2.get('description', '').lower()

        # Check for explicit contradictions
        contradictions = {
            'zero': ['rise', 'increase', 'growth'],
            'no cases': ['rise in cases', 'increase in cases'],
            'decline': ['increase', 'rise'],
            'denied': ['confirmed', 'admitted'],
        }

        for key, conflicting_terms in contradictions.items():
            if key in desc1:
                for term in conflicting_terms:
                    if term in desc2:
                        return True, 1.0, f"Mutually exclusive: '{key}' conflicts with '{term}'"

            if key in desc2:
                for term in conflicting_terms:
                    if term in desc1:
                        return True, 1.0, f"Mutually exclusive: '{key}' conflicts with '{term}'"

        return False, 0.0, "No mutual exclusivity violation"


class SourceConsistencyRule(LogicRule):
    """Same source should not contradict itself"""

    def __init__(self):
        super().__init__(
            "Source Consistency",
            "The same source should provide consistent information"
        )

    def check(self, event1: Dict, event2: Dict) -> Tuple[bool, float, str]:
        """Check source consistency"""
        # Check if events share the same source
        entities1 = set(event1.get('entities_involved', []))
        entities2 = set(event2.get('entities_involved', []))

        common_sources = entities1.intersection(entities2)

        if common_sources:
            # Same source, check for contradictions
            desc1 = event1.get('description', '').lower()
            desc2 = event2.get('description', '').lower()

            if self._are_contradictory(desc1, desc2):
                source_names = ', '.join(common_sources)
                return True, 0.95, f"Source '{source_names}' provides contradictory information"

        return False, 0.0, "Source consistency maintained"

    def _are_contradictory(self, desc1: str, desc2: str) -> bool:
        """Check if descriptions contradict"""
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


class EventOrderingRule(LogicRule):
    """Events should follow causal ordering"""

    def __init__(self):
        super().__init__(
            "Event Ordering",
            "Effects cannot occur before their causes"
        )

    def check(self, event1: Dict, event2: Dict) -> Tuple[bool, float, str]:
        """Check event ordering"""
        time1 = event1.get('timestamp')
        time2 = event2.get('timestamp')

        if not time1 or not time2:
            return False, 0.0, "No temporal information"

        desc1 = event1.get('description', '').lower()
        desc2 = event2.get('description', '').lower()

        # Check for causal keywords
        causes = ['because', 'due to', 'caused by', 'led to', 'resulted in']
        effects = ['therefore', 'thus', 'consequently', 'as a result']

        # If event1 is a cause, it should happen before event2
        if any(word in desc1 for word in causes):
            if time1 > time2:
                return True, 0.8, "Cause occurs after effect"

        return False, 0.0, "Event ordering is valid"


class LogicRuleEngine:
    """Engine to apply all logic rules"""

    def __init__(self):
        self.rules = [
            TemporalConsistencyRule(),
            MutualExclusivityRule(),
            SourceConsistencyRule(),
            EventOrderingRule(),
        ]

    def check_all_rules(
        self,
        event1_id: str,
        event1_data: Dict,
        event2_id: str,
        event2_data: Dict
    ) -> List[Contradiction]:
        """
        Apply all rules to a pair of events

        Args:
            event1_id: First event ID
            event1_data: First event data
            event2_id: Second event ID
            event2_data: Second event data

        Returns:
            List of detected contradictions
        """
        contradictions = []

        for rule in self.rules:
            is_violated, severity, explanation = rule.check(event1_data, event2_data)

            if is_violated:
                contradiction = Contradiction(
                    id=f"contradiction_{uuid.uuid4().hex[:8]}",
                    fact1_id=event1_id,
                    fact2_id=event2_id,
                    contradiction_type=rule.name,
                    severity=severity,
                    explanation=explanation,
                    temporal_conflict='Temporal' in rule.name,
                    semantic_conflict='Exclusivity' in rule.name or 'Consistency' in rule.name
                )
                contradictions.append(contradiction)

        return contradictions

    def add_rule(self, rule: LogicRule):
        """Add a custom rule"""
        self.rules.append(rule)
