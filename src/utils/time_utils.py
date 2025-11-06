"""Temporal utilities for parsing and reasoning about time"""

from datetime import datetime, timedelta
from typing import Optional, Tuple
import dateparser


class TemporalParser:
    """Parse and normalize temporal expressions"""

    @staticmethod
    def parse_temporal_expression(
        text: str,
        reference_date: Optional[datetime] = None
    ) -> Optional[datetime]:
        """
        Parse a temporal expression into a datetime object

        Args:
            text: Temporal expression (e.g., "yesterday", "earlier that day")
            reference_date: Reference date for relative expressions

        Returns:
            Parsed datetime or None if parsing fails
        """
        if reference_date is None:
            reference_date = datetime.now()

        # Use dateparser for flexible parsing
        settings = {
            'RELATIVE_BASE': reference_date,
            'PREFER_DATES_FROM': 'past'
        }

        parsed = dateparser.parse(text, settings=settings)
        return parsed

    @staticmethod
    def extract_temporal_order(time1: datetime, time2: datetime) -> str:
        """
        Determine temporal ordering between two timestamps

        Returns:
            'before', 'after', 'simultaneous', or 'unknown'
        """
        if time1 is None or time2 is None:
            return 'unknown'

        # Allow 1-minute margin for "simultaneous"
        if abs((time1 - time2).total_seconds()) < 60:
            return 'simultaneous'
        elif time1 < time2:
            return 'before'
        else:
            return 'after'

    @staticmethod
    def are_temporally_consistent(
        event1: Tuple[str, datetime],
        event2: Tuple[str, datetime],
        constraint: str
    ) -> bool:
        """
        Check if two events satisfy a temporal constraint

        Args:
            event1: (description, timestamp)
            event2: (description, timestamp)
            constraint: 'before', 'after', 'simultaneous', or 'not_simultaneous'

        Returns:
            True if constraint is satisfied
        """
        _, time1 = event1
        _, time2 = event2

        order = TemporalParser.extract_temporal_order(time1, time2)

        if constraint == 'before':
            return order == 'before'
        elif constraint == 'after':
            return order == 'after'
        elif constraint == 'simultaneous':
            return order == 'simultaneous'
        elif constraint == 'not_simultaneous':
            return order != 'simultaneous'

        return False

    @staticmethod
    def normalize_to_reference(
        temporal_expr: str,
        reference_date: datetime
    ) -> Optional[datetime]:
        """
        Normalize a temporal expression relative to a reference date

        Example:
            "earlier that day" with reference 2024-01-15 14:00
            -> 2024-01-15 10:00 (earlier in the same day)
        """
        parsed = TemporalParser.parse_temporal_expression(
            temporal_expr, reference_date
        )

        if parsed:
            # Handle "earlier that day" specifically
            if "earlier" in temporal_expr.lower() and "day" in temporal_expr.lower():
                # Set to morning of the reference date
                return reference_date.replace(hour=8, minute=0, second=0)

            # Handle "later that day"
            if "later" in temporal_expr.lower() and "day" in temporal_expr.lower():
                # Set to evening of the reference date
                return reference_date.replace(hour=18, minute=0, second=0)

        return parsed
