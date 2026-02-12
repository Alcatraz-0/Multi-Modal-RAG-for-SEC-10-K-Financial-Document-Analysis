"""
Query router for classifying queries and routing to appropriate retrieval strategy
"""
from typing import Dict, Any
import re


class QueryRouter:
    """Route queries based on type and requirements"""

    def __init__(self):
        """Initialize query router"""
        # Keywords indicating table-centric queries
        self.table_keywords = [
            'yoy', 'year-over-year', 'change', 'growth', 'ratio',
            'compared to', 'versus', 'vs', 'percentage', '%',
            'segment', 'total', 'revenue', 'expense', 'debt', 'equity',
            'fiscal year', 'fy', '2022', '2023', '2024'
        ]

        # Keywords indicating math requirements
        self.math_keywords = [
            'calculate', 'compute', 'difference', 'sum', 'total',
            'ratio', 'percentage', 'change', 'growth', 'increase', 'decrease'
        ]

    def route(self, query: str) -> Dict[str, Any]:
        """
        Classify query and determine routing

        Args:
            query: User query string

        Returns:
            Dictionary with routing information
        """
        query_lower = query.lower()

        # Detect table-centric query
        is_table_centric = self._is_table_centric(query_lower)

        # Detect math requirement
        requires_math = self._requires_math(query_lower)

        # Determine query type
        if is_table_centric and requires_math:
            query_type = 'numeric_table'
        elif is_table_centric:
            query_type = 'table_lookup'
        elif requires_math:
            query_type = 'numeric_text'
        else:
            query_type = 'narrative'

        # Confidence score (simplified)
        confidence = 0.8 if is_table_centric else 0.6

        return {
            'query_type': query_type,
            'is_table_centric': is_table_centric,
            'requires_math': requires_math,
            'confidence': confidence
        }

    def _is_table_centric(self, query: str) -> bool:
        """Check if query is table-centric"""
        # Count table-related keywords
        keyword_count = sum(1 for kw in self.table_keywords if kw in query)
        return keyword_count >= 2

    def _requires_math(self, query: str) -> bool:
        """Check if query requires mathematical operations"""
        return any(kw in query for kw in self.math_keywords)
