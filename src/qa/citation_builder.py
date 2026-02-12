"""
Citation builder for generating precise source citations
"""
from typing import List, Dict, Any


class CitationBuilder:
    """Build precise citations for answers"""

    def __init__(self):
        """Initialize citation builder with default citation format"""

    def build_citations(
        self,
        answer: str,
        evidence: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Build citations from evidence

        Args:
            answer: Generated answer
            evidence: Retrieved evidence pieces

        Returns:
            List of citation strings
        """
        citations = []

        # Find which evidence pieces are actually referenced
        # (simple approach: take top pieces; better would parse answer for refs)

        for ev in evidence[:3]:  # Top 3 pieces
            meta = ev['metadata']
            content_type = meta['content_type']

            if content_type == 'table':
                citation = self._build_table_citation(meta)
            else:
                citation = self._build_text_citation(meta)

            citations.append(citation)

        # Remove duplicates while preserving order
        seen = set()
        unique_citations = []
        for cit in citations:
            if cit not in seen:
                seen.add(cit)
                unique_citations.append(cit)

        return unique_citations

    def _build_table_citation(self, metadata: Dict[str, Any]) -> str:
        """Build citation for table evidence"""
        ticker = metadata['ticker']
        fiscal_year = metadata['fiscal_year']
        table_id = metadata.get('table_id', 'Unknown')
        section = metadata.get('section', 'Unknown Section')
        row_idx = metadata.get('row_idx', '')

        citation = f"{ticker} {fiscal_year} 10-K, {section}, Table {table_id}"

        if row_idx != '':
            citation += f", Row {row_idx}"

        return citation

    def _build_text_citation(self, metadata: Dict[str, Any]) -> str:
        """Build citation for text evidence"""
        ticker = metadata['ticker']
        fiscal_year = metadata['fiscal_year']
        section_title = metadata.get('section_title', 'Unknown Section')

        citation = f"{ticker} {fiscal_year} 10-K, {section_title}"

        return citation
