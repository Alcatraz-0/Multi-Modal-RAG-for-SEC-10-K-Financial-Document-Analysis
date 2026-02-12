"""
Section extractor for identifying and extracting key sections from 10-K filings
"""
from typing import List, Dict, Any
import re


class SectionExtractor:
    """Extract key sections from 10-K filings"""

    def __init__(self):
        """Initialize section extractor"""
        # Common 10-K section patterns
        self.section_patterns = {
            'Item 1': r'Item\s+1[.:\s]+Business',
            'Item 1A': r'Item\s+1A[.:\s]+Risk\s+Factors',
            'Item 7': r'Item\s+7[.:\s]+Management.*Discussion',
            'Item 8': r'Item\s+8[.:\s]+Financial\s+Statements',
            'Notes': r'Notes\s+to\s+(Consolidated\s+)?Financial\s+Statements'
        }

    def extract_sections(
        self,
        parsed_doc: Dict,
        target_sections: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract sections from parsed document

        Args:
            parsed_doc: Parsed document dictionary
            target_sections: List of section titles to extract (optional)

        Returns:
            List of section dictionaries
        """
        # The parsed_doc['text'] is already a list of section dicts from filing_parser
        # Just return them directly
        sections = parsed_doc.get('text', [])

        if not sections:
            return []

        # If sections is a list of strings instead of dicts, convert them
        if sections and isinstance(sections[0], str):
            sections = [{'title': f'Section {i+1}', 'content': s} for i, s in enumerate(sections)]

        return sections

    def create_section_abstract(self, section: Dict, max_words: int = 200) -> str:
        """
        Create abstract of section (title + first N words)

        Args:
            section: Section dictionary
            max_words: Maximum words for abstract

        Returns:
            Section abstract string
        """
        title = section['title']
        content = section['content']

        # Take first max_words
        words = content.split()[:max_words]
        abstract = title + ". " + " ".join(words)

        return abstract
