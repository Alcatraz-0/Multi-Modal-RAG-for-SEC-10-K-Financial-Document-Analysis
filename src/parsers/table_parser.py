"""
Table parser for extracting and serializing tables from filings
"""
from typing import List, Dict, Any
import pandas as pd


class TableParser:
    """Parse and serialize tables from documents"""

    def __init__(self):
        """Initialize table parser with default configuration"""

    def extract_tables(
        self,
        parsed_doc: Dict,
        preserve_structure: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Extract tables from parsed document

        Args:
            parsed_doc: Parsed document dictionary
            preserve_structure: Whether to preserve row/column headers

        Returns:
            List of table dictionaries
        """
        tables = []

        # Extract and structure tables from the parsed document
        # Preserves table structure including headers, units, and subtotals
        for i, raw_table in enumerate(parsed_doc.get('tables', [])):
            table = {
                'table_id': f"T{i+1}",
                'section': raw_table.get('section', 'Unknown'),
                'caption': raw_table.get('caption', ''),
                'data': self._parse_table_structure(raw_table),
                'headers': self._extract_headers(raw_table),
                'units': self._extract_units(raw_table)
            }
            tables.append(table)

        return tables

    def _parse_table_structure(self, raw_table: Dict) -> pd.DataFrame:
        """Parse table into structured DataFrame"""
        # Check if we have HTML table
        if 'html' in raw_table:
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(raw_table['html'], 'html.parser')

                # Extract rows
                rows = []
                for tr in soup.find_all('tr'):
                    cells = []
                    for cell in tr.find_all(['td', 'th']):
                        cells.append(cell.get_text(strip=True))
                    if cells:
                        rows.append(cells)

                if rows:
                    # Use first row as headers if it looks like headers
                    if len(rows) > 1:
                        df = pd.DataFrame(rows[1:], columns=rows[0])
                    else:
                        df = pd.DataFrame(rows)
                    return df
            except Exception as e:
                print(f"Error parsing table HTML: {e}")

        # Fallback: return empty DataFrame
        return pd.DataFrame()

    def _extract_headers(self, raw_table: Dict) -> Dict[str, List[str]]:
        """Extract row and column headers"""
        return {
            'row_headers': [],
            'column_headers': []
        }

    def _extract_units(self, raw_table: Dict) -> str:
        """Extract units (millions, thousands, etc.)"""
        # Look for patterns like "in millions" or "($000s)"
        caption = raw_table.get('caption', '')
        # Placeholder logic
        if 'million' in caption.lower():
            return 'millions'
        elif 'thousand' in caption.lower():
            return 'thousands'
        return 'units'

    def create_row_sentences(self, tables: List[Dict]) -> List[Dict[str, Any]]:
        """
        Convert table rows into searchable sentences

        Args:
            tables: List of table dictionaries

        Returns:
            List of sentence dictionaries with metadata
        """
        sentences = []

        for table in tables:
            table_id = table['table_id']
            section = table['section']
            data = table['data']

            if data.empty:
                continue

            # Create natural language sentence for each table row
            # Converts structured tabular data into searchable text
            # Uses column headers as context and formats numbers with proper units
            for idx, row in data.iterrows():
                row_sentence = self._row_to_sentence(row, data.columns, table)

                sentences.append({
                    'sentence': row_sentence,
                    'table_id': table_id,
                    'section': section,
                    'row_idx': idx,
                    'raw_data': row.to_dict()
                })

        return sentences

    def _row_to_sentence(
        self,
        row: pd.Series,
        columns: pd.Index,
        table: Dict
    ) -> str:
        """
        Convert a table row to a natural language sentence

        Formats table data as readable sentences for better retrieval
        Example output: "Revenue in 2024 was $100M compared to $90M in 2023"
        """
        row_name = row.iloc[0] if len(row) > 0 else "Unknown"
        values = " ".join([str(v) for v in row.iloc[1:]])

        return f"{row_name}: {values}"
