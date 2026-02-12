"""
Text chunker for splitting documents into searchable chunks
"""
from typing import List, Dict, Any
import re


class TextChunker:
    """Chunk text into overlapping segments"""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        respect_sentence_boundaries: bool = True
    ):
        """
        Initialize text chunker

        Args:
            chunk_size: Target chunk size in tokens (approximate)
            chunk_overlap: Number of overlapping tokens between chunks
            respect_sentence_boundaries: Try to break at sentence boundaries
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.respect_sentence_boundaries = respect_sentence_boundaries

    def chunk_text(
        self,
        text: str,
        metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk text into segments

        Args:
            text: Input text to chunk
            metadata: Metadata to attach to each chunk

        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text or len(text.strip()) == 0:
            return []

        # Simple word-based chunking (approximates tokens)
        words = text.split()
        chunks = []

        start_idx = 0
        chunk_id = 0

        while start_idx < len(words):
            # Get chunk of words
            end_idx = min(start_idx + self.chunk_size, len(words))
            chunk_words = words[start_idx:end_idx]

            # If respecting sentence boundaries, try to end at sentence
            if self.respect_sentence_boundaries and end_idx < len(words):
                chunk_text = ' '.join(chunk_words)

                # Find last sentence boundary
                last_period = chunk_text.rfind('.')
                last_question = chunk_text.rfind('?')
                last_exclamation = chunk_text.rfind('!')

                last_boundary = max(last_period, last_question, last_exclamation)

                if last_boundary > len(chunk_text) * 0.5:  # At least 50% through
                    # Adjust to end at sentence
                    chunk_text = chunk_text[:last_boundary + 1]
                    chunk_words = chunk_text.split()

            chunk_text = ' '.join(chunk_words)

            # Create chunk object
            chunk = {
                'text': chunk_text,
                'chunk_id': chunk_id,
                'start_word_idx': start_idx,
                'end_word_idx': start_idx + len(chunk_words),
                'metadata': metadata if metadata else {}
            }

            chunks.append(chunk)

            # Move forward with overlap
            start_idx += self.chunk_size - self.chunk_overlap
            chunk_id += 1

        return chunks

    def chunk_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents

        Args:
            documents: List of document dictionaries with 'text' and 'metadata'

        Returns:
            List of all chunks across documents
        """
        all_chunks = []

        for doc in documents:
            text = doc.get('text', '')
            metadata = doc.get('metadata', {})

            chunks = self.chunk_text(text, metadata)
            all_chunks.extend(chunks)

        return all_chunks
