"""
Hierarchical retriever implementing two-stage retrieval (sections -> content)
"""
from typing import List, Dict, Any
import numpy as np
import faiss


class HierarchicalRetriever:
    """Two-stage hierarchical retrieval system"""

    def __init__(
        self,
        section_index: faiss.Index,
        text_index: faiss.Index,
        table_index: faiss.Index,
        section_data: Dict,
        text_data: Dict,
        table_data: Dict,
        embedding_model
    ):
        """
        Initialize hierarchical retriever

        Args:
            section_index: FAISS index for section abstracts
            text_index: FAISS index for text chunks
            table_index: FAISS index for table sentences
            section_data: Section content and metadata
            text_data: Text chunk content and metadata
            table_data: Table sentence content and metadata
            embedding_model: Sentence transformer model
        """
        self.section_index = section_index
        self.text_index = text_index
        self.table_index = table_index

        self.section_data = section_data
        self.text_data = text_data
        self.table_data = table_data

        self.embedding_model = embedding_model

    def retrieve_sections(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Stage A: Retrieve relevant sections

        Args:
            query: User query
            k: Number of sections to retrieve

        Returns:
            List of section results with metadata
        """
        # Encode query
        query_embedding = self.embedding_model.encode([query])[0]

        # Search section index
        distances, indices = self.section_index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            k
        )

        # Compile results
        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                'section_abstract': self.section_data['abstracts'][idx],
                'metadata': self.section_data['metadata'][idx],
                'score': 1.0 / (1.0 + distances[0][i])  # Convert distance to similarity
            })

        return results

    def retrieve(
        self,
        query: str,
        route_info: Dict,
        top_k_sections: int = 5,
        top_k_content: int = 10,
        use_hybrid: bool = True
    ) -> Dict[str, Any]:
        """
        Full hierarchical retrieval pipeline

        Args:
            query: User query
            route_info: Query routing information
            top_k_sections: Number of sections to retrieve (Stage A)
            top_k_content: Number of content pieces to retrieve (Stage B)
            use_hybrid: Whether to use hybrid (dense + BM25) search

        Returns:
            Dictionary with retrieved content and metadata
        """
        # Stage A: Retrieve relevant sections
        sections = self.retrieve_sections(query, k=top_k_sections)

        # Get section identifiers for filtering
        section_identifiers = [
            (s['metadata']['ticker'], s['metadata']['fiscal_year'], s['metadata']['section_title'])
            for s in sections
        ]

        # Stage B: Retrieve content within selected sections
        if route_info['is_table_centric']:
            # Prioritize table retrieval
            content = self._retrieve_tables(
                query,
                section_identifiers,
                k=top_k_content,
                use_hybrid=use_hybrid
            )
        else:
            # Retrieve text content
            content = self._retrieve_text(
                query,
                section_identifiers,
                k=top_k_content,
                use_hybrid=use_hybrid
            )

        return {
            'sections': sections,
            'content': content
        }

    def _retrieve_tables(
        self,
        query: str,
        section_filter: List,
        k: int,
        use_hybrid: bool
    ) -> List[Dict[str, Any]]:
        """Retrieve table sentences"""
        # Check if table sentences exist
        if not self.table_data.get('sentences') or len(self.table_data['sentences']) == 0:
            # Fall back to text retrieval if no table data
            return self._retrieve_text(query, section_filter, k, use_hybrid)

        # Encode query
        query_embedding = self.embedding_model.encode([query])[0]

        # Dense search
        distances, indices = self.table_index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            k * 2  # Retrieve more for filtering
        )

        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= len(self.table_data['sentences']):
                continue

            metadata = self.table_data['metadata'][idx]

            # Filter by section (if applicable)
            # Simplified - actual implementation would check section match

            results.append({
                'content': self.table_data['sentences'][idx],
                'metadata': metadata,
                'score': 1.0 / (1.0 + distances[0][i])
            })

        # Optionally add BM25 results
        if use_hybrid and 'bm25' in self.table_data and self.table_data['bm25'] is not None:
            bm25_results = self._bm25_search(
                query,
                self.table_data['bm25'],
                self.table_data['sentences'],
                self.table_data['metadata'],
                k=k
            )
            results = self._merge_results(results, bm25_results)

        return results[:k]

    def _retrieve_text(
        self,
        query: str,
        section_filter: List,
        k: int,
        use_hybrid: bool
    ) -> List[Dict[str, Any]]:
        """Retrieve text chunks"""
        # Similar to _retrieve_tables but for text
        query_embedding = self.embedding_model.encode([query])[0]

        distances, indices = self.text_index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            k * 2
        )

        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                'content': self.text_data['chunks'][idx],
                'metadata': self.text_data['metadata'][idx],
                'score': 1.0 / (1.0 + distances[0][i])
            })

        if use_hybrid and 'bm25' in self.text_data:
            bm25_results = self._bm25_search(
                query,
                self.text_data['bm25'],
                self.text_data['chunks'],
                self.text_data['metadata'],
                k=k
            )
            results = self._merge_results(results, bm25_results)

        return results[:k]

    def _bm25_search(self, query: str, bm25_index, documents: List, metadata: List, k: int) -> List[Dict]:
        """BM25 keyword search"""
        query_tokens = query.lower().split()
        scores = bm25_index.get_scores(query_tokens)

        # Get top k
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            results.append({
                'content': documents[idx],
                'metadata': metadata[idx],
                'score': scores[idx]
            })

        return results

    def _merge_results(self, dense_results: List, bm25_results: List, alpha: float = 0.7) -> List[Dict]:
        """Merge dense and BM25 results with weighted fusion"""
        # Simplified reciprocal rank fusion
        merged = {}

        for i, result in enumerate(dense_results):
            key = result['content'][:50]  # Use content prefix as key
            merged[key] = {
                **result,
                'score': alpha * result['score']
            }

        for i, result in enumerate(bm25_results):
            key = result['content'][:50]
            if key in merged:
                merged[key]['score'] += (1 - alpha) * result['score']
            else:
                merged[key] = {
                    **result,
                    'score': (1 - alpha) * result['score']
                }

        # Sort by merged score
        sorted_results = sorted(merged.values(), key=lambda x: x['score'], reverse=True)

        return sorted_results
