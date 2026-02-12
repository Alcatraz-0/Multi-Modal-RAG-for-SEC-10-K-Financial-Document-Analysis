"""
Hybrid search combining dense and sparse retrieval
"""
from typing import List, Dict, Any
import numpy as np


class HybridSearcher:
    """Combine dense and BM25 search results"""

    def __init__(
        self,
        dense_weight: float = 0.7,
        bm25_weight: float = 0.3
    ):
        """
        Initialize hybrid searcher

        Args:
            dense_weight: Weight for dense retrieval scores
            bm25_weight: Weight for BM25 scores
        """
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

        # Weights should sum to 1
        total = dense_weight + bm25_weight
        self.dense_weight /= total
        self.bm25_weight /= total

    def fuse_results(
        self,
        dense_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Fuse dense and BM25 results

        Args:
            dense_results: Results from dense retrieval
            bm25_results: Results from BM25
            top_k: Number of results to return

        Returns:
            Fused and ranked results
        """
        # Use reciprocal rank fusion
        scores = {}

        # Process dense results
        for rank, result in enumerate(dense_results):
            doc_id = result.get('id', str(result))
            scores[doc_id] = self.dense_weight / (rank + 1)

        # Process BM25 results
        for rank, result in enumerate(bm25_results):
            doc_id = result.get('id', str(result))
            if doc_id in scores:
                scores[doc_id] += self.bm25_weight / (rank + 1)
            else:
                scores[doc_id] = self.bm25_weight / (rank + 1)

        # Sort by combined score
        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Return top k
        return sorted_ids[:top_k]
