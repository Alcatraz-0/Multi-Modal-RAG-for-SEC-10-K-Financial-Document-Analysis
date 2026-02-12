"""
Index builder for creating FAISS and BM25 indices
"""
from typing import List, Dict, Any
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
import pickle
from pathlib import Path


class IndexBuilder:
    """Build and manage search indices"""

    def __init__(self):
        """Initialize index builder for FAISS and BM25 indices"""

    def build_faiss_index(
        self,
        embeddings: np.ndarray,
        index_type: str = "flat"
    ) -> faiss.Index:
        """
        Build FAISS index from embeddings

        Args:
            embeddings: Array of embeddings (N x D)
            index_type: Type of index ('flat', 'ivf', 'hnsw')

        Returns:
            FAISS index
        """
        dimension = embeddings.shape[1]

        if index_type == "flat":
            # Simple flat L2 index
            index = faiss.IndexFlatL2(dimension)
        elif index_type == "ivf":
            # IVF index for larger datasets
            quantizer = faiss.IndexFlatL2(dimension)
            nlist = min(100, embeddings.shape[0] // 10)  # Number of clusters
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            index.train(embeddings.astype('float32'))
        elif index_type == "hnsw":
            # HNSW index for fast approximate search
            index = faiss.IndexHNSWFlat(dimension, 32)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        # Add vectors to index
        index.add(embeddings.astype('float32'))

        return index

    def build_bm25_index(
        self,
        documents: List[str],
        tokenizer=None
    ) -> BM25Okapi:
        """
        Build BM25 index from documents

        Args:
            documents: List of document strings
            tokenizer: Optional tokenizer function

        Returns:
            BM25Okapi index
        """
        if tokenizer is None:
            # Simple whitespace tokenizer
            tokenizer = lambda x: x.lower().split()

        tokenized_docs = [tokenizer(doc) for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)

        return bm25

    def save_index(
        self,
        index: faiss.Index,
        path: Path
    ):
        """
        Save FAISS index to disk

        Args:
            index: FAISS index
            path: Path to save to
        """
        faiss.write_index(index, str(path))

    def load_index(
        self,
        path: Path
    ) -> faiss.Index:
        """
        Load FAISS index from disk

        Args:
            path: Path to load from

        Returns:
            FAISS index
        """
        return faiss.read_index(str(path))

    def save_bm25_index(
        self,
        bm25: BM25Okapi,
        path: Path
    ):
        """
        Save BM25 index to disk

        Args:
            bm25: BM25 index
            path: Path to save to
        """
        with open(path, 'wb') as f:
            pickle.dump(bm25, f)

    def load_bm25_index(
        self,
        path: Path
    ) -> BM25Okapi:
        """
        Load BM25 index from disk

        Args:
            path: Path to load from

        Returns:
            BM25 index
        """
        with open(path, 'rb') as f:
            return pickle.load(f)
