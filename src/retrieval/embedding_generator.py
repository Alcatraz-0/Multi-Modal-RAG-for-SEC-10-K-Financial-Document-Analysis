"""
Embedding generator using sentence transformers
"""
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class EmbeddingGenerator:
    """Generate embeddings using sentence transformers"""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        batch_size: int = 32,
        device: str = 'cpu'
    ):
        """
        Initialize embedding generator

        Args:
            model_name: Name of sentence transformer model
            batch_size: Batch size for encoding
            device: Device to use ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device

        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        print(f"Model loaded on {device}")

    def encode(
        self,
        texts: List[str],
        show_progress: bool = False,
        convert_to_numpy: bool = True
    ) -> np.ndarray:
        """
        Encode texts to embeddings

        Args:
            texts: List of text strings to encode
            show_progress: Show progress bar
            convert_to_numpy: Convert to numpy array

        Returns:
            Array of embeddings
        """
        if not texts:
            return np.array([])

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=convert_to_numpy
        )

        return embeddings

    def encode_queries(
        self,
        queries: List[str]
    ) -> np.ndarray:
        """
        Encode queries (alias for encode)

        Args:
            queries: List of query strings

        Returns:
            Array of query embeddings
        """
        return self.encode(queries, show_progress=False)

    def get_embedding_dimension(self) -> int:
        """Get dimension of embeddings"""
        return self.model.get_sentence_embedding_dimension()
