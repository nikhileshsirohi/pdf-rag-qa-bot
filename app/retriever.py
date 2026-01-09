"""
We will:
use FAISS IndexFlatIP (cosine similarity)
normalize embeddings
store:
    FAISS index
    chunk texts
    metadata (optional later) 
"""

import faiss
import numpy as np


class FAISSRetriever:
    """
    This class is responsible for:
    1. Storing document chunk embeddings
    2. Searching the most similar chunks for a query
    """
    def __init__(self, embedding_dim: int):
        """
        Initialize FAISS index.

        Args:
            embedding_dim (int): dimension of embedding vectors
                                 (e.g. 384 for MiniLM)
        """
        self.index = faiss.IndexFlatIP(embedding_dim)
        # IndexFlatIP:
        # - Flat = exact search (no approximation)
        # - IP = Inner Product
        # After normalization, inner product == cosine similarity

        self.chunks = []
        # This list stores the original text chunks
        # Index of this list MUST match FAISS vector index

    def add_embeddings(self, embeddings: np.ndarray, chunks: list[str]):
        """
        Add document embeddings and their corresponding text chunks
        into FAISS.

        Args:
            embeddings (np.ndarray): shape (num_chunks, embedding_dim)
            chunks (list[str]): original text chunks
        """
        # IMPORTANT:
        # We normalize embeddings so cosine similarity works correctly
        # Without this, similarity scores will be incorrect
        faiss.normalize_L2(embeddings)

        self.index.add(embeddings) # Add vectors to FAISS index
        self.chunks.extend(chunks) # Store corresponding text chunks
        # The order MUST match FAISS index order

    def search(self, query_embedding: np.ndarray, top_k: int = 3):
        """
        Search FAISS index using query embedding.

        Args:
            query_embedding (np.ndarray): shape (1, embedding_dim)
            top_k (int): number of best matches to retrieve

        Returns:
            List of dictionaries containing:
            - similarity score
            - corresponding text chunk
        """
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)
        # Perform FAISS search
        # scores: similarity values
        # indices: index positions of matched vectors

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                results.append(
                    {
                        "score": float(score),
                        "text": self.chunks[idx]
                    }
                )

        return results
