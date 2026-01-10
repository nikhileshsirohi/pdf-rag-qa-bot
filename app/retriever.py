"""
FAISS stores only embeddings.
FAISS automatically assigns IDs: 0, 1, 2, … in the order you add vectors.

During search:
	•	FAISS returns (scores, indices)
	•	indices[i] tells you which embedding matched
	•	You use that index to fetch the corresponding text from self.chunks

"""

# import faiss
# import numpy as np

# class FAISSRetriever:
#     """
#     This class is responsible for:
#     1. Storing document chunk embeddings
#     2. Searching the most similar chunks for a query
#     """
#     def __init__(self, embedding_dim: int):
#         """
#         Initialize FAISS index.

#         Args:
#             embedding_dim (int): dimension of embedding vectors
#                                  (e.g. 384 for MiniLM)
#         """
#         self.index = faiss.IndexFlatIP(embedding_dim)
#         # IndexFlatIP:
#         # - Flat = exact search (no approximation)
#         # - IP = Inner Product
#         # After normalization, inner product == cosine similarity

#         self.chunks = []
#         # This list stores the original text chunks
#         # Index of this list MUST match FAISS vector index

#     def add_embeddings(self, embeddings: np.ndarray, chunks: list[str]):
#         """
#         Add document embeddings and their corresponding text chunks
#         into FAISS.

#         Args:
#             embeddings (np.ndarray): shape (num_chunks, embedding_dim)
#             chunks (list[str]): original text chunks
#         """
#         # IMPORTANT:
#         # We normalize embeddings so cosine similarity works correctly
#         # Without this, similarity scores will be incorrect
#         faiss.normalize_L2(embeddings)

#         self.index.add(embeddings) # Add vectors to FAISS index
#         self.chunks.extend(chunks) # Store corresponding text chunks
#         # The order MUST match FAISS index order

#     def search(self, query_embedding: np.ndarray, top_k: int = 3):
#         """
#         Search FAISS index using query embedding.

#         Args:
#             query_embedding (np.ndarray): shape (1, embedding_dim)
#             top_k (int): number of best matches to retrieve

#         Returns:
#             List of dictionaries containing:
#             - similarity score
#             - corresponding text chunk
#         """
#         faiss.normalize_L2(query_embedding)

#         scores, indices = self.index.search(query_embedding, top_k)
#         # Perform FAISS search
#         # scores: similarity values
#         # indices: index positions of matched vectors
#         # print(f"Scores: {scores}") # Scores: [[0.42406178 0.33950064 0.3065765 ]]
#         # print(f"Indices: {indices}") # Indices: [[17 16 18]]
#         results = []
#         for score, idx in zip(scores[0], indices[0]):
#             if idx < len(self.chunks):
#                 results.append(
#                     {
#                         "score": float(score),
#                         "text": self.chunks[idx]
#                     }
#                 )

#         return results


import faiss
import numpy as np
import pickle
import os


class FAISSRetriever:
    def __init__(self, embedding_dim: int, index_path: str = "vector_store/index.faiss", meta_path: str = "vector_store/chunks.pkl"):
        self.embedding_dim = embedding_dim
        self.index_path = index_path
        self.meta_path = meta_path

        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            # Load existing index
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "rb") as f:
                self.chunks = pickle.load(f)
        else:
            # Create new index
            self.index = faiss.IndexFlatIP(embedding_dim)
            self.chunks = []

    def add_embeddings(self, embeddings: np.ndarray, chunks: list[str]):
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def save(self):
        """
        Persist FAISS index and chunks to disk.
        """
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        faiss.write_index(self.index, self.index_path)

        with open(self.meta_path, "wb") as f:
            pickle.dump(self.chunks, f)

    def search(self, query_embedding: np.ndarray, top_k: int = 2):
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                results.append({
                    "score": float(score),
                    "text": self.chunks[idx]
                })
        return results
