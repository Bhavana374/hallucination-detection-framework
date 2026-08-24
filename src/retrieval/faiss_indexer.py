"""
FAISS Vector Indexer for Evidence Retrieval.

Provides index building, top-k vector search, and persistence. Supports
fallback to NumPy-based cosine similarity matrix lookup if faiss is not installed.
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from src.utils.logger import get_logger

logger = get_logger("faiss_indexer")

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    logger.warning("FAISS library not found. Falling back to NumPy matrix similarity search.")


class FAISSIndexer:
    """
    Vector index manager using FAISS (or NumPy fallback) for dense evidence retrieval.
    """

    def __init__(self, embedding_dim: int = 384, metric: str = "inner_product"):
        self.embedding_dim = embedding_dim
        self.metric = metric
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index = None

        if HAS_FAISS:
            if metric == "inner_product":
                self.index = faiss.IndexFlatIP(embedding_dim)
            else:
                self.index = faiss.IndexFlatL2(embedding_dim)

    def add_embeddings(self, embeddings: np.ndarray, documents: List[Dict[str, Any]]) -> None:
        """
        Add embedding vectors and associated document metadata to index.

        Args:
            embeddings: 2D float32 numpy array of shape (N, dim).
            documents: List of dict metadata matching vectors.
        """
        if len(embeddings) == 0:
            return

        embeddings = np.ascontiguousarray(embeddings.astype("float32"))
        
        # Normalize for cosine similarity if using inner product
        if self.metric == "inner_product":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            embeddings = embeddings / norms

        actual_dim = embeddings.shape[1]
        if actual_dim != self.embedding_dim:
            self.embedding_dim = actual_dim
            if HAS_FAISS:
                self.index = faiss.IndexFlatIP(actual_dim) if self.metric == "inner_product" else faiss.IndexFlatL2(actual_dim)

        if HAS_FAISS and self.index is not None:
            self.index.add(embeddings)
        
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

        self.documents.extend(documents)
        logger.info(f"Added {len(documents)} vectors to index (Total: {len(self.documents)})")

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search top_k nearest neighbors for a given query embedding vector.

        Args:
            query_vector: 1D or 2D numpy float32 array.
            top_k: Number of nearest documents to return.

        Returns:
            List of (doc_metadata, similarity_score) tuples.
        """
        if len(self.documents) == 0:
            return []

        query_vector = np.ascontiguousarray(query_vector.astype("float32"))
        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)

        # Normalize query if inner product
        if self.metric == "inner_product":
            norm = np.linalg.norm(query_vector, axis=1, keepdims=True)
            norm[norm == 0] = 1e-10
            query_vector = query_vector / norm

        top_k = min(top_k, len(self.documents))

        if HAS_FAISS and self.index is not None:
            scores, indices = self.index.search(query_vector, top_k)
            scores = scores[0]
            indices = indices[0]
        else:
            # NumPy matrix inner product fallback
            sims = np.dot(self.embeddings, query_vector.T).squeeze(1)
            indices = np.argsort(sims)[::-1][:top_k]
            scores = sims[indices]

        results = []
        for idx, score in zip(indices, scores):
            if idx >= 0 and idx < len(self.documents):
                results.append((self.documents[idx], float(score)))

        return results

    def save(self, index_path: str, metadata_path: str) -> None:
        """Save index and metadata to disk."""
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

        if HAS_FAISS and self.index is not None:
            faiss.write_index(self.index, index_path)
        else:
            np.save(index_path, self.embeddings)

        with open(metadata_path, "wb") as f:
            pickle.dump(self.documents, f)

        logger.info(f"Saved FAISS index to {index_path} and metadata to {metadata_path}")

    def load(self, index_path: str, metadata_path: str) -> None:
        """Load index and metadata from disk."""
        if HAS_FAISS and os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        elif os.path.exists(index_path + ".npy"):
            self.embeddings = np.load(index_path + ".npy")

        if os.path.exists(metadata_path):
            with open(metadata_path, "rb") as f:
                self.documents = pickle.load(f)

        logger.info(f"Loaded index from {index_path} with {len(self.documents)} documents.")
