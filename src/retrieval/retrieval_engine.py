"""
Evidence Retrieval Engine.

Encodes reference documents/context passages into dense vectors and retrieves
top-k relevant evidence sentences/passages for target claims using FAISSIndexer.
"""

import re
import numpy as np
from typing import List, Dict, Any, Optional
from src.retrieval.faiss_indexer import FAISSIndexer
from src.utils.logger import get_logger

logger = get_logger("retrieval_engine")

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    logger.warning("sentence-transformers package not found. Using fallback lexical dense encoder.")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class FallbackLexicalEncoder:
    """Fallback TF-IDF lexical dense encoder when sentence-transformers is unavailable."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.vectorizer = TfidfVectorizer(max_features=dim, stop_words="english") if HAS_SKLEARN else None
        self.is_fitted = False

    def encode(self, texts: List[str], normalize_embeddings: bool = True) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        if HAS_SKLEARN and self.vectorizer is not None:
            if not self.is_fitted:
                self.vectorizer.fit(texts)
                self.is_fitted = True

            sparse_matrix = self.vectorizer.transform(texts)
            arr = sparse_matrix.toarray().astype(np.float32)
            if arr.shape[1] < self.dim:
                pad = np.zeros((arr.shape[0], self.dim - arr.shape[1]), dtype=np.float32)
                arr = np.hstack([arr, pad])
            elif arr.shape[1] > self.dim:
                arr = arr[:, :self.dim]
        else:
            # Fallback simple char/word hash embedding
            arr = np.zeros((len(texts), self.dim), dtype=np.float32)
            for i, text in enumerate(texts):
                words = text.lower().split()
                for w in words:
                    idx = abs(hash(w)) % self.dim
                    arr[i, idx] += 1.0

        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            arr = arr / norms

        return arr.astype(np.float32)


class EvidenceRetriever:
    """
    Evidence Retrieval engine for segmenting context documents into evidence chunks,
    encoding them, and finding relevant evidence for claim queries.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        top_k: int = 3,
        score_threshold: float = 0.35,
    ):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.indexer = FAISSIndexer(embedding_dim=embedding_dim, metric="inner_product")

        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.model = SentenceTransformer(model_name)
                logger.info(f"Loaded SentenceTransformer model: {model_name}")
            except Exception as e:
                logger.warning(f"Could not load SentenceTransformer '{model_name}': {e}. Using fallback encoder.")
                self.model = FallbackLexicalEncoder(dim=embedding_dim)
        else:
            self.model = FallbackLexicalEncoder(dim=embedding_dim)

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split document text into individual sentence chunks."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode list of strings into float32 embedding matrix."""
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        if HAS_SENTENCE_TRANSFORMERS and isinstance(self.model, SentenceTransformer):
            embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return embeddings.astype(np.float32)
        else:
            return self.model.encode(texts, normalize_embeddings=True)

    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Segment documents into sentence chunks and add them to the vector index.

        Args:
            documents: List of dicts with keys 'doc_id', 'text', and optional metadata.
        """
        all_chunks = []
        metadata_list = []

        for doc in documents:
            doc_id = doc.get("doc_id", "doc_0")
            full_text = doc.get("text", "")
            sentences = self._split_into_sentences(full_text)
            if not sentences:
                sentences = [full_text]

            for idx, sent in enumerate(sentences):
                all_chunks.append(sent)
                metadata_list.append({
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}_chunk_{idx}",
                    "text": sent,
                    "full_context": full_text
                })

        if not all_chunks:
            return

        embeddings = self.encode(all_chunks)
        self.indexer.add_embeddings(embeddings, metadata_list)
        logger.info(f"Indexed {len(all_chunks)} evidence sentence chunks from {len(documents)} documents.")

    def retrieve_evidence(self, query_claim: str, top_k: Optional[int] = None, claim_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve top_k evidence passages for a query claim.

        Args:
            query_claim: Sentence or claim string to find evidence for.
            top_k: Optional top_k override.
            claim_id: Optional claim ID string.

        Returns:
            List of structured evidence objects containing claim_id, evidence_id, text, similarity_score.
        """
        k = top_k if top_k is not None else self.top_k
        query_vec = self.encode([query_claim])
        raw_results = self.indexer.search(query_vec, top_k=k)

        evidences = []
        for doc_meta, score in raw_results:
            chunk_id = doc_meta.get("chunk_id", "ev_001")
            evidences.append({
                "claim_id": claim_id or "claim_001",
                "evidence_id": chunk_id,
                "chunk_id": chunk_id,
                "doc_id": doc_meta.get("doc_id", "doc_0"),
                "text": doc_meta.get("text", ""),
                "score": float(score),
                "similarity_score": float(score),
                "is_above_threshold": float(score) >= self.score_threshold
            })

        return evidences
