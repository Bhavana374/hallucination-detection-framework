"""
Unit tests for FAISSIndexer and EvidenceRetriever.
"""

import unittest
import numpy as np
from src.retrieval.faiss_indexer import FAISSIndexer
from src.retrieval.retrieval_engine import EvidenceRetriever


class TestRetrieval(unittest.TestCase):

    def test_faiss_indexer(self):
        indexer = FAISSIndexer(embedding_dim=4, metric="inner_product")
        vectors = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float32)
        docs = [{"doc_id": "d1", "text": "doc1"}, {"doc_id": "d2", "text": "doc2"}]

        indexer.add_embeddings(vectors, docs)
        results = indexer.search(np.array([1.0, 0.0, 0.0, 0.0]), top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0]["doc_id"], "d1")
        self.assertGreater(results[0][1], 0.8)

    def test_evidence_retriever(self):
        retriever = EvidenceRetriever(embedding_dim=16, top_k=2)
        documents = [
            {"doc_id": "d1", "text": "Paris is the capital of France. It has many museums."},
            {"doc_id": "d2", "text": "Tokyo is the capital of Japan."}
        ]
        retriever.index_documents(documents)

        results = retriever.retrieve_evidence("Tell me about Paris capital", top_k=1)
        self.assertGreaterEqual(len(results), 1)
        self.assertIn("text", results[0])


if __name__ == "__main__":
    unittest.main()
