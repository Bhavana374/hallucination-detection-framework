"""
Unit tests for multi-dimensional feature extractors and FeatureVectorizer.
"""

import unittest
import numpy as np
from src.features.nli_features import NLIFeatureExtractor
from src.features.semantic_features import SemanticFeatureExtractor
from src.features.factual_features import FactualFeatureExtractor
from src.features.linguistic_features import LinguisticFeatureExtractor
from src.features.feature_vectorizer import FeatureVectorizer


class TestFeatures(unittest.TestCase):

    def test_nli_features(self):
        extractor = NLIFeatureExtractor()
        res = extractor.extract_features("Paris is in France", "Paris is the capital of France.")
        self.assertIn("nli_prob_entailment", res)
        self.assertIn("nli_prob_contradiction", res)

    def test_semantic_features(self):
        extractor = SemanticFeatureExtractor()
        res = extractor.extract_features("The sun is hot", "The sun is warm and hot.")
        self.assertIn("semantic_cosine_sim", res)

    def test_factual_features(self):
        extractor = FactualFeatureExtractor()
        res = extractor.extract_features("Einstein was born in 1879.", "Albert Einstein was born in Ulm in 1879.")
        self.assertGreater(res["entity_overlap_ratio"], 0.0)
        self.assertEqual(res["number_match_ratio"], 1.0)

    def test_linguistic_features(self):
        extractor = LinguisticFeatureExtractor()
        res = extractor.extract_features("Perhaps it might rain", "Rain is expected.")
        self.assertGreaterEqual(res["hedge_word_count"], 1.0)

    def test_feature_vectorizer(self):
        vectorizer = FeatureVectorizer()
        samples = [
            {"claim": "Paris is in France.", "evidence": "Paris is in France.", "retrieval_score": 0.9},
            {"claim": "Tokyo is in Mars.", "evidence": "Tokyo is in Japan.", "retrieval_score": 0.2}
        ]
        X = vectorizer.fit_transform(samples)
        self.assertEqual(X.shape[0], 2)
        self.assertGreater(X.shape[1], 5)


if __name__ == "__main__":
    unittest.main()
