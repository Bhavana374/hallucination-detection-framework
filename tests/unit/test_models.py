"""
Unit tests for baseline, transformer, and hybrid model architectures.
"""

import unittest
from src.models.baselines.random_forest_baseline import TfidfRandomForestBaseline
from src.models.bert.bert_classifier import BERTClassifier
from src.models.deberta.deberta_classifier import DeBERTaClassifier
from src.models.hybrid.hybrid_classifier import HybridClassifier


class TestModels(unittest.TestCase):

    def test_random_forest_baseline(self):
        model = TfidfRandomForestBaseline(n_estimators=10)
        train_texts = ["Paris is in France", "Berlin is in Germany", "Unknown claim"]
        train_labels = [0, 0, 1]
        model.fit(train_texts, train_labels)

        probs = model.predict_proba(["Paris is in France"])
        self.assertEqual(len(probs), 1)

    def test_bert_classifier(self):
        model = BERTClassifier(num_labels=2)
        probs = model.predict_proba(["Test text"])
        self.assertEqual(probs.shape, (1, 2))

    def test_deberta_classifier(self):
        model = DeBERTaClassifier(num_labels=2)
        probs = model.predict_proba(["Test text"])
        self.assertEqual(probs.shape, (1, 2))

    def test_hybrid_classifier(self):
        model = HybridClassifier(n_estimators=10)
        train_insts = [
            {"claim": "Paris is capital", "evidence": "Paris is capital", "retrieval_score": 0.9},
            {"claim": "Moon is cheese", "evidence": "Moon is rock", "retrieval_score": 0.2}
        ]
        train_labels = [0, 1]
        model.fit(train_insts, train_labels)

        scores = model.predict_hallucination_score([{"claim": "Paris is capital", "evidence": "Paris is capital", "retrieval_score": 0.9}])
        self.assertEqual(len(scores), 1)


if __name__ == "__main__":
    unittest.main()
