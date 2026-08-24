"""
Unit tests for end-to-end HallucinationDetector and ExplainabilityEngine.
"""

import unittest
from src.pipeline.hallucination_detector import HallucinationDetector
from src.explainability.explainability_engine import ExplainabilityEngine


class TestPipeline(unittest.TestCase):

    def test_explainability_engine(self):
        engine = ExplainabilityEngine()
        feats = {
            "nli_prob_entailment": 0.1,
            "nli_prob_contradiction": 0.8,
            "semantic_cosine_sim": 0.2,
            "entity_overlap_ratio": 0.3
        }
        res = engine.generate_claim_explanation(
            claim="Mars is inhabited by dinosaurs.",
            evidence="Mars is a cold barren planet.",
            feature_dict=feats,
            hallucination_prob=0.85,
            verdict="Hallucinated"
        )
        self.assertEqual(res["verdict"], "Hallucinated")
        self.assertGreater(len(res["risk_triggers"]), 0)

    def test_hallucination_detector_pipeline(self):
        detector = HallucinationDetector()
        res = detector.analyze_response(
            query="Where is Paris?",
            generated_response="Paris is the capital of France. It is located on Mars.",
            reference_context="Paris is the capital and largest city of France located in Western Europe."
        )
        self.assertIn("overall_hallucination_score", res)
        self.assertIn("claim_breakdown", res)
        self.assertEqual(res["total_claims"], 2)


if __name__ == "__main__":
    unittest.main()
