"""
Controlled scenario test fixtures for end-to-end pipeline verification.
"""

import unittest
from src.pipeline.hallucination_detector import HallucinationDetector


class TestControlledScenarios(unittest.TestCase):

    def setUp(self):
        self.detector = HallucinationDetector(score_threshold=0.50, retrieval_threshold=0.30)
        self.context = (
            "Apollo 11 was the American spaceflight that first landed humans on the Moon. "
            "Commander Neil Armstrong and Lunar Module Pilot Buzz Aldrin landed the Apollo Lunar Module Eagle on July 20, 1969. "
            "Water boils at 100 degrees Celsius at sea level."
        )

    def test_scenario_1_clearly_factual(self):
        """Test 1: Clearly factual response."""
        res = self.detector.analyze_response(
            query="When did Apollo 11 land on the Moon?",
            generated_response="Apollo 11 landed humans on the Moon on July 20, 1969.",
            reference_context=self.context
        )
        self.assertEqual(res["total_claims"], 1)
        self.assertIn(res["claim_breakdown"][0]["verdict"], ["SUPPORTED", "Factual"])

    def test_scenario_2_clearly_contradictory(self):
        """Test 2: Clearly contradictory statement."""
        res = self.detector.analyze_response(
            query="Where did Apollo 11 land?",
            generated_response="Apollo 11 landed on Mars in October 1975.",
            reference_context=self.context
        )
        self.assertGreaterEqual(res["contradicted_claims_count"] + res["insufficient_evidence_claims_count"], 1)

    def test_scenario_3_mixed_factual_and_false(self):
        """Test 3: Response containing one factual and one false claim."""
        res = self.detector.analyze_response(
            query="Tell me about water and Apollo 11.",
            generated_response="Water boils at 100 degrees Celsius. Apollo 11 was launched by ancient Romans.",
            reference_context=self.context
        )
        self.assertEqual(res["total_claims"], 2)

    def test_scenario_4_evidence_unavailable(self):
        """Test 4: Claim for which evidence is unavailable."""
        res = self.detector.analyze_response(
            query="Tell me about quantum computing.",
            generated_response="Quantum computers use qubits for superposition.",
            reference_context="The Eiffel Tower is in Paris, France."
        )
        self.assertEqual(res["claim_breakdown"][0]["verdict"], "INSUFFICIENT_EVIDENCE")

    def test_scenario_5_multiple_claims(self):
        """Test 5: Multiple claims response."""
        res = self.detector.analyze_response(
            query="What is water boiling point and Apollo 11?",
            generated_response="Water boils at 100 degrees Celsius. Neil Armstrong was the commander of Apollo 11.",
            reference_context=self.context
        )
        self.assertEqual(res["total_claims"], 2)


if __name__ == "__main__":
    unittest.main()
