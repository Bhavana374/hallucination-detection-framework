"""
End-to-End Hallucination Detection Pipeline.

Executes response segmentation, claim extraction, evidence retrieval,
multi-dimensional feature extraction, hybrid classification, 3-state claim-level verdicts
(SUPPORTED, CONTRADICTED, INSUFFICIENT_EVIDENCE), and configurable response-level hallucination scoring.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np

from src.data.claim_extractor import ClaimExtractor, StructuredClaim
from src.retrieval.retrieval_engine import EvidenceRetriever
from src.features.feature_vectorizer import FeatureVectorizer
from src.models.hybrid.hybrid_classifier import HybridClassifier
from src.explainability.explainability_engine import ExplainabilityEngine
from src.utils.logger import get_logger

logger = get_logger("hallucination_detector")


class HallucinationDetector:
    """
    End-to-end Hallucination Detection Engine.
    """

    def __init__(
        self,
        retriever: Optional[EvidenceRetriever] = None,
        vectorizer: Optional[FeatureVectorizer] = None,
        model: Optional[Any] = None,
        claim_extractor: Optional[ClaimExtractor] = None,
        score_threshold: float = 0.50,
        retrieval_threshold: float = 0.30,
        aggregation_strategy: str = "max_risk"
    ):
        self.retriever = retriever or EvidenceRetriever()
        self.vectorizer = vectorizer or FeatureVectorizer()
        self.model = model or HybridClassifier(vectorizer=self.vectorizer)
        self.claim_extractor = claim_extractor or ClaimExtractor()
        self.explainability_engine = ExplainabilityEngine()
        self.score_threshold = score_threshold
        self.retrieval_threshold = retrieval_threshold
        self.aggregation_strategy = aggregation_strategy

        # Pre-fit default state if model isn't fitted yet
        if hasattr(self.model, "is_fitted") and not self.model.is_fitted:
            self._initialize_default_state()

    def _initialize_default_state(self):
        """Fit model on synthetic default grounding benchmark data for ready-to-use pipeline."""
        from src.data.sample_generator import generate_synthetic_halueval_dataset
        records = generate_synthetic_halueval_dataset(num_samples=40)
        training_insts = []
        training_labels = []
        for r in records:
            lbl = r.get("binary_label", 0)
            text = r.get("right_answer", "") if lbl == 0 else r.get("hallucinated_answer", "")
            evid = r.get("knowledge_context", r.get("evidence_text", ""))
            training_insts.append({
                "claim": text,
                "evidence": evid,
                "retrieval_score": 0.90 if lbl == 0 else 0.25
            })
            training_labels.append(lbl)

        self.model.fit(training_insts, training_labels)

    def extract_claims(self, text: str) -> List[StructuredClaim]:
        """Segment generated response text into structured claim objects."""
        return self.claim_extractor.extract_claims(text)

    def determine_claim_verdict(self, hallucination_prob: float, retrieval_score: float, evidence_text: str) -> str:
        """
        Determine 3-state claim verdict.

        Possible verdicts:
        - INSUFFICIENT_EVIDENCE: if evidence is absent or retrieval score is below threshold.
        - CONTRADICTED: if evidence exists and hallucination probability >= score_threshold.
        - SUPPORTED: if evidence exists and hallucination probability < score_threshold.
        """
        if not evidence_text or retrieval_score < self.retrieval_threshold:
            return "INSUFFICIENT_EVIDENCE"

        if hallucination_prob >= self.score_threshold:
            return "CONTRADICTED"

        return "SUPPORTED"

    def aggregate_response_verdict(self, claim_scores: List[float], verdicts: List[str]) -> Tuple[float, str]:
        """
        Compute configurable response-level hallucination score and verdict.
        """
        if not claim_scores:
            return 0.0, "SUPPORTED"

        if self.aggregation_strategy == "max_risk":
            score = float(np.max(claim_scores))
        elif self.aggregation_strategy == "weighted_mean":
            weights = np.linspace(1.0, 1.5, len(claim_scores))
            score = float(np.average(claim_scores, weights=weights))
        else:
            score = float(np.mean(claim_scores))

        if "CONTRADICTED" in verdicts or score >= self.score_threshold:
            overall_verdict = "HALLUCINATION_DETECTED"
        elif "INSUFFICIENT_EVIDENCE" in verdicts:
            overall_verdict = "INSUFFICIENT_EVIDENCE"
        else:
            overall_verdict = "SUPPORTED"

        return score, overall_verdict

    def analyze_response(
        self,
        query: str,
        generated_response: str,
        reference_context: Union[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Analyze generated response against reference context.
        """
        if isinstance(reference_context, str):
            documents = [{"doc_id": "ref_doc_0", "text": reference_context}]
        else:
            documents = reference_context

        # Index context
        self.retriever.index_documents(documents)

        # Extract structured claims
        structured_claims = self.extract_claims(generated_response)

        claim_results = []
        supported_count = 0
        contradicted_count = 0
        insufficient_count = 0
        claim_scores = []
        verdicts = []

        for claim_obj in structured_claims:
            claim_text = claim_obj.claim_text
            c_id = claim_obj.claim_id

            # Retrieve structured evidence
            evidences = self.retriever.retrieve_evidence(claim_text, top_k=1, claim_id=c_id)
            if evidences:
                best_evidence = evidences[0]["text"]
                retrieval_score = float(evidences[0]["similarity_score"])
                ev_id = evidences[0]["evidence_id"]
            else:
                best_evidence = ""
                retrieval_score = 0.0
                ev_id = "ev_none"

            inst = {
                "claim": claim_text,
                "evidence": best_evidence,
                "retrieval_score": retrieval_score
            }

            f_dict = self.vectorizer.extract_dict(claim_text, best_evidence, retrieval_score)

            if hasattr(self.model, "predict_hallucination_score"):
                prob = float(self.model.predict_hallucination_score([inst])[0])
            else:
                prob = float(f_dict.get("nli_prob_contradiction", 0.5))

            verdict = self.determine_claim_verdict(prob, retrieval_score, best_evidence)
            if verdict == "SUPPORTED":
                supported_count += 1
            elif verdict == "CONTRADICTED":
                contradicted_count += 1
            else:
                insufficient_count += 1

            claim_scores.append(prob)
            verdicts.append(verdict)

            explanation = self.explainability_engine.generate_claim_explanation(
                claim=claim_text,
                evidence=best_evidence,
                feature_dict=f_dict,
                hallucination_prob=prob,
                verdict="Hallucinated" if verdict == "CONTRADICTED" else ("Uncertain" if verdict == "INSUFFICIENT_EVIDENCE" else "Factual")
            )

            claim_results.append({
                "claim_id": c_id,
                "claim_text": claim_text,
                "sentence_index": claim_obj.sentence_index,
                "evidence_id": ev_id,
                "retrieved_evidence": best_evidence,
                "retrieval_score": retrieval_score,
                "nli_entailment": float(f_dict.get("nli_prob_entailment", 0.0)),
                "nli_contradiction": float(f_dict.get("nli_prob_contradiction", 0.0)),
                "nli_neutral": float(f_dict.get("nli_prob_neutral", 0.0)),
                "semantic_similarity": float(f_dict.get("semantic_cosine_sim", 0.0)),
                "hallucination_probability": prob,
                "verdict": verdict,
                "explanation": explanation
            })

        overall_score, overall_verdict = self.aggregate_response_verdict(claim_scores, verdicts)

        return {
            "query": query,
            "generated_response": generated_response,
            "total_claims": len(structured_claims),
            "supported_claims_count": supported_count,
            "contradicted_claims_count": contradicted_count,
            "insufficient_evidence_claims_count": insufficient_count,
            "hallucinated_claims_count": contradicted_count,
            "overall_hallucination_score": float(overall_score),
            "overall_verdict": overall_verdict,
            "claim_breakdown": claim_results
        }
