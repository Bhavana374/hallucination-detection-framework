"""
Pydantic API Request and Response Schemas.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class DetectionRequest(BaseModel):
    query: str = Field(..., example="What is the capital of France?")
    generated_response: str = Field(..., example="Paris is the capital of France and has a population of 20 million.")
    reference_context: str = Field(..., example="Paris is the capital and most populous city of France, with an estimated population of 2.1 million.")


class VerificationRequest(BaseModel):
    claim: str = Field(..., example="Paris is the capital of France.")
    evidence: str = Field(..., example="Paris is the capital of France.")


class RetrievalRequest(BaseModel):
    claim: str = Field(..., example="Apollo 11 landed on the Moon.")
    context: str = Field(..., example="Apollo 11 was the spaceflight that landed Neil Armstrong on the Moon.")
    top_k: int = Field(default=3, example=3)


class GenerateAndDetectRequest(BaseModel):
    prompt: str = Field(..., example="Explain the Apollo 11 moon landing.")
    reference_context: str = Field(..., example="Apollo 11 landed on the Moon on July 20, 1969 with Neil Armstrong.")


class ClaimDetail(BaseModel):
    claim_id: str
    claim_text: str
    sentence_index: int
    evidence_id: str
    retrieved_evidence: str
    retrieval_score: float
    hallucination_probability: float
    verdict: str
    explanation: Dict[str, Any]


class DetectionResponse(BaseModel):
    query: str
    generated_response: str
    total_claims: int
    supported_claims_count: int
    contradicted_claims_count: int
    insufficient_evidence_claims_count: int
    hallucinated_claims_count: int
    overall_hallucination_score: float
    overall_verdict: str
    claim_breakdown: List[ClaimDetail]


class VerificationResponse(BaseModel):
    claim: str
    evidence: str
    nli_entailment: float
    nli_contradiction: float
    nli_neutral: float
    semantic_similarity: float
    verdict: str


class StructuredEvidenceItem(BaseModel):
    claim_id: str
    evidence_id: str
    text: str
    similarity_score: float
    is_above_threshold: bool


class RetrievalResponse(BaseModel):
    claim: str
    evidences: List[StructuredEvidenceItem]


class HealthResponse(BaseModel):
    status: str
    version: str
    retriever_ready: bool
    model_ready: bool
