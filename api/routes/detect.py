"""
FastAPI route definitions for hallucination detection, verification, retrieval, and LLM generation endpoints.
"""

from fastapi import APIRouter, HTTPException
from api.schemas.schemas import (
    DetectionRequest, DetectionResponse,
    VerificationRequest, VerificationResponse,
    RetrievalRequest, RetrievalResponse, StructuredEvidenceItem,
    GenerateAndDetectRequest, HealthResponse
)
from src.pipeline.hallucination_detector import HallucinationDetector
from src.utils.llm_provider import get_llm_provider
from src.utils.logger import get_logger

logger = get_logger("api_router")

router = APIRouter()
detector = HallucinationDetector()
llm_provider = get_llm_provider()


@router.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        retriever_ready=True,
        model_ready=detector.model.is_fitted
    )


@router.post("/detect", response_model=DetectionResponse)
def detect_hallucination(request: DetectionRequest):
    try:
        result = detector.analyze_response(
            query=request.query,
            generated_response=request.generated_response,
            reference_context=request.reference_context
        )
        return result
    except Exception as e:
        logger.error(f"Error during detection: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/verify", response_model=VerificationResponse)
def verify_claim(request: VerificationRequest):
    try:
        f_dict = detector.vectorizer.extract_dict(request.claim, request.evidence, 0.90)
        prob = detector.model.predict_hallucination_score([{
            "claim": request.claim,
            "evidence": request.evidence,
            "retrieval_score": 0.90
        }])[0]
        verdict = detector.determine_claim_verdict(prob, 0.90, request.evidence)

        return VerificationResponse(
            claim=request.claim,
            evidence=request.evidence,
            nli_entailment=float(f_dict.get("nli_prob_entailment", 0.0)),
            nli_contradiction=float(f_dict.get("nli_prob_contradiction", 0.0)),
            nli_neutral=float(f_dict.get("nli_prob_neutral", 0.0)),
            semantic_similarity=float(f_dict.get("semantic_cosine_sim", 0.0)),
            verdict=verdict
        )
    except Exception as e:
        logger.error(f"Error during verification: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrieve", response_model=RetrievalResponse)
def retrieve_evidence_passages(request: RetrievalRequest):
    try:
        detector.retriever.index_documents([{"doc_id": "api_doc_0", "text": request.context}])
        evidences = detector.retriever.retrieve_evidence(request.claim, top_k=request.top_k)
        structured_items = [
            StructuredEvidenceItem(
                claim_id=ev.get("claim_id", "claim_001"),
                evidence_id=ev.get("evidence_id", "ev_001"),
                text=ev.get("text", ""),
                similarity_score=float(ev.get("similarity_score", 0.0)),
                is_above_threshold=bool(ev.get("is_above_threshold", True))
            ) for ev in evidences
        ]
        return RetrievalResponse(claim=request.claim, evidences=structured_items)
    except Exception as e:
        logger.error(f"Error during retrieval: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-and-detect", response_model=DetectionResponse)
def generate_and_detect(request: GenerateAndDetectRequest):
    try:
        generated_text = llm_provider.generate_response(request.prompt)
    except RuntimeError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e)
        )
    try:
        result = detector.analyze_response(
            query=request.prompt,
            generated_response=generated_text,
            reference_context=request.reference_context
        )
        return result
    except Exception as e:
        logger.error(f"Error during generate-and-detect: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

