"""
FastAPI Server Entry Point for Evidence-Grounded Hallucination Detection API.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes.detect import router

app = FastAPI(
    title="Evidence-Grounded Hallucination Detection API",
    description="FastAPI backend providing claim-level verdict and aggregated response hallucination scores.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
def root():
    return {
        "name": "Evidence-Grounded Hallucination Detection API",
        "status": "running",
        "docs_url": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
