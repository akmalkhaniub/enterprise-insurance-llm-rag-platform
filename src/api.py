"""FastAPI Backend for the Enterprise Insurance Mining, LLM Fine-Tuning, RAG & Agentic AI Platform."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.agentic_claims_workflow import AgenticClaimsOrchestrator
from src.insurance_corpus import build_insurance_claims_cases, build_insurance_policy_clauses
from src.llm_finetuning_and_mlops import compute_drift_and_retraining_status, get_llm_finetuning_benchmarks
from src.long_context_engine import benchmark_long_context_extraction
from src.rag_engine import HybridInsuranceRAGEngine


app = FastAPI(
    title="Enterprise Insurance Mining, LLM Fine-Tuning (Unsloth/QLoRA), Hybrid RAG & Agentic AI Platform",
    version="1.0.0",
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

RAG_ENGINE = HybridInsuranceRAGEngine()
AGENT_ORCHESTRATOR = AgenticClaimsOrchestrator(rag_engine=RAG_ENGINE)


class ClaimAdjudicationRequest(BaseModel):
    claim_id: str | None = Field(default="CLM-2026-9001")
    policy_id: str | None = Field(default="POL-CYB-2026-001")
    claimed_amount_usd: float | None = Field(default=850_000.0, ge=0.0)
    custom_narrative: str | None = Field(default=None)


@app.get("/", response_class=HTMLResponse)
def serve_console() -> HTMLResponse:
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Enterprise Insurance LLM & RAG Platform Active</h1>")


@app.get("/api/health")
def health_check() -> dict[str, Any]:
    return {
        "status": "healthy",
        "service": "enterprise-insurance-llm-rag-platform",
        "indexed_policy_clauses": len(build_insurance_policy_clauses()),
        "benchmark_claims": len(build_insurance_claims_cases()),
    }


@app.get("/api/overview")
def get_platform_overview() -> dict[str, Any]:
    rag_bench = RAG_ENGINE.evaluate_retrieval_benchmarks()
    agent_bench = AGENT_ORCHESTRATOR.evaluate_all_claims_benchmark()
    long_ctx = benchmark_long_context_extraction()
    llm_ft = get_llm_finetuning_benchmarks()

    return {
        "rag_benchmarks": rag_bench,
        "agentic_claims_benchmarks": agent_bench,
        "long_context_benchmarks": long_ctx,
        "llm_finetuning_mlops": llm_ft,
    }


@app.get("/api/rag/search")
def search_policy_clauses(
    query: str = Query(
        default="LockBit ransomware encrypted servers for 36 hours with MFA active",
        min_length=3,
    ),
    policy_id: str | None = Query(default=None),
    mode: str = Query(default="hybrid_rrf"),
    top_k: int = Query(default=4, ge=1, le=10),
) -> dict[str, Any]:
    results = RAG_ENGINE.retrieve(
        query=query,
        policy_id_filter=policy_id if policy_id else None,
        top_k=top_k,
        mode=mode,
    )
    return {
        "query": query,
        "mode": mode,
        "policy_id_filter": policy_id,
        "results": results,
    }


@app.post("/api/agent/adjudicate")
def run_agentic_adjudication(payload: ClaimAdjudicationRequest) -> dict[str, Any]:
    return AGENT_ORCHESTRATOR.adjudicate_claim(
        claim_id=payload.claim_id,
        custom_narrative=payload.custom_narrative,
        claimed_amount_usd=payload.claimed_amount_usd,
        policy_id=payload.policy_id,
    )


@app.get("/api/mlops/drift")
def check_model_drift(psi_shock: float = Query(default=0.0, ge=0.0, le=2.0)) -> dict[str, Any]:
    return compute_drift_and_retraining_status(psi_shock=psi_shock)
