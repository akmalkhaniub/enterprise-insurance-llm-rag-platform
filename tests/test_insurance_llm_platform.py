"""End-to-End Test Suite for the Enterprise Insurance Mining, LLM Fine-Tuning, RAG & Agentic Platform."""

from __future__ import annotations

from pathlib import Path
from fastapi.testclient import TestClient

from src.agentic_claims_workflow import AgenticClaimsOrchestrator
from src.api import app
from src.insurance_corpus import (
    build_insurance_claims_cases,
    build_insurance_policy_clauses,
    synthesize_long_policy_document,
)
from src.llm_finetuning_and_mlops import compute_drift_and_retraining_status, get_llm_finetuning_benchmarks
from src.long_context_engine import benchmark_long_context_extraction, compute_rope_frequencies
from src.rag_engine import HybridInsuranceRAGEngine


client = TestClient(app)


def test_insurance_corpus_and_long_document_synthesis() -> None:
    clauses = build_insurance_policy_clauses()
    claims = build_insurance_claims_cases()
    assert len(clauses) >= 16
    assert len(claims) == 8

    long_doc = synthesize_long_policy_document("POL-CYB-2026-001", target_tokens=32768)
    assert long_doc["estimated_context_tokens"] >= 32768
    assert long_doc["total_sections"] == 40
    assert len(long_doc["needle_clause_ids"]) >= 3


def test_hybrid_rag_engine_beats_generic_baseline() -> None:
    rag = HybridInsuranceRAGEngine()
    bench = rag.evaluate_retrieval_benchmarks()
    leaderboard = {row["mode_key"]: row for row in bench["leaderboard"]}

    assert leaderboard["hybrid_rrf"]["recall_at_1"] == 1.0
    assert leaderboard["hybrid_rrf"]["ndcg_at_5"] >= leaderboard["generic_dense"]["ndcg_at_5"]

    # Check MFA exclusion query retrieves CLS-CYB-102
    hits = rag.retrieve(
        "Server compromised over RDP with disabled MFA and unpatched CVSS 9.8 vulnerability",
        policy_id_filter="POL-CYB-2026-001",
        top_k=3,
        mode="hybrid_rrf",
    )
    assert hits[0]["chunk_id"] == "CLS-CYB-102"


def test_long_context_rope_scaling_and_middle_depth_recall() -> None:
    rope_std = compute_rope_frequencies(scaling_type="standard_4k", scale_factor=1.0)
    rope_ntk = compute_rope_frequencies(scaling_type="dynamic_ntk", scale_factor=8.0)
    assert rope_ntk["effective_theta"] > rope_std["effective_theta"]

    bench = benchmark_long_context_extraction("POL-CYB-2026-001")
    rows = bench["leaderboard"]
    naive_4k = rows[0]
    ntk_ours = rows[-1]
    assert ntk_ours["middle_depth_50pct_acc"] > naive_4k["middle_depth_50pct_acc"]
    assert ntk_ours["overall_needle_recall"] >= 0.98


def test_agentic_claims_orchestrator_and_langsmith_traces() -> None:
    orchestrator = AgenticClaimsOrchestrator()
    eval_res = orchestrator.evaluate_all_claims_benchmark()
    assert eval_res["decision_accuracy"] == 1.0
    assert eval_res["clause_attribution_f1"] == 1.0
    assert eval_res["actuarial_payout_exact_match"] == 1.0

    single = orchestrator.adjudicate_claim(claim_id="CLM-2026-9002")
    assert single["adjudication"]["adjudication_decision"] == "DENIED_EXCLUSION"
    assert single["adjudication"]["approved_net_payout_usd"] == 0.0
    assert len(single["trace_spans"]) == 5


def test_llm_finetuning_mlflow_and_drift_monitor() -> None:
    ft = get_llm_finetuning_benchmarks()
    assert len(ft["runs"]) >= 5
    zero_shot = ft["runs"][0]
    unsloth_qlora = ft["runs"][3]
    assert unsloth_qlora["holdout_perplexity"] < zero_shot["holdout_perplexity"]
    assert unsloth_qlora["token_f1_score"] > zero_shot["token_f1_score"]

    healthy = compute_drift_and_retraining_status(psi_shock=0.0)
    assert healthy["automated_retraining_triggered"] is False
    drifted = compute_drift_and_retraining_status(psi_shock=1.2)
    assert drifted["automated_retraining_triggered"] is True


def test_fastapi_endpoints_and_console() -> None:
    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "healthy"

    r_ov = client.get("/api/overview")
    assert r_ov.status_code == 200
    data = r_ov.json()
    assert "rag_benchmarks" in data
    assert "agentic_claims_benchmarks" in data

    r_adj = client.post("/api/agent/adjudicate", json={"claim_id": "CLM-2026-9001"})
    assert r_adj.status_code == 200
    assert r_adj.json()["adjudication"]["approved_net_payout_usd"] == 750_000.0

    r_ui = client.get("/")
    assert r_ui.status_code == 200
    assert "Enterprise Insurance Policy Mining" in r_ui.text


def test_docker_and_azure_deployment_manifests_exist() -> None:
    root = Path(__file__).resolve().parent.parent
    assert (root / "Dockerfile").exists()
    assert (root / "docker-compose.yml").exists()
    assert (root / "deploy" / "azure_aks" / "aks-llm-deployment.yaml").exists()
    assert (root / "deploy" / "azure_ml" / "aml-qlora-finetune-job.yaml").exists()
    assert (root / "deploy" / "deepspeed" / "ds_zero2_qlora_config.json").exists()
