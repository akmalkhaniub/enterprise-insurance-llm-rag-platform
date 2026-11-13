# Commercial Insurance Policies & Claims Corpus (RAG, LoRA & Agentic Adjudication)

## Overview & Context
This synthetic commercial insurance corpus pairs **structured policy clause wording and contractual limits** (`16` clauses across `6` commercial lines) with **multi-clause actuarial claims adjudication test cases** (`8` claims with ground-truth coverage decisions, net payouts, governing clause citations, and structured extraction targets). Produced alongside the [`enterprise-insurance-llm-rag-platform`](https://github.com/akmalkhaniub/enterprise-insurance-llm-rag-platform) benchmark suite, it is designed for **Hybrid RAG (`FAISS` + `BM25` + `RRF`)**, **structured JSON extraction via LoRA/QLoRA**, **DPO grounded-citation alignment**, and **agentic claims adjudication evaluation**.

The corpus spans 6 commercial lines:
1. **Cyber & Network Security Liability** (ransomware extortion, incident forensics, MFA/ISO-27001 condition precedents)
2. **Commercial Industrial Property & BI** (machinery breakdown, named storm flood sublimits, maintenance exclusions)
3. **Ocean Marine & Pharmaceutical Cold-Chain Cargo** (Institute Cargo Clauses A, reefer breakdown spoilage)
4. **Directors & Officers (D&O) Liability** (Side A/B securities litigation, intentional fraud exclusions)
5. **Employer Medical Stop-Loss & Catastrophic Health** (specific stop-loss attachment points, biologics/ICU care)
6. **Commercial Heavy Logistics & Telematics Fleet** (Class-8 tractor collision, ELD compliance exclusions)

---

## Files Included (2 Relational CSV Tables)

### 1. `insurance_policy_clauses.csv` (16 rows × 12 columns)
Structured commercial policy clauses (`COVERAGE_GRANT`: 6, `EXCLUSION`: 6, `SUBLIMIT`: 2, `CONDITION_PRECEDENT`: 2) indexed for dense/lexical retrieval and long-context extraction depth experiments:
- `chunk_id` *(string)*: Unique clause identifier (e.g., `CLS-CYB-101`).
- `policy_id` *(string)*: Master policy identifier joining directly to `insurance_claims_benchmark.csv` (`POL-CYB-2026-001`, etc.).
- `product_line` *(string)*: Commercial insurance line name.
- `section_code` *(string)*: Formal contract section label (e.g., `SECTION I.A`).
- `section_title` *(string)*: Title of the insuring agreement, exclusion, sublimit, or condition precedent.
- `clause_type` *(string)*: Categorical role (`COVERAGE_GRANT`, `EXCLUSION`, `SUBLIMIT`, `CONDITION_PRECEDENT`).
- `token_offset` *(integer)*: Nominal token position inside a full-length master policy document.
- `text` *(string)*: Full legal contract wording of the clause.
- `max_limit_usd` *(number)*: Maximum coverage limit or sublimit in USD (`0.0` for exclusions).
- `deductible_usd` *(number)*: Mandatory deductible/retention in USD (`0.0` when not applicable).
- `waiting_period_hours` *(integer)*: Qualifying business interruption or outage waiting period in hours.
- `requires_iso27001_or_mfa` *(boolean)*: Technical security safeguard prerequisite flag.

### 2. `insurance_claims_benchmark.csv` (8 rows × 13 columns)
End-to-end commercial claims adjudication benchmark (`APPROVED_PARTIAL`: 5, `DENIED_EXCLUSION`: 3) with deterministic net payouts and target structured JSON outputs:
- `claim_id` *(string)*: Unique claim identifier (e.g., `CLM-2026-9001`).
- `policy_id` *(string)*: Foreign key linking to `insurance_policy_clauses.csv`.
- `product_line` *(string)*: Commercial insurance line name.
- `insured_entity` *(string)*: Synthetic corporate policyholder name.
- `loss_date` *(string)*: ISO date (`YYYY-MM-DD`) when the loss event occurred.
- `reported_date` *(string)*: ISO date (`YYYY-MM-DD`) when the claim was notified to the insurer.
- `claimed_amount_usd` *(number)*: Initial gross loss amount claimed in USD.
- `peril_category` *(string)*: Standardized category of the triggering peril.
- `claim_narrative` *(string)*: Unstructured loss narrative and field adjuster findings.
- `ground_truth_decision` *(string)*: Correct adjudication outcome (`APPROVED_PARTIAL` or `DENIED_EXCLUSION`).
- `ground_truth_payout_usd` *(number)*: Verified net settlement payout in USD after deductibles, sublimits, or exclusion denial.
- `governing_clause_id` *(string)*: Foreign key pointing to the controlling `chunk_id` in `insurance_policy_clauses.csv`.
- `expected_json_extraction` *(string)*: Target serialized dictionary/JSON structure for supervised extraction and schema evaluation.

---

## Use Cases & Source
- **Hybrid RAG Benchmarking:** Measure `Recall@1` and `nDCG@5` mapping `claim_narrative` queries to `governing_clause_id` across dense embeddings, BM25, and Reciprocal Rank Fusion.
- **Deterministic Payout & Exclusion Reasoning:** Test whether LLM agents apply `max_limit_usd`, `deductible_usd`, `waiting_period_hours`, and `requires_iso27001_or_mfa` without hallucinating coverage on excluded losses.
- **Source Repository:** https://github.com/akmalkhaniub/enterprise-insurance-llm-rag-platform
- **License:** CC0-1.0 (Public Domain — synthetic benchmark data).
