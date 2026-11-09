# Commercial Insurance Policies & Claims Corpus (RAG & LLM Mining) 🛡️📄

## Overview & Context
This dataset provides a realistic, structured commercial insurance policy clause library and claims adjudication benchmark designed specifically for **Retrieval-Augmented Generation (RAG)**, **Long-Context Window Modeling**, and **LLM Fine-Tuning (PEFT, QLoRA, DPO)**.

Unlike simplistic toy insurance datasets that only contain numeric tabular records, this corpus bridges the gap between **unstructured policy contracts** and **structured actuarial loss adjudication**. It covers 6 major commercial insurance lines:
1. **Cyber & Network Security Liability** (Ransomware extortion, incident forensics, MFA exclusions, Business Email Compromise sublimits)
2. **Commercial Industrial Property & Business Interruption** (Direct physical loss, machinery breakdown, Named Storm flash flood sublimits, gradual roof rot exclusions)
3. **Ocean Marine & Pharmaceutical Cold-Chain Cargo** (Institute Cargo Clauses A, temperature spoilage from reefer compressor breakdown)
4. **Directors & Officers (D&O) Liability** (Side A/B/C securities litigation, intentional fraud exclusions)
5. **Employer Medical Stop-Loss & Catastrophic Health** (Specific stop-loss reimbursement, ICU care, FDA-approved oncology biologics)
6. **Commercial Heavy Logistics & Telematics Fleet** (Class-8 tractors, collision coverage, Electronic Logging Device / ELD tampering exclusions)

---

## Files Included

### 1. `insurance_policy_clauses.csv`
Contains 16 granular commercial policy clauses and endorsements.
* `chunk_id` *(string)*: Unique identifier for each policy clause (e.g., `CLS-CYB-101`).
* `policy_id` *(string)*: Master policy schedule code (e.g., `POL-CYB-2026-001`).
* `product_line` *(string)*: Commercial insurance line.
* `section_code` *(string)*: Formal contract section (e.g., `SECTION I.A`, `SECTION IV.E.7`).
* `section_title` *(string)*: Descriptive title of the insuring agreement or endorsement.
* `clause_type` *(string)*: Categorical clause classification (`COVERAGE_GRANT`, `EXCLUSION`, `SUBLIMIT`, `CONDITION_PRECEDENT`).
* `token_offset` *(integer)*: Approximate token depth inside a standard master contract.
* `text` *(string)*: Full contractual clause legal wording.
* `max_limit_usd` *(float)*: Maximum coverage limit in USD ($0 for pure exclusions).
* `deductible_usd` *(float)*: Mandatory policy retention/deductible in USD.
* `waiting_period_hours` *(integer)*: Business interruption qualifying waiting period.
* `requires_iso27001_or_mfa` *(boolean)*: Specific technical safeguard prerequisite.

### 2. `insurance_claims_benchmark.csv`
Contains 8 multi-clause commercial claims test cases with ground-truth actuarial payouts.
* `claim_id` *(string)*: Unique claim ID (e.g., `CLM-2026-9001`).
* `policy_id` *(string)*: Matching policy code.
* `product_line` *(string)*: Target insurance line.
* `insured_entity` *(string)*: Commercial insured corporate name.
* `loss_date` & `reported_date` *(ISO date string)*: Timestamps for condition precedent analysis.
* `claimed_amount_usd` *(float)*: Initial claimed loss.
* `peril_category` *(string)*: Nature of the loss event (e.g., `Ransomware Encryption`, `Reefer Breakdown`).
* `claim_narrative` *(string)*: Unstructured loss event description and adjuster notes.
* `ground_truth_decision` *(string)*: Correct adjudication outcome (`APPROVED_PARTIAL`, `DENIED_EXCLUSION`, `APPROVED_FULL`).
* `ground_truth_payout_usd` *(float)*: Mathematically verified net payout after deductibles and sublimits.
* `governing_clause_id` *(string)*: Target clause ID that governs coverage or exclusion.
* `expected_json_extraction` *(JSON string)*: Target structured JSON schema for LLM instruction-tuning.

---

## Inspiration & Use Cases
1. **Hybrid RAG Benchmark:** Evaluate Dense Embeddings (e.g., `all-MiniLM-L6-v2`, `BGE`) vs. BM25 Lexical vs. Reciprocal Rank Fusion (RRF) on complex insurance policy retrieval.
2. **LLM Fine-Tuning (LoRA / QLoRA):** Train small open models (`Qwen2.5`, `Llama-3`, `SmolLM2`) to extract structured adjudication JSON from unstructured loss narratives.
3. **Direct Preference Optimization (DPO):** Train models to prefer strictly grounded policy citations over hallucinated claims approvals.
4. **Agentic Workflows:** Build multi-step LangGraph / LangChain agents that parse claims, retrieve clauses, audit exclusions, and calculate net payouts.

## License
CC0: Public Domain. Free for research, benchmarking, and commercial development.
