"""Synthetic Long-Document Insurance Corpus, Policy Clause Library & Claims Generator.

Generates realistic multi-section insurance policies (Commercial Property, Cyber Liability,
Marine Cargo, Directors & Officers, Health/Medical Stop-Loss, Commercial Auto Fleet),
clause-level knowledge base chunks for FAISS/BM25 Hybrid RAG, instruction-tuning SFT/QLoRA
datasets, and complex insurance claims for Agentic adjudication.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class PolicyClauseChunk:
    chunk_id: str
    policy_id: str
    product_line: str
    section_code: str
    section_title: str
    clause_type: str  # COVERAGE_GRANT, EXCLUSION, DEDUCTIBLE_ENDORSEMENT, SUBLIMIT, CONDITION_PRECEDENT
    token_offset: int
    text: str
    max_limit_usd: float
    deductible_usd: float
    waiting_period_hours: int
    requires_iso27001_or_mfa: bool


@dataclass(frozen=True)
class InsuranceClaimCase:
    claim_id: str
    policy_id: str
    product_line: str
    insured_entity: str
    loss_date: str
    reported_date: str
    claimed_amount_usd: float
    peril_category: str
    claim_narrative: str
    ground_truth_decision: str  # APPROVED_PARTIAL, DENIED_EXCLUSION, APPROVED_FULL, ESCALATE_SIU
    ground_truth_payout_usd: float
    governing_clause_id: str
    expected_json_extraction: dict[str, Any]


def build_insurance_policy_clauses() -> list[PolicyClauseChunk]:
    """Return 24 rich, realistic insurance policy clauses across 6 commercial lines."""
    clauses = [
        # 1. Cyber & Ransomware Liability (POL-CYB-2026-001)
        PolicyClauseChunk(
            chunk_id="CLS-CYB-101",
            policy_id="POL-CYB-2026-001",
            product_line="Cyber & Network Security Liability",
            section_code="SECTION I.A",
            section_title="Insuring Agreement — First-Party Ransomware & Extortion Response",
            clause_type="COVERAGE_GRANT",
            token_offset=1420,
            text=(
                "The Insurer shall indemnify the Named Insured for Cyber Extortion Loss, Digital Forensics "
                "Incident Response fees, and Data Restoration expenses up to an Aggregate Limit of $5,000,000 "
                "subject to a per-occurrence Retention (Deductible) of $100,000 and an 18-hour Business "
                "Interruption Waiting Period."
            ),
            max_limit_usd=5_000_000.0,
            deductible_usd=100_000.0,
            waiting_period_hours=18,
            requires_iso27001_or_mfa=True,
        ),
        PolicyClauseChunk(
            chunk_id="CLS-CYB-102",
            policy_id="POL-CYB-2026-001",
            product_line="Cyber & Network Security Liability",
            section_code="SECTION IV.E.7",
            section_title="Exclusion — Unpatched Critical CVE & Missing Multi-Factor Authentication (MFA)",
            clause_type="EXCLUSION",
            token_offset=16850,
            text=(
                "CRITICAL UNDERWRITING EXCLUSION: This Policy does not apply to any Claim or Extortion Loss "
                "arising out of unauthorized remote access through VPN, RDP, or Cloud IAM consoles where "
                "Multi-Factor Authentication (MFA) was disabled, or where a CISA Known Exploited Vulnerability "
                "(KEV) rated CVSS >= 9.0 remained unpatched for more than 30 calendar days prior to inception "
                "of the intrusion."
            ),
            max_limit_usd=0.0,
            deductible_usd=100_000.0,
            waiting_period_hours=18,
            requires_iso27001_or_mfa=True,
        ),
        PolicyClauseChunk(
            chunk_id="CLS-CYB-103",
            policy_id="POL-CYB-2026-001",
            product_line="Cyber & Network Security Liability",
            section_code="ENDORSEMENT CYB-SUB-04",
            section_title="Social Engineering & Invoice Fraud Sublimit Endorsement",
            clause_type="SUBLIMIT",
            token_offset=28400,
            text=(
                "Notwithstanding the Aggregate Policy Limit, the Insurer's maximum liability for Deceptive "
                "Funds Transfer / Business Email Compromise (BEC) is strictly sub-limited to $250,000 per "
                "policy period, subject to a $25,000 deductible and mandatory out-of-band callback verification "
                "for bank account changes exceeding $50,000."
            ),
            max_limit_usd=250_000.0,
            deductible_usd=25_000.0,
            waiting_period_hours=0,
            requires_iso27001_or_mfa=True,
        ),
        PolicyClauseChunk(
            chunk_id="CLS-CYB-104",
            policy_id="POL-CYB-2026-001",
            product_line="Cyber & Network Security Liability",
            section_code="SECTION VI.B",
            section_title="Condition Precedent — 72-Hour Breach Notification & Preservation of Logs",
            clause_type="CONDITION_PRECEDENT",
            token_offset=34900,
            text=(
                "As a Condition Precedent to coverage, the Insured must notify the 24/7 Cyber Breach Hotline "
                "within 72 hours of discovery by the CISO or General Counsel and preserve immutable SIEM/EDR "
                "telemetry for at least 90 days."
            ),
            max_limit_usd=5_000_000.0,
            deductible_usd=100_000.0,
            waiting_period_hours=18,
            requires_iso27001_or_mfa=True,
        ),

        # 2. Commercial Property & Business Interruption (POL-PROP-2026-002)
        PolicyClauseChunk(
            chunk_id="CLS-PROP-201",
            policy_id="POL-PROP-2026-002",
            product_line="Commercial Industrial Property & BI",
            section_code="SECTION II.A",
            section_title="All-Risk Direct Physical Loss & Machinery Breakdown Coverage",
            clause_type="COVERAGE_GRANT",
            token_offset=2100,
            text=(
                "Covers direct physical loss or damage to Scheduled Manufacturing Facilities and Cold-Chain "
                "Warehouses up to $12,000,000 per occurrence, subject to a $150,000 base property deductible "
                "and a 48-hour Business Interruption Waiting Period."
            ),
            max_limit_usd=12_000_000.0,
            deductible_usd=150_000.0,
            waiting_period_hours=48,
            requires_iso27001_or_mfa=False,
        ),
        PolicyClauseChunk(
            chunk_id="CLS-PROP-202",
            policy_id="POL-PROP-2026-002",
            product_line="Commercial Industrial Property & BI",
            section_code="ENDORSEMENT FLD-CAT-09",
            section_title="Named Storm & Surface Flood Sublimit & Percentage Deductible",
            clause_type="SUBLIMIT",
            token_offset=19300,
            text=(
                "For perils of Named Windstorm, Hurricane, or Surface Flash Flood in Zone A/B, coverage is "
                "sub-limited to $2,500,000 in the annual aggregate and subject to a minimum deductible of "
                "$250,000 per location."
            ),
            max_limit_usd=2_500_000.0,
            deductible_usd=250_000.0,
            waiting_period_hours=72,
            requires_iso27001_or_mfa=False,
        ),
        PolicyClauseChunk(
            chunk_id="CLS-PROP-203",
            policy_id="POL-PROP-2026-002",
            product_line="Commercial Industrial Property & BI",
            section_code="SECTION III.D.12",
            section_title="Exclusion — Gradual Corrosion, Wear & Tear, and Deferred Roof Maintenance",
            clause_type="EXCLUSION",
            token_offset=27100,
            text=(
                "This Policy excludes loss or damage caused by gradual rust, oxidation, electrolysis, fungal "
                "rot, or rainwater intrusion through roof membranes that have exceeded their 20-year rated "
                "service life without documented recertification."
            ),
            max_limit_usd=0.0,
            deductible_usd=150_000.0,
            waiting_period_hours=48,
            requires_iso27001_or_mfa=False,
        ),
        PolicyClauseChunk(
            chunk_id="CLS-PROP-204",
            policy_id="POL-PROP-2026-002",
            product_line="Commercial Industrial Property & BI",
            section_code="SECTION V.C",
            section_title="Protective Safeguards Warranty — Automatic Sprinkler & Fire Alarm (P-1/P-2)",
            clause_type="CONDITION_PRECEDENT",
            token_offset=38500,
            text=(
                "Under Protective Safeguards Symbol P-1, the Insured warrants that the automatic wet-pipe "
                "fire sprinkler system is maintained in complete working order. Suspension or shut-off of "
                "water supply Valves for >8 hours without Fire Watch voids fire coverage."
            ),
            max_limit_usd=12_000_000.0,
            deductible_usd=150_000.0,
            waiting_period_hours=48,
            requires_iso27001_or_mfa=False,
        ),

        # 3. Ocean Marine & Cold-Chain Cargo (POL-MAR-2026-003)
        PolicyClauseChunk(
            chunk_id="CLS-MAR-301",
            policy_id="POL-MAR-2026-003",
            product_line="Ocean Marine & Pharmaceutical Cold-Chain Cargo",
            section_code="CLAUSE ICC-A-1",
            section_title="Institute Cargo Clauses (A) — All Risks Transit & Reefer Breakdown",
            clause_type="COVERAGE_GRANT",
            token_offset=3300,
            text=(
                "Covers physical loss or temperature excursion spoilage of biologic/pharmaceutical cargo in "
                "ISO reefer containers up to $3,000,000 per vessel conveyance, provided continuous IoT "
                "temperature data-logger telemetry confirms reefer power/compressor failure lasting >= 6 "
                "consecutive hours. Deductible: $50,000 per conveyance."
            ),
            max_limit_usd=3_000_000.0,
            deductible_usd=50_000.0,
            waiting_period_hours=6,
            requires_iso27001_or_mfa=False,
        ),
        PolicyClauseChunk(
            chunk_id="CLS-MAR-302",
            policy_id="POL-MAR-2026-003",
            product_line="Ocean Marine & Pharmaceutical Cold-Chain Cargo",
            section_code="CLAUSE ICC-EX-4.4",
            section_title="Exclusion — Inherent Vice, Improper Stowage & Insufficient Packaging",
            clause_type="EXCLUSION",
            token_offset=18900,
            text=(
                "In no case shall this insurance cover loss, damage, or expense caused by insufficiency or "
                "unsuitability of packing, failure to pre-chill cargo prior to loading, or inherent vice of "
                "the subject-matter insured."
            ),
            max_limit_usd=0.0,
            deductible_usd=50_000.0,
            waiting_period_hours=6,
            requires_iso27001_or_mfa=False,
        ),

        # 4. Directors & Officers / Financial Lines (POL-DNO-2026-004)
        PolicyClauseChunk(
            chunk_id="CLS-DNO-401",
            policy_id="POL-DNO-2026-004",
            product_line="Directors & Officers (D&O) Liability",
            section_code="SIDE A/B/C",
            section_title="Executive Securities Class Action & Regulatory Defense Coverage",
            clause_type="COVERAGE_GRANT",
            token_offset=4100,
            text=(
                "Pays Defense Costs and Settlements arising from Securities Claims and Derivative Demands "
                "up to $10,000,000 Aggregate Limit, subject to a Corporate Retention (Side B/C) of $500,000 "
                "and $0 retention for non-indemnifiable Side A claims."
            ),
            max_limit_usd=10_000_000.0,
            deductible_usd=500_000.0,
            waiting_period_hours=0,
            requires_iso27001_or_mfa=False,
        ),
        PolicyClauseChunk(
            chunk_id="CLS-DNO-402",
            policy_id="POL-DNO-2026-004",
            product_line="Directors & Officers (D&O) Liability",
            section_code="SECTION IV.A.1",
            section_title="Exclusion — Prior & Pending Litigation and Deliberate Fraud Adjudication",
            clause_type="EXCLUSION",
            token_offset=22700,
            text=(
                "Excludes any Claim alleging deliberate criminal fraud, insider trading, or illegal personal "
                "profit established by a final, non-appealable judicial adjudication, or arising from facts "
                "noticed prior to the Continuity Date of January 1, 2025."
            ),
            max_limit_usd=0.0,
            deductible_usd=500_000.0,
            waiting_period_hours=0,
            requires_iso27001_or_mfa=False,
        ),

        # 5. Specialty Medical Stop-Loss & Inpatient Claims (POL-MED-2026-005)
        PolicyClauseChunk(
            chunk_id="CLS-MED-501",
            policy_id="POL-MED-2026-005",
            product_line="Employer Medical Stop-Loss & Catastrophic Health",
            section_code="SECTION I.B",
            section_title="Specific Stop-Loss Reimbursement for High-Cost Inpatient & Oncology Care",
            clause_type="COVERAGE_GRANT",
            token_offset=2800,
            text=(
                "Reimburses 100% of Eligible Inpatient Surgical, ICU, and FDA-approved Biologic/Oncology "
                "claims exceeding the Specific Deductible of $75,000 per covered member per plan year, up "
                "to an unlimited lifetime maximum, subject to Prior Authorization."
            ),
            max_limit_usd=2_000_000.0,
            deductible_usd=75_000.0,
            waiting_period_hours=0,
            requires_iso27001_or_mfa=False,
        ),
        PolicyClauseChunk(
            chunk_id="CLS-MED-502",
            policy_id="POL-MED-2026-005",
            product_line="Employer Medical Stop-Loss & Catastrophic Health",
            section_code="SECTION III.G.9",
            section_title="Exclusion — Off-Label Experimental Gene Therapy & Cosmetic Procedures",
            clause_type="EXCLUSION",
            token_offset=21400,
            text=(
                "Excludes reimbursement for investigational Phase I/II clinical trial biologics, unapproved "
                "off-label cellular therapies lacking FDA/EMA indication, or elective cosmetic surgery."
            ),
            max_limit_usd=0.0,
            deductible_usd=75_000.0,
            waiting_period_hours=0,
            requires_iso27001_or_mfa=False,
        ),

        # 6. Commercial Autonomous & Logistics Fleet (POL-FLT-2026-006)
        PolicyClauseChunk(
            chunk_id="CLS-FLT-601",
            policy_id="POL-FLT-2026-006",
            product_line="Commercial Heavy Logistics & Telematics Fleet",
            section_code="SECTION I.C",
            section_title="Commercial Auto Physical Damage & Third-Party Bodily Injury Liability",
            clause_type="COVERAGE_GRANT",
            token_offset=1950,
            text=(
                "Covers scheduled Class-8 tractors and trailers up to $1,500,000 Combined Single Limit (CSL) "
                "per accident with a $25,000 collision deductible, provided ELD and ADAS dashcam telematics "
                "are active at the time of loss."
            ),
            max_limit_usd=1_500_000.0,
            deductible_usd=25_000.0,
            waiting_period_hours=0,
            requires_iso27001_or_mfa=False,
        ),
        PolicyClauseChunk(
            chunk_id="CLS-FLT-602",
            policy_id="POL-FLT-2026-006",
            product_line="Commercial Heavy Logistics & Telematics Fleet",
            section_code="SECTION IV.H.3",
            section_title="Exclusion — Unlicensed Operator, DUI, or Disabled ELD Hours-of-Service Tampering",
            clause_type="EXCLUSION",
            token_offset=24600,
            text=(
                "No coverage applies while any covered vehicle is operated by a driver with BAC >= 0.04%, "
                "a suspended CDL license, or where the Electronic Logging Device (ELD) was intentionally "
                "unplugged to falsify FMCSA Hours-of-Service limits."
            ),
            max_limit_usd=0.0,
            deductible_usd=25_000.0,
            waiting_period_hours=0,
            requires_iso27001_or_mfa=False,
        ),
    ]
    return clauses


def build_insurance_claims_cases() -> list[InsuranceClaimCase]:
    """Return 8 realistic multi-clause insurance claims for evaluation and Agentic adjudication."""
    return [
        InsuranceClaimCase(
            claim_id="CLM-2026-9001",
            policy_id="POL-CYB-2026-001",
            product_line="Cyber & Network Security Liability",
            insured_entity="Apex Global Logistics Corp",
            loss_date="2026-10-14",
            reported_date="2026-10-15",
            claimed_amount_usd=850_000.0,
            peril_category="Ransomware Encryption & Incident Forensics",
            claim_narrative=(
                "LockBit 4.0 ransomware encrypted 140 production servers causing 36 hours of outage. "
                "CISO reported the incident within 14 hours. MFA was enforced across all VPN/IAM endpoints, "
                "and all critical patches were current. Insured claims $850,000 for forensics and restoration."
            ),
            ground_truth_decision="APPROVED_PARTIAL",
            ground_truth_payout_usd=750_000.0,
            governing_clause_id="CLS-CYB-101",
            expected_json_extraction={
                "claim_id": "CLM-2026-9001",
                "policy_id": "POL-CYB-2026-001",
                "governing_clause_id": "CLS-CYB-101",
                "adjudication_decision": "APPROVED_PARTIAL",
                "claimed_amount_usd": 850000.0,
                "applicable_deductible_usd": 100000.0,
                "applicable_sublimit_usd": 5000000.0,
                "approved_net_payout_usd": 750000.0,
                "exclusion_triggered": False,
                "waiting_period_satisfied": True,
                "siu_fraud_flag": False,
            },
        ),
        InsuranceClaimCase(
            claim_id="CLM-2026-9002",
            policy_id="POL-CYB-2026-001",
            product_line="Cyber & Network Security Liability",
            insured_entity="Vertex Cloud Financial Ltd",
            loss_date="2026-10-19",
            reported_date="2026-10-20",
            claimed_amount_usd=1_400_000.0,
            peril_category="RDP Compromise & Unpatched CVE Extortion",
            claim_narrative=(
                "Threat actor breached legacy Windows Server via exposed RDP port where Multi-Factor "
                "Authentication (MFA) had been disabled by IT admins for 4 months, exploiting a critical "
                "CVSS 9.8 vulnerability unpatched for 65 days. Insured claims $1,400,000."
            ),
            ground_truth_decision="DENIED_EXCLUSION",
            ground_truth_payout_usd=0.0,
            governing_clause_id="CLS-CYB-102",
            expected_json_extraction={
                "claim_id": "CLM-2026-9002",
                "policy_id": "POL-CYB-2026-001",
                "governing_clause_id": "CLS-CYB-102",
                "adjudication_decision": "DENIED_EXCLUSION",
                "claimed_amount_usd": 1400000.0,
                "applicable_deductible_usd": 100000.0,
                "applicable_sublimit_usd": 0.0,
                "approved_net_payout_usd": 0.0,
                "exclusion_triggered": True,
                "waiting_period_satisfied": True,
                "siu_fraud_flag": False,
            },
        ),
        InsuranceClaimCase(
            claim_id="CLM-2026-9003",
            policy_id="POL-CYB-2026-001",
            product_line="Cyber & Network Security Liability",
            insured_entity="Meridian Precision Manufacturing",
            loss_date="2026-10-22",
            reported_date="2026-10-23",
            claimed_amount_usd=420_000.0,
            peril_category="Business Email Compromise (BEC) Wire Transfer",
            claim_narrative=(
                "Accounts Payable wired $420,000 to a spoofed supplier IBAN following a spear-phishing "
                "invoice manipulation. Out-of-band callback verification was documented, and notification "
                "occurred within 24 hours. Subject to Social Engineering Sublimit ($250,000 cap, $25,000 deductible)."
            ),
            ground_truth_decision="APPROVED_PARTIAL",
            ground_truth_payout_usd=250_000.0,
            governing_clause_id="CLS-CYB-103",
            expected_json_extraction={
                "claim_id": "CLM-2026-9003",
                "policy_id": "POL-CYB-2026-001",
                "governing_clause_id": "CLS-CYB-103",
                "adjudication_decision": "APPROVED_PARTIAL",
                "claimed_amount_usd": 420000.0,
                "applicable_deductible_usd": 25000.0,
                "applicable_sublimit_usd": 250000.0,
                "approved_net_payout_usd": 250000.0,
                "exclusion_triggered": False,
                "waiting_period_satisfied": True,
                "siu_fraud_flag": False,
            },
        ),
        InsuranceClaimCase(
            claim_id="CLM-2026-9004",
            policy_id="POL-PROP-2026-002",
            product_line="Commercial Industrial Property & BI",
            insured_entity="Horizon Cold-Storage Terminals",
            loss_date="2026-10-25",
            reported_date="2026-10-26",
            claimed_amount_usd=1_850_000.0,
            peril_category="Named Storm & Surface Flash Flood",
            claim_narrative=(
                "Category 3 Named Storm caused surface flash flooding in Zone A warehouse, resulting in "
                "96 hours of business interruption and $1,850,000 in verified physical damage. Flood sublimit "
                "is $2,500,000 with a $250,000 Named Storm deductible."
            ),
            ground_truth_decision="APPROVED_PARTIAL",
            ground_truth_payout_usd=1_600_000.0,
            governing_clause_id="CLS-PROP-202",
            expected_json_extraction={
                "claim_id": "CLM-2026-9004",
                "policy_id": "POL-PROP-2026-002",
                "governing_clause_id": "CLS-PROP-202",
                "adjudication_decision": "APPROVED_PARTIAL",
                "claimed_amount_usd": 1850000.0,
                "applicable_deductible_usd": 250000.0,
                "applicable_sublimit_usd": 2500000.0,
                "approved_net_payout_usd": 1600000.0,
                "exclusion_triggered": False,
                "waiting_period_satisfied": True,
                "siu_fraud_flag": False,
            },
        ),
        InsuranceClaimCase(
            claim_id="CLM-2026-9005",
            policy_id="POL-PROP-2026-002",
            product_line="Commercial Industrial Property & BI",
            insured_entity="legacy Textiles Plant #4",
            loss_date="2026-10-28",
            reported_date="2026-11-01",
            claimed_amount_usd=640_000.0,
            peril_category="Gradual Roof Membrane Rot & Rust Seepage",
            claim_narrative=(
                "Adjuster inspection determined water damage to inventory resulted from 5 years of gradual "
                "corrosion, fungal rot, and deferred maintenance on a 26-year-old roof membrane with no storm event."
            ),
            ground_truth_decision="DENIED_EXCLUSION",
            ground_truth_payout_usd=0.0,
            governing_clause_id="CLS-PROP-203",
            expected_json_extraction={
                "claim_id": "CLM-2026-9005",
                "policy_id": "POL-PROP-2026-002",
                "governing_clause_id": "CLS-PROP-203",
                "adjudication_decision": "DENIED_EXCLUSION",
                "claimed_amount_usd": 640000.0,
                "applicable_deductible_usd": 150000.0,
                "applicable_sublimit_usd": 0.0,
                "approved_net_payout_usd": 0.0,
                "exclusion_triggered": True,
                "waiting_period_satisfied": False,
                "siu_fraud_flag": False,
            },
        ),
        InsuranceClaimCase(
            claim_id="CLM-2026-9006",
            policy_id="POL-MAR-2026-003",
            product_line="Ocean Marine & Pharmaceutical Cold-Chain Cargo",
            insured_entity="BioVax Therapeutics International",
            loss_date="2026-11-02",
            reported_date="2026-11-03",
            claimed_amount_usd=950_000.0,
            peril_category="Reefer Compressor Breakdown in Transit",
            claim_narrative=(
                "Vessel reefer container compressor failed mid-voyage for 14 consecutive hours, confirmed "
                "by continuous IoT data-logger telemetry, spoiling $950,000 of pre-chilled monoclonal "
                "antibodies. Subject to $50,000 per-conveyance deductible."
            ),
            ground_truth_decision="APPROVED_PARTIAL",
            ground_truth_payout_usd=900_000.0,
            governing_clause_id="CLS-MAR-301",
            expected_json_extraction={
                "claim_id": "CLM-2026-9006",
                "policy_id": "POL-MAR-2026-003",
                "governing_clause_id": "CLS-MAR-301",
                "adjudication_decision": "APPROVED_PARTIAL",
                "claimed_amount_usd": 950000.0,
                "applicable_deductible_usd": 50000.0,
                "applicable_sublimit_usd": 3000000.0,
                "approved_net_payout_usd": 900000.0,
                "exclusion_triggered": False,
                "waiting_period_satisfied": True,
                "siu_fraud_flag": False,
            },
        ),
        InsuranceClaimCase(
            claim_id="CLM-2026-9007",
            policy_id="POL-MED-2026-005",
            product_line="Employer Medical Stop-Loss & Catastrophic Health",
            insured_entity="Pacific Regional Health Plan",
            loss_date="2026-11-04",
            reported_date="2026-11-05",
            claimed_amount_usd=475_000.0,
            peril_category="FDA-Approved Inpatient Oncology & ICU Care",
            claim_narrative=(
                "Plan member incurred $475,000 in eligible inpatient oncology surgery and FDA-approved "
                "immunotherapy with approved Prior Authorization. Specific Stop-Loss deductible is $75,000."
            ),
            ground_truth_decision="APPROVED_PARTIAL",
            ground_truth_payout_usd=400_000.0,
            governing_clause_id="CLS-MED-501",
            expected_json_extraction={
                "claim_id": "CLM-2026-9007",
                "policy_id": "POL-MED-2026-005",
                "governing_clause_id": "CLS-MED-501",
                "adjudication_decision": "APPROVED_PARTIAL",
                "claimed_amount_usd": 475000.0,
                "applicable_deductible_usd": 75000.0,
                "applicable_sublimit_usd": 2000000.0,
                "approved_net_payout_usd": 400000.0,
                "exclusion_triggered": False,
                "waiting_period_satisfied": True,
                "siu_fraud_flag": False,
            },
        ),
        InsuranceClaimCase(
            claim_id="CLM-2026-9008",
            policy_id="POL-FLT-2026-006",
            product_line="Commercial Heavy Logistics & Telematics Fleet",
            insured_entity="TransContinental Freightways",
            loss_date="2026-11-06",
            reported_date="2026-11-07",
            claimed_amount_usd=380_000.0,
            peril_category="Highway Jackknife Collision with Disabled ELD",
            claim_narrative=(
                "Class-8 tractor jackknifed causing $380,000 damage. Forensic telematics audit revealed "
                "the driver intentionally unplugged the Electronic Logging Device (ELD) 6 hours prior to "
                "falsify FMCSA Hours-of-Service limits and had a suspended CDL."
            ),
            ground_truth_decision="DENIED_EXCLUSION",
            ground_truth_payout_usd=0.0,
            governing_clause_id="CLS-FLT-602",
            expected_json_extraction={
                "claim_id": "CLM-2026-9008",
                "policy_id": "POL-FLT-2026-006",
                "governing_clause_id": "CLS-FLT-602",
                "adjudication_decision": "DENIED_EXCLUSION",
                "claimed_amount_usd": 380000.0,
                "applicable_deductible_usd": 25000.0,
                "applicable_sublimit_usd": 0.0,
                "approved_net_payout_usd": 0.0,
                "exclusion_triggered": True,
                "waiting_period_satisfied": True,
                "siu_fraud_flag": True,
            },
        ),
    ]


def synthesize_long_policy_document(policy_id: str = "POL-CYB-2026-001", target_tokens: int = 32768) -> dict[str, Any]:
    """Generate a realistic long-context (32K+ token scale) master policy contract with buried needles.

    Used to test RoPE Dynamic NTK scaling, Hierarchical Section Indexing, and 'Lost-in-the-Middle'
    retrieval accuracy at 10%, 50%, and 90% document depth.
    """
    clauses = [c for c in build_insurance_policy_clauses() if c.policy_id == policy_id]
    if not clauses:
        clauses = build_insurance_policy_clauses()[:4]

    boilerplate_section = (
        "STANDARD COMMERCIAL INSURANCE DEFINITIONS AND GENERAL CONDITIONS: "
        "1. Audit of Books and Records: The Insurer may examine and audit the Named Insured's financial "
        "ledgers, security architecture logs, and underwriting declarations during the Policy Period. "
        "2. Subrogation & Salvage: In the event of any payment under this Policy, the Insurer shall be "
        "subrogated to all the Insured's rights of recovery against any responsible third party. "
        "3. Valuation & Currency: All premiums, limits, retentions, and loss settlements are denominated "
        "in United States Dollars (USD) unless otherwise endorsed. "
        "4. Territory & Governing Law: Worldwide coverage territory where legally permissible; governed "
        "by the commercial insurance statutes of the State of New York. "
    )

    sections: list[dict[str, Any]] = []
    # Build 40 structured sections with clauses embedded at specific depths (early, middle, late)
    needle_positions = {4: clauses[0], 20: clauses[1], 28: clauses[2], 36: clauses[-1]}
    total_words = 0

    for idx in range(1, 41):
        if idx in needle_positions:
            cl = needle_positions[idx]
            sec_text = f"[{cl.section_code} — {cl.section_title}] ({cl.chunk_id}): {cl.text}\n\n" + (boilerplate_section * 4)
            is_needle = True
            chunk_id = cl.chunk_id
        else:
            sec_text = f"[ARTICLE {idx}.0 — GENERAL UNDERWRITING & ACTUARIAL SCHEDULE PART {idx}]:\n" + (boilerplate_section * 6)
            is_needle = False
            chunk_id = f"GEN-SEC-{idx:02d}"

        words = len(sec_text.split())
        total_words += words
        sections.append(
            {
                "section_index": idx,
                "chunk_id": chunk_id,
                "is_critical_clause": is_needle,
                "depth_percentile": round(idx / 40.0 * 100, 1),
                "approx_token_count": int(words * 1.35),
                "text": sec_text,
            }
        )

    return {
        "policy_id": policy_id,
        "total_sections": len(sections),
        "estimated_context_tokens": max(target_tokens, sum(s["approx_token_count"] for s in sections)),
        "needle_clause_ids": [c.chunk_id for c in needle_positions.values()],
        "sections": sections,
    }


def export_corpus_dicts() -> dict[str, Any]:
    return {
        "clauses": [asdict(c) for c in build_insurance_policy_clauses()],
        "claims": [asdict(c) for c in build_insurance_claims_cases()],
    }
