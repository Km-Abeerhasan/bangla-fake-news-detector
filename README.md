# BiN: Evidence-Calibrated Multimodal Fact Verification with Audit Gates

**Department of Computer Science and Engineering | Jahangirnagar University**  
Master of Science in Computer Science & Engineering (PMCS Program, Batch 42)

* **Candidate:** Khan Muhammad Abeer Hasan (Student ID: CSE2026095)
* **Project Supervisor:** Dr. Md. Humayun Kabir, B.Sc (DU), M.Sc (DU), PhD (Ireland)  
  *Professor, Dept. of CSE, JU (Research: Data Mining, Automatic Program Construction, Software Architecture)*
* **Course Coordinator:** Dr. Md. Ezharul Islam, B.Sc (JU), M.Engg (Japan), PhD (Japan)  
  *Professor, Dept. of CSE, JU (Research: IoT, Machine Learning, Artificial Intelligence)*
* **Official Repository:** `https://github.com/Km-Abeerhasan/bin-audit-then-believe`
* **Local Hardware Target:** Apple Silicon M1 (16 GB Unified Memory), PyTorch MPS

---

## 1. Core Scientific Axiom
> *"Confidence should not exceed what retrieved empirical evidence mathematically and factually supports."*

BiN addresses low-resource and multimodal misinformation (such as manipulated screenshot cards, altered official gazettes, and cross-platform rumors). It eliminates ungrounded generative predictions and opaque probability scores by implementing an auditable 8-stage verification pipeline with calibrated abstention.

---

## 2. Decoupled Verdict Schema

### Factual Output Labels
* `SUPPORTED`: Credible independent evidence directly entails the proposition.
* `REFUTED`: Credible independent evidence directly contradicts the proposition.
* `MISLEADING`: Distorts a material aspect (scale, timeframe, units, attribution).
* `CONFLICTING`: Credible independent sources support materially incompatible conclusions.
* `INSUFFICIENT_EVIDENCE`: Audited search within snapshot scope confirms evidence cannot resolve the claim.

### Operational Pipeline Statuses
* `NON_FACTUAL`: Personal opinion, satire, or speculative commentary.
* `RETRIEVAL_DEFICIT`: Query execution aborted, failed, or timed out.
* `ABSTAIN`: Automated gate triggers abstention due to low epistemic coverage.
* `HUMAN_REVIEW`: Safety-critical or high-discrepancy claim routed to manual audit.

---

## 3. Modular Architecture (Stages S1–S8)
1. **Offline Store:** Dated snapshot registry across 4 source tiers (Official/Govt, Fact-Checkers, Mainstream News, General Web) with point-in-time leakage protection ($T_{\text{claim}}, T_{\text{event}}, T_{\text{pub}}, T_{\text{verif}}$).
2. **S1 (Ingestion & OCR Recovery):** Default Unicode NFC text normalization with reversible audit log; script-specific EasyOCR with recovery routing for low-confidence tokens.
3. **S2 (Claim Processing):** Modality typing (Factual, Opinion, Satire, Prediction), rule-based proposition splitting, and 4-way temporal anchoring.
4. **S3 (Hybrid Dual-Pass Retrieval):** BM25 (Bengali-tuned) + dense `multilingual-e5-base` with Reciprocal Rank Fusion (RRF) over Pass A (written) & Pass B (balanced contrastive).
5. **S4 (Evidence Audit):** MinHash/SimHash deduplication clustering, publisher independence counting, and temporal eligibility filtering.
6. **S5 (Evidence Analysis):** XLM-RoBERTa passage-level stance estimation (IndicXNLI fine-tuned) + temperature scaling + deterministic regex/table numeric module.
7. **S6 (Decision Aggregator):** Deterministic 9-priority decision table logging `rule_fired` and secondary flags.
8. **S7 (Multi-Signal Reliability Gate):** Trained on grouped out-of-fold (OOF) cross-validation splits to optimize selective classification (AURC).
9. **S8 (Report & Citation Audit):** Multi-tier citation audit (exact-span match, provenance, URL, NLI stance) with zero free-form text generation.
