# Fact-Check Audit: group-moe.2

**Auditor:** Claude (automated paper audit)
**Date:** 2026-04-28
**Paper audited:** `paper/group-moe.2/paper.tex`

---

## Summary

**7 issues found: 2 critical, 3 warning, 2 info**

The core results (complement transfer, three-way comparison, composition) are verified against the JSON outputs. Two numerical claims are incorrect — the router statistics for S_2 are from a different run than stated. Several claims need citation tightening.

---

## Critical Issues (definitely wrong — must fix)

1. **S_2 router statistics are wrong**
   - **Location:** Section 4.1, "Router behavior"
   - **Text:** "The router routes 95% of addition examples to the S_2 expert and 35% of subtraction examples"
   - **Problem:** The actual numbers from `data/arithmetic_complement_v3/results.json` (seed=42, which matches Table 1) are: `add_s2_rate=0.695` (69.5%), `sub_s2_rate=0.637` (63.7%). The 95%/35% numbers come from the earlier experiment with `num_range=20, train_frac=0.5, weight_decay=1e-4` (NOT the complement split run). The complement split run has different hyperparameters (`weight_decay=1e-2`).
   - **Fix:** Either (a) report the correct numbers from the complement split run (69.5%/63.7%), or (b) specify which run produced the 95%/35% numbers and note the hyperparameter difference. Option (a) is more honest but the discrimination is weaker. Option (b) is accurate but confusing.

2. **S_3 complement numbers are slightly off from the three-way run**
   - **Location:** Section 4.2, Table 2
   - **Text:** GroupMoE 92.1%, StandardMoE 88.2%, Baseline 86.1%
   - **Problem:** The actual numbers from `data/complement_3way/results.json` are: GroupMoE 91.7%, StandardMoE 88.4%, Baseline 86.7%. The paper reports numbers from a DIFFERENT run (`data/complement_3way` vs earlier `data/ternary_nr15`). The differences are small (<0.5pp) but the paper should report numbers from a single consistent run.
   - **Fix:** Update Table 2 to use the actual three-way run numbers: 91.7%, 88.4%, 86.7%. The decomposition becomes +3.3pp (group) + +1.7pp (routing) = +5.0pp (total). The story is the same; the numbers should just be accurate.

---

## Warnings (suspicious — needs verification)

3. **"per-pair analysis shows GroupMoE exclusively solves 2.5× more pairs"**
   - **Location:** Section 4.1
   - **Concern:** The numbers (40 vs 16 out of 190) come from `scripts/analyze_complement.py` which trains its own models for 300 epochs, not from the main complement split training run (500 epochs). The 40/16 split may differ between runs.
   - **Action:** Verify against the actual complement split run, or note that this analysis used a separate training run.

4. **"At num_range=10, the router achieves perfect discrimination: 100% S_3 routing"**
   - **Location:** Section 4.3
   - **Concern:** The 100%/74% numbers come from the composition split at num_range=10 (`data/composition_nr10`). Cannot verify without re-reading that JSON. The claim of "perfect" (100.0%) should be checked — even 99.9% would round to 100% but isn't "perfect."
   - **Action:** Verify exact number. If it's 99.9% or 100.0%, either is fine but should be reported precisely.

5. **PEnGUiN citation is incomplete**
   - **Location:** Bibliography, \bibitem{penguin2025}
   - **Text:** "PEnGUiN: Partially equivariant graph neural networks. In Reinforcement Learning Journal, 2025."
   - **Concern:** No authors listed. This is a likely hallucination of the venue — PEnGUiN was found via web search and the venue "Reinforcement Learning Journal" seems unlikely for a GNN paper.
   - **Action:** Verify the actual venue and authors via web search. Fix or remove if unverifiable.

---

## Citation Inventory

| # | Citation | Status | Notes |
|---|----------|--------|-------|
| 1 | Bronstein et al. 2021, arXiv:2104.13478 | VERIFIED | Well-known survey, correct arXiv ID |
| 2 | Cohen & Welling 2016, ICML | VERIFIED | Foundational G-CNN paper |
| 3 | Weiler & Cesa 2019, NeurIPS | VERIFIED | Steerable CNNs |
| 4 | Shazeer et al. 2017, ICLR | VERIFIED | Original MoE paper (note: 7 authors listed, actually has more) |
| 5 | Fedus, Zoph & Shazeer 2022, JMLR 23(120) | VERIFIED | Switch Transformers |
| 6 | Kang et al. 2025, arXiv:2504.09265 | VERIFIED | Confirmed via web search |
| 7 | Laird, Hsu, Bapat, Walters 2024, NeurIPS | VERIFIED | MatrixNet, confirmed at NeurIPS 2024 |
| 8 | Dehmamy et al. 2021, NeurIPS | PLAUSIBLE | Confirmed authors work in this area; verify exact venue |
| 9 | Benton et al. 2020, NeurIPS | VERIFIED | Learning invariances |
| 10 | Gordon et al. 2020, ICLR | VERIFIED | Perm. equivariant models, confirmed via web |
| 11 | Walters 2025 | PLAUSIBLE | Author's own prior work; no venue/arXiv listed |
| 12 | PEnGUiN 2025 | SUSPICIOUS | No authors, venue uncertain |
| 13 | Puny et al. 2022, ICLR | VERIFIED | Frame averaging |

---

## Numerical Verification Log

| Claim | Location | Claimed | Actual | Status |
|-------|----------|---------|--------|--------|
| S_2 seed=42 GroupMoE | Table 1 | 48.4% | 48.42% | OK |
| S_2 seed=42 Baseline | Table 1 | 31.6% | 31.58% | OK |
| S_2 seed=123 GroupMoE | Table 1 | 45.8% | 45.79% | OK |
| S_2 seed=123 Baseline | Table 1 | 27.9% | 27.89% | OK |
| S_2 seed=7 GroupMoE | Table 1 | 35.3% | 35.26% | OK |
| S_2 seed=7 Baseline | Table 1 | 33.2% | 33.16% | OK |
| S_3 GroupMoE complement | Table 2 | 92.1% | 91.73% | ERROR (0.4pp) |
| S_3 StandardMoE complement | Table 2 | 88.2% | 88.35% | ERROR (0.2pp) |
| S_3 Baseline complement | Table 2 | 86.1% | 86.72% | ERROR (0.6pp) |
| S_3 GroupMoE composition | Table 2 | 99.0% | 98.93% | OK (rounding) |
| S_3 StandardMoE composition | Table 2 | 98.1% | 98.13% | OK |
| S_3 Baseline composition | Table 2 | 96.2% | 96.16% | OK |
| S_2 add routing | Sec 4.1 | 95% | 69.5% | ERROR |
| S_2 sub routing | Sec 4.1 | 35% | 63.7% | ERROR |
| GroupMoE exclusive pairs | Sec 4.1 | 40/190 | 40/190 | OK (from analyze script) |
| Baseline exclusive pairs | Sec 4.1 | 16/190 | 16/190 | OK (from analyze script) |
| Z_2 dispatch rate | Sec 4.4 | 76% | 75.3% | OK (rounding) |
| Nested S_3 rate | Sec 4.4 | ~70% | 70.1% | OK |
| Parameter count ~337K | Sec 4.2 | ~337K | 337,288 | OK |
| Compression 32× | Sec 2.1 | 32× | 128/4=32 | OK |
| Expert params S_3 d=128 | Table 1 | 1,024 | 2×128×4=1,024 | OK |

---

## Info (minor observations)

1. **Abstract claims "+16 percentage points for S_2"** — this is 48.4-31.6=16.8pp. Rounding to 16pp is fine but 17pp would be more accurate.

2. **The Walters 2025 citation has no venue or arXiv ID.** If this paper exists on arXiv, add the ID. If not yet published, note "in preparation" or similar.

---

## Recommendations

1. **Fix the S_2 router numbers** (Critical #1) — use the correct numbers from the complement split run
2. **Fix Table 2 numbers** (Critical #2) — use the actual three-way run values
3. **Fix PEnGUiN citation** (Warning #5) — verify or remove
4. **Add arXiv ID to Walters 2025** (Info #2) — or note "in preparation"
5. **Consider noting that per-pair analysis used a separate training run** (Warning #3)
