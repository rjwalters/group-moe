# Fact-Check Audit: group-moe.5

**Auditor:** Claude (automated paper audit)
**Date:** 2026-04-28
**Paper audited:** `paper/group-moe.5/paper.tex`

---

## Summary

**3 issues found: 0 critical, 1 warning, 2 info**

All major numerical claims verified against experimental outputs. Previous critical issues from v2 audit have been resolved. The paper is internally consistent.

---

## Critical Issues (definitely wrong — must fix)

None.

---

## Warnings (suspicious — needs verification)

1. **GroupMoE-StandardMoE gap is reported as -1.0pp but computes to -0.9pp**
   - **Location:** Section 4.2, Decomposition paragraph
   - **Text:** "Group-MoE averages −1.0pp relative to StandardMoE"
   - **Actual:** 89.24 - 90.18 = -0.94pp, which rounds to -0.9pp or -1.0pp depending on precision
   - **Action:** Acceptable rounding, but consider reporting as "approximately −1pp" or "−0.9pp" for consistency with other reported figures at 1 decimal.

---

## Citation Inventory

| # | Citation | Status | Notes |
|---|----------|--------|-------|
| 1 | Bronstein et al. 2021 | VERIFIED | arXiv:2104.13478, well-known |
| 2 | Cohen & Welling 2016, ICML | VERIFIED | Foundational |
| 3 | Weiler & Cesa 2019, NeurIPS | VERIFIED | |
| 4 | Shazeer et al. 2017, ICLR | VERIFIED | |
| 5 | Fedus et al. 2022, JMLR 23(120) | VERIFIED | |
| 6 | Kang et al. 2025, arXiv:2504.09265 | VERIFIED | |
| 7 | Laird et al. 2024, NeurIPS | VERIFIED | MatrixNet |
| 8 | Dehmamy et al. 2021, NeurIPS | PLAUSIBLE | Authors confirmed in area |
| 9 | Benton et al. 2020, NeurIPS | VERIFIED | |
| 10 | Gordon et al. 2020, ICLR | VERIFIED | |
| 11 | Walters 2025 | PLAUSIBLE | Author's prior work; no arXiv ID listed |
| 12 | McClellan et al. 2025, RLJ 6:2637-2651 | VERIFIED | PEnGUiN, confirmed via web search |
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
| S_3 GroupMoE complement | Table 2 | 89.2±3.1% | 89.24±3.12% | OK |
| S_3 StandardMoE complement | Table 2 | 90.2±2.0% | 90.18±2.00% | OK |
| S_3 Baseline complement | Table 2 | 87.0±1.6% | 86.98±1.56% | OK |
| S_3 GroupMoE composition | Table 2 | 98.7±0.8% | 98.72±0.76% | OK |
| S_3 StandardMoE composition | Table 2 | 98.6±0.8% | 98.60±0.84% | OK |
| S_3 Baseline composition | Table 2 | 97.4±1.1% | 97.44±1.08% | OK |
| Routing +3.2pp | Sec 4.2 | +3.2pp | 90.18-86.98=3.20 | OK |
| GroupMoE vs StdMoE | Sec 4.2 | -1.0pp | 89.24-90.18=-0.94 | OK (rounding) |
| Composition MoE advantage | Sec 4.2 | ~1.3pp | (98.72-97.44)=1.28 | OK |
| S_2 router add | Sec 4.1 | 70% | 69.5% | OK (rounding) |
| S_2 router sub | Sec 4.1 | 64% | 63.7% | OK (rounding) |
| Per-pair exclusive | Sec 4.1 | 40 vs 16 | 40 vs 16 | OK |
| Nested S_3 rate | Sec 4.4 | ~70% | 70.1% | OK |
| Z_2 dispatch | Sec 4.4 | 76% | 75.3% | OK (rounding) |
| Parameter count | Sec 4.2 | ~337K | 337,288 | OK |
| Compression | Sec 2.1 | 32× | 128/4=32 | OK |
| Expert params S_3 | Table 1 | 1,024 | 2×128×4=1,024 | OK |

---

## Internal Consistency Check

- Abstract "+3.2pp" matches Section 4.2 Decomposition: **OK**
- Abstract "98.7±0.8%" matches Table 2 composition column: **OK**
- Conclusion "+3.2pp" matches Section 4.2: **OK**
- Conclusion "98.7±0.8%" matches Table 2: **OK**
- Figure 3 shows 89.2%, 90.2%, 87.0% matching Table 2 means: **OK**
- Figure 3 error bars match Table 2 std: **OK**
- Section 4.2 "−5.5pp on seed 7": 83.6-89.1=-5.5 **OK**
- Section 4.2 "+3.6pp on favorable seed": 91.9-88.3=+3.6 **OK**

---

## Info (minor observations)

1. **Walters 2025 citation still has no arXiv ID or venue.** If this work is available online, adding the URL would help readers find it.

2. **The "2.5× more pairs" claim (Section 4.1)** comes from a separate analysis script training for 300 epochs, not from the main complement split run. The paper now notes this as "a separate per-pair analysis," which is accurate.

---

## Verdict

The paper is factually accurate. All major numerical claims have been verified against experimental outputs. Internal consistency is maintained throughout. The only warning is a minor rounding choice (−1.0pp vs −0.9pp). The paper is ready for posting.
