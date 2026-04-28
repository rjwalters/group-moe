# Review: group-moe.3

**Reviewer:** Claude (automated paper review)
**Date:** 2026-04-28
**Paper reviewed:** `paper/group-moe.3/paper.tex`

---

## Overall Assessment: NEARLY READY

**Score: 32/40**

| Dimension | Score | Key Issue |
|-----------|-------|-----------|
| Technical Soundness | 4/5 | Sound; router-element paragraph is a good addition |
| Novelty & Contribution | 4/5 | Clear contribution; well-positioned vs MatrixNet |
| Experimental Rigor | 4/5 | Error bars added; 3 runs is minimal but sufficient for this venue |
| Clarity & Writing | 4/5 | Strong pedagogical structure; minor issues below |
| Related Work Coverage | 4/5 | PEnGUiN, frame averaging added; comprehensive now |
| Figures & Tables | 3/5 | 5 figures + 2 tables; fig3 annotations need updating |
| Reproducibility | 4/5 | Good detail; code availability should be stated |
| Presentation & Structure | 5/5 | Excellent pedagogical arc; each experiment answers one question |

---

## Critical Issues (must fix)

None. Previous critical issues (wrong router numbers, Table 2 discrepancies) have been addressed.

---

## Important Issues (should fix)

1. **Figure 3 annotations don't match corrected text** (Dimension: 6)
   - Problem: The fig3_threeway.png still shows the old decomposition (+3.9pp group, +2.1pp routing) while the text now says +3.3pp and +1.7pp. The bar heights also show the old numbers (92.1%, 88.2%, 86.1%) instead of the corrected means (91.8%, 88.5%, 86.8%).
   - Recommendation: Regenerate fig3 with corrected numbers and error bars.

2. **Code/data availability not stated** (Dimension: 7)
   - Problem: The paper describes experiments in detail but never states whether code or data will be released. For an arXiv preprint, this is expected.
   - Recommendation: Add a sentence at the end of Section 4 or in the Conclusion: "Code and experimental scripts are available at [URL]."

3. **Table 2 error bar formatting** (Dimension: 6)
   - Problem: "$\pm$ 0.1\%" reads as "plus or minus 0.1 percent" which could be ambiguous — is it 0.1 percentage points or 0.1% of the value? For clarity, use "pp" or format as "91.8\% ($\pm$0.1)".
   - Recommendation: Use the format "91.8\% {\scriptsize($\pm$0.1)}" for clarity.

---

## Suggestions (nice to have)

4. The S_2 router discrimination (70% vs 64%) is weaker than the paper's framing suggests. Consider acknowledging this more directly — "modest but consistent" is better than implying strong discrimination.

5. The abstract still references "+16 percentage points for $S_2$" but the body text says "53% relative improvement" (seed 42). The 16pp is 48.4-31.6=16.8pp. Consider rounding to 17pp for accuracy.

6. The composition result "98.5--99.0%" in Section 4.3 text doesn't match the table (99.2 ± 0.5%). Update the text to reference the table.

7. Consider adding a brief "Reproducibility" statement as the last subsection of Section 4, noting seeds, hardware, and training time.

---

## Verification of Previous Issues

| # | Previous Issue | Status |
|---|---------------|--------|
| 1 | No figures | **FIXED** — 5 figures included |
| 2 | No error bars | **FIXED** — 3-run mean ± std in Table 2 |
| 3 | Seed 7 undermines S_2 | **FIXED** — seed variability analyzed explicitly |
| 4 | Router-element underexplained | **FIXED** — paragraph added |
| 5 | P† notation misleading | **FIXED** — renamed to Q |
| 6 | Transformer section too long | **FIXED** — shortened to one paragraph |
| 7 | LUT analysis speculative | **FIXED** — marked as untested prediction |
| 8 | Missing equivariant baseline | **Not addressed** — noted below |
| MW | Missing related work | **FIXED** — PEnGUiN, frame averaging added |
| A1 | S_2 router stats wrong | **FIXED** — corrected to 70%/64% |
| A2 | Table 2 numbers wrong | **FIXED** — corrected to verified three-way values |
| A5 | PEnGUiN citation | **FIXED** — authors, volume, pages added |

**Remaining from v1 review:** Issue #8 (missing equivariant baseline) was not addressed. This is a valid gap but acceptable for an arXiv preprint — it can be noted as future work. The paper already compares to StandardMoE (learned transforms) which is the more informative comparison.

---

## Next Step

Score is 32/40 with 0 critical issues — meets convergence criteria. Fix the important issues above (fig3 update, code availability, formatting), then the paper is ready for `/pub-audit` as a final check before posting.
