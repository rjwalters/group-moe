# Review: group-moe.5

**Reviewer:** Claude (automated paper review)
**Date:** 2026-04-28
**Paper reviewed:** `paper/group-moe.5/paper.tex`

---

## Overall Assessment: STRONG

**Score: 36/40**

| Dimension | Score | Key Issue |
|-----------|-------|-----------|
| Technical Soundness | 5/5 | Honest claims, all match evidence |
| Novelty & Contribution | 4/5 | Architecture + complement split are genuine; results are modest |
| Experimental Rigor | 5/5 | 5-seed error bars, three-way comparison, honest negative findings |
| Clarity & Writing | 4/5 | Clean pedagogical arc; two minor prose issues |
| Related Work Coverage | 5/5 | Comprehensive and well-positioned |
| Figures & Tables | 4/5 | Fig3 now matches data; one remaining issue |
| Reproducibility | 5/5 | Code URL, hardware, seeds, runtimes |
| Presentation & Structure | 4/5 | Strong overall; section 4.3 partially redundant with Table 2 |

---

## Critical Issues (must fix)

None.

---

## Important Issues (should fix)

1. **Section 4.3 is partially redundant with Section 4.2** (Dimension: 8)
   - Problem: Table 2 already reports the composition split numbers (98.7±0.8% etc.) as a column. Section 4.3 re-reports the same 98.7±0.8% number and adds only two sentences of analysis (the projection P generalizes, and the router achieves 100%/74% discrimination at num_range=10). This doesn't warrant its own subsection.
   - Recommendation: Merge Section 4.3 into Section 4.2 as a closing paragraph. The subsection header "Does irrep composition generalize?" is a great question — answer it within the three-way section where the data already lives. This tightens the paper and avoids the impression of padding.

2. **The 100%/74% router claim in Section 4.3 is from num_range=10, not 15** (Dimension: 1)
   - Problem: All other S_3 results use num_range=15. This section silently switches to num_range=10 for the router statistic. A reader who notices will wonder why.
   - Recommendation: Either report the num_range=15 router stats (which we know from the 5-seed runs), or note explicitly: "At num_range=10, where the task is harder and the signal is clearer, the router achieves..."

3. **"Algebraic structure helps when" list in Section 5 needs updating** (Dimension: 4)
   - Problem: Item (1) says "the complement split isolates symmetry transfer from memorization" — but the complement split is a methodology, not a condition for algebraic structure helping. The 5-seed results show that even with the complement split, algebraic structure (irreps specifically) doesn't reliably beat learned transforms.
   - Recommendation: Rewrite the "helps when" list to be more precise: (1) the composition property is needed (not just general transfer), (2) the group is large enough that learned transforms can't approximate the structure, (3) the router initialization supports exploitation of the irrep basis.

---

## Suggestions (nice to have)

4. Consider adding a one-paragraph "Limitations" subsection before the Conclusion. The paper is honest throughout, but consolidating limitations in one place helps reviewers find them and signals maturity. Key limitations: (a) synthetic tasks only, (b) sum function doesn't create combinatorial challenge, (c) irreps don't consistently beat learned transforms at this scale, (d) S_2 experiment lacks three-way comparison.

5. The figure caption for fig2 (S_2 heatmaps) mentions "seed=42" but doesn't note this is one of the stronger seeds. A parenthetical "(one of the higher-performing seeds; see Table 1 for seed variability)" would preempt reviewer concern about cherry-picking.

6. In the Discussion, the "Symmetry discovery" paragraph could briefly mention that MatrixNet~\cite{laird2024matrixnet} already addresses part of this — learning representations from data — and note the potential for combining MatrixNet's learned representations with Group-MoE's learned routing.

7. The paper would benefit from a brief notation table or paragraph at the start of Section 2: $d$ = model dimension, $k$ = irrep total dimension, $G$ = group, $|G|$ = order, $P$ = projection, $Q$ = injection, $\alpha$ = confidence. Currently these are introduced inline but a reader jumping to the equations may be lost.

---

## What's Working Well

- **The honesty is the paper's greatest strength.** The 5-seed results that show StandardMoE matching Group-MoE — reported straightforwardly, not hidden — make the paper more credible than a paper claiming irreps always win. Reviewers trust authors who report inconvenient findings.

- **The complement split** is a genuine methodological contribution. Even if the Group-MoE architecture itself doesn't dominate, the complement split is useful for anyone studying symmetry in neural networks.

- **The LUT vs algebra framework** provides a clear intellectual framework that makes predictions beyond the current experiments. The paper isn't just reporting results — it's building a theory.

- **The ASIC analogy** is memorable and precise. It gives readers a mental model that persists after they put the paper down.

---

## Next Step

Score is 36/40 with 0 critical issues. Address the important items (merge 4.3 into 4.2, fix the num_range discrepancy, update the "helps when" list), then run `/pub-audit` as the final gate before posting.
