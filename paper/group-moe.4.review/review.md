# Review: group-moe.4

**Reviewer:** Claude (automated paper review)
**Date:** 2026-04-28
**Paper reviewed:** `paper/group-moe.4/paper.tex`

---

## Overall Assessment: STRONG

**Score: 34/40**

| Dimension | Score | Key Issue |
|-----------|-------|-----------|
| Technical Soundness | 5/5 | Honest 5-seed analysis; claims match evidence |
| Novelty & Contribution | 4/5 | Complement split + architecture are genuine contributions |
| Experimental Rigor | 4/5 | 5 seeds, three-way comparison, multiple ablations |
| Clarity & Writing | 4/5 | Excellent pedagogical structure; abstract is dense |
| Related Work Coverage | 5/5 | Comprehensive; good positioning vs MatrixNet, PEnGUiN |
| Figures & Tables | 3/5 | Fig3 still shows old numbers; needs regeneration |
| Reproducibility | 5/5 | Code URL, hardware, runtimes all stated |
| Presentation & Structure | 4/5 | Strong arc; minor flow issues noted below |

---

## Critical Issues (must fix)

None.

---

## Important Issues (should fix)

1. **Figure 3 still shows old single-seed numbers** (Dimension: 6)
   - Problem: fig3_threeway.png shows 92.1%, 88.2%, 86.1% (old single-seed) and annotations +3.9pp/+2.1pp. Table 2 now says 89.2±3.1%, 90.2±2.0%, 87.0±1.6%. The figure contradicts the table.
   - Recommendation: Regenerate fig3 with the 5-seed means and add error bars to the bars.

2. **The S_2 experiment (Section 4.1) now lacks a three-way comparison** (Dimension: 3)
   - Problem: Section 4.2 has the three-way (GroupMoE vs StandardMoE vs Baseline) but Section 4.1 only compares GroupMoE vs Baseline. A reviewer will ask: is the S_2 complement advantage also attributable to routing rather than group structure?
   - Recommendation: Either (a) run a three-way S_2 experiment and report it, or (b) explicitly note that the S_2 experiment predates the StandardMoE control and the three-way decomposition is only available for S_3. Option (b) is simpler and honest.

3. **Abstract is overloaded** (Dimension: 4)
   - Problem: The abstract tries to convey both the positive results AND the nuanced finding that irreps don't consistently beat learned transforms. The sentence "We find that the routing architecture consistently helps, while the fixed irrep matrices provide the strongest advantage on specific seeds but do not reliably outperform learned transforms across seeds" is an important caveat but makes the abstract read as self-undermining.
   - Recommendation: Shorten the abstract. Lead with what works (routing + complement split + composition), then add one sentence on the nuance. The full analysis belongs in the body.

---

## Suggestions (nice to have)

4. The Section 4.2 "Interpretation" paragraph is the strongest paragraph in the paper. Consider promoting its key insight — "the routing architecture is the primary source of complement transfer at this scale" — to the introduction contributions list.

5. Section 4.3 partially overlaps with Table 2's composition column. Consider merging: present the composition result IN Section 4.2 as part of the three-way table (already done), then make Section 4.3 shorter — focus on the theoretical significance (why composition matters) rather than re-reporting numbers.

6. The S_2 experiment uses num_range=20 with opaque embeddings; the S_3 uses num_range=15 with the same embeddings. A brief sentence explaining why different ranges are used would help (S_2 needs 20 for enough pairs; S_3 has more training data per number at 15 due to triples).

7. Consider a "Summary of Findings" table after Section 4 that lists each experiment, its question, and the one-line answer. This gives readers a quick reference.

---

## Verification of Previous Issues

All critical issues from v1 review and v2 audit have been addressed. The 5-seed results represent a significant improvement in rigor. The honest revision of claims (from "+3.3pp group advantage" to "routing helps, irreps inconsistent") strengthens rather than weakens the paper — reviewers respect honesty.

---

## Next Step

Score is 34/40 with 0 critical issues. Fix fig3, address the abstract density, and the paper is ready for final audit and posting.
