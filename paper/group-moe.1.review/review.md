# Review: group-moe.1

**Reviewer:** Claude (automated paper review)
**Date:** 2026-04-28
**Paper reviewed:** `paper/group-moe.1/paper.tex`

---

## Overall Assessment: NEEDS WORK

**Score: 26/40**

| Dimension | Score | Key Issue |
|-----------|-------|-----------|
| Technical Soundness | 4/5 | Architecture is sound; router-element correspondence needs clarification |
| Novelty & Contribution | 4/5 | Genuine contribution; well-positioned against related work |
| Experimental Rigor | 3/5 | No error bars/confidence intervals; seed 7 result weakens S_2 claim |
| Clarity & Writing | 4/5 | Strong prose; some notation inconsistencies |
| Related Work Coverage | 3/5 | Missing partial/selective equivariance literature |
| Figures & Tables | 1/5 | Zero actual figures — critical gap for an experimental paper |
| Reproducibility | 3/5 | Code described but key details missing from paper text |
| Presentation & Structure | 4/5 | Pedagogical structure works well; appendices are thin |

---

## Critical Issues (must fix)

1. **No figures** (Dimension: 6)
   - Problem: The paper reports 5 experiments but contains zero figures. There is one TODO placeholder. An experimental paper without figures is unpublishable.
   - Impact: Reviewers cannot visually assess training dynamics, router behavior, error patterns, or scaling trends. The paper reads as a technical report, not a research paper.
   - Recommendation: Generate at minimum: (a) architecture diagram, (b) S_2 complement heatmaps (already generated in `data/analysis/`), (c) three-way comparison training curves, (d) router discrimination visualization (nested vs non-nested), (e) compositional generalization bar chart. Run `/pub-figures group-moe.1`.

2. **No error bars or confidence intervals** (Dimension: 3)
   - Problem: All results are single-run numbers except the S_2 seed table. The S_3 three-way comparison (Table 2 — the central result) has no variance estimate. The 3.9pp group structure advantage could be within noise.
   - Impact: A skeptical reviewer will question whether the effects are statistically significant.
   - Recommendation: Run the three-way comparison with at least 3 seeds. Report mean ± std. If the 3.9pp gap is robust across seeds, this becomes a much stronger claim.

3. **Seed 7 result undermines the S_2 claim** (Dimension: 3)
   - Problem: Table 1 shows seed 7 at only +6% relative gain (35.3% vs 33.2% — barely above noise). This is reported alongside +53% and +64%, making the effect look inconsistent.
   - Impact: A reviewer will zero in on this and question whether the effect is robust.
   - Recommendation: Either (a) run more seeds and report the distribution, showing seed 7 is an outlier, or (b) investigate what makes seed 7 different (router behavior, embedding quality) and discuss it explicitly. Do not hide it — the current honest reporting is better than omission, but needs more analysis.

---

## Important Issues (should fix)

4. **Router-element correspondence is underexplained** (Dimension: 1)
   - Problem: The router selects a specific group element $g$, but the paper doesn't explain how the router "knows" which element to select. For a test pair $(b, +, a)$ where the training pair was $(a, +, b)$, the correct element is the transposition $(01)$. But the router sees the concatenated embedding, not the raw indices — how does it determine that the transposition is needed?
   - Recommendation: Add a paragraph explaining that the router's job is NOT to identify the specific permutation but to decide WHETHER to apply the group expert. The element selection is soft (confidence-weighted) and the projection P absorbs the element-specific mapping. This is a subtle but important point that reviewers will probe.

5. **$P$ and $P^\dagger$ are not constrained to be pseudo-inverses** (Dimension: 1)
   - Problem: The notation $P^\dagger$ suggests a pseudo-inverse, but the inject matrix is independently learned. This could confuse readers.
   - Recommendation: Either use different notation (e.g., $Q$) or clarify that $P^\dagger$ denotes a separately learned injection, not the pseudo-inverse of $P$.

6. **The transformer experiment adds little value in its current form** (Dimension: 3)
   - Problem: All models reach 100%, so the experiment doesn't demonstrate any advantage or disadvantage. The "zero-degradation drop-in" claim is interesting but could be made in one sentence rather than a full subsection.
   - Recommendation: Shorten Section 4.5 substantially (merge into Discussion or a brief paragraph in Section 4.4). Use the space for more informative content like a proper scaling analysis figure or additional seeds.

7. **The "LUT vs algebra" analysis is speculative** (Dimension: 1)
   - Problem: The $S_{20}$ with $10^{18}$ elements argument is compelling in theory but the paper provides no empirical evidence for the scaling prediction. The num_range and coverage sweeps (Appendix C) actually show NO crossover — both models converge to 100%.
   - Recommendation: Be more explicit that this is a prediction, not a finding. The current framing is mostly fine but the phrase "the compelling case for Group-MoE is the large-table regime" reads as if evidence exists. Rephrase to clearly mark this as future work motivation.

8. **Missing comparison: equivariant baseline** (Dimension: 3)
   - Problem: The paper compares Group-MoE to a standard MoE and a plain MLP, but never compares to a model with hard-coded equivariance (e.g., a model that averages over all orderings of its input, or uses a DeepSets-style permutation-invariant layer). This is the geometric deep learning baseline.
   - Recommendation: Add a simple equivariant baseline (e.g., sort inputs before processing, or use a sum-pooling layer) to show that Group-MoE achieves similar transfer without the rigidity. This strengthens the "optional symmetry" narrative.

---

## Suggestions (nice to have)

9. The abstract is dense — consider splitting the contribution list into two sentences rather than one long enumeration.

10. Section 3 (Complement Split) could benefit from a small figure showing the split visually: which orderings go to train vs test.

11. The ASIC analogy, while compelling, appears three times (intro, Section 4.4, discussion). Consider consolidating to two occurrences — introduction and discussion — to avoid repetition.

12. Table 1 caption should spell out what "num_range=20" means for readers unfamiliar with the codebase.

13. Consider adding a "Notation" paragraph or table at the start of Section 2.

14. Appendix A is a single sentence — either flesh it out with actual irrep matrices or remove it and reference the codebase directly.

---

## Missing Related Work

- **PEnGUiN (2025)**: Partially equivariant GNN with learnable symmetry scores controlling degree of equivariance per layer. Relevant as another "selective equivariance" approach — competes with Group-MoE's routing mechanism for the same design goal.
  - Relevance: Direct competitor for selective equivariance. Should cite and discuss.
  - Recommendation: Add to Related Work under a new "Selective/partial equivariance" subsection.

- **Symmetry breaking in equivariant networks (2023-2025)**: Puny et al. "Frame Averaging," Kim et al. "Probabilistic Symmetry Breaking." These allow equivariant networks to break symmetry when needed — the inverse of Group-MoE's problem.
  - Relevance: Complementary perspective — they start equivariant and learn when to break; we start non-equivariant and learn when to apply.
  - Recommendation: Cite and discuss as a dual approach.

- **Equivariant fine-tuning (AAAI 2023)**: Equi-Tuning applies group equivariance to pretrained models. Relevant for future work on inserting Group-MoE into pretrained transformers.
  - Relevance: Not critical but useful context.
  - Recommendation: Cite in Discussion/Future Work.

---

## Next Step

Run `/pub-revise group-moe.1` to create version 2 incorporating this review.
