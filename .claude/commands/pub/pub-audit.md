---
name: "pub-audit"
description: "Fact-check a research paper for hallucinated citations, fabricated numbers, and internal inconsistencies"
domain: pub
type: command
---

# Audit Research Paper

You are a technical fact-checker auditing a research paper for **fabricated, hallucinated, or nonsensical content**. Complementary to `/pub-review` (which scores drafting quality) — your job is to verify every factual claim is actually true.

LLM-drafted papers commonly contain:
- Fabricated citations (made-up papers, wrong authors, nonexistent venues)
- Incorrect performance claims (numbers that violate physics or arithmetic)
- Internal inconsistencies (numbers in one section contradicting another)
- Fake equations (dimensionally wrong or mathematically meaningless)
- Invented terminology presented as established

## Invocation

```
/pub-audit {thread.N}
```

**Arguments**: `$ARGUMENTS`

If no version specified, audit the latest. If just a thread name, audit the latest version.

## Locate the Paper

Same resolution as `/pub-review`: `paper/{thread}.{N}/`

Read ALL documents: `paper.tex`, `literature.md`, figures, data.

## Audit Checklist

### 1. Citation Verification (HIGHEST PRIORITY)

For EVERY citation in the paper:

**Academic papers:**
- Is the journal/conference real?
- Are the authors real people in this field?
- Does the paper title match the claimed venue and date?
- Do DOIs resolve? Are arXiv IDs plausible?
- Are there vague citations ("IEEE 2020") — hallucination red flags?

Classify each:
- **VERIFIED**: Confident this is real (well-known work, famous authors)
- **PLAUSIBLE**: Sounds right, cannot independently confirm
- **SUSPICIOUS**: Vague venue, approximate dates, unknown journal
- **LIKELY FABRICATED**: Journal doesn't exist, dates impossible, contradicts known facts

### 2. Numerical Claims

**Arithmetic:**
- Re-derive every calculation
- Verify unit consistency
- Check totals equal sums of parts

**Physics / domain knowledge:**
- Do claimed values make sense for the domain?
- Are performance comparisons fair (same conditions)?
- Are statistical claims (p-values, confidence intervals) used correctly?

**Experimental results:**
- Are results reproducible from described methodology?
- Do figures match numbers in the text?
- Are error bars / variance measures consistent?

### 3. Internal Consistency

- Same quantity, different values across sections?
- Claims vs. abstract — do they match?
- Figures vs. text — do captions match descriptions?
- Tables vs. text — do numbers agree?

### 4. Equations

- Dimensional analysis on every equation
- Verify standard formulas are stated correctly
- Check that notation is defined before use
- Verify that derived results follow from premises

### 5. Reproducibility Claims

- Is claimed code/data availability real?
- Are described experimental setups plausible?
- Are hyperparameters/settings actually sufficient to reproduce?

## Output

Write to `paper/{thread}.{N}.audit/audit.md`:

```markdown
# Fact-Check Audit: {thread}.{N}

**Auditor:** Claude (automated paper audit)
**Date:** {date}
**Paper audited:** `{path}`

---

## Summary

**{N} issues found: {C} critical, {W} warning, {I} info**

{1-2 sentence overall assessment}

---

## Critical Issues (definitely wrong — must fix)

1. **{Issue title}**
   - **Location:** {section, line if possible}
   - **Text:** "{the problematic text}"
   - **Problem:** {why it's wrong}
   - **Fix:** {specific correction}

---

## Warnings (suspicious — needs verification)

1. **{Issue title}**
   - **Location:** {section}
   - **Concern:** {why suspicious}
   - **Action:** {what to verify}

---

## Citation Inventory

| # | Citation | Status | Notes |
|---|----------|--------|-------|
| 1 | {author, title, venue, year} | VERIFIED / PLAUSIBLE / SUSPICIOUS / FABRICATED | {notes} |

---

## Numerical Verification Log

| Claim | Location | Expected | Actual | Status |
|-------|----------|----------|--------|--------|
| {description} | {section} | {claimed} | {calculated} | OK / ERROR / UNVERIFIED |

---

## Info (minor observations)

1. {observation}

---

## Recommendations

1. {prioritized actions before submission}
```

Present a summary to the user.

## Important Notes

### Be Skeptical, Not Paranoid

Recent papers (2025-2026) may be real but outside training data. Flag as PLAUSIBLE or SUSPICIOUS, not LIKELY FABRICATED, unless there are specific red flags.

### Relationship to /pub-review

| Aspect | /pub-review | /pub-audit |
|--------|-----------|-----------|
| Focus | Writing quality, contribution, rigor | Factual accuracy, citation validity |
| Output | Scored review (X/40) | Issue inventory with severity |
| Timing | After each draft/revision | Before submission, after convergence |

Run `/pub-review` iteratively during drafting. Run `/pub-audit` once converged (>= 32/40) as final quality gate.
