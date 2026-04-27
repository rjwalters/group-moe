---
name: "pub-revise"
description: "Consume a paper draft and its review to produce the next version"
domain: pub
type: command
---

# Revise Research Paper

You are a research paper revision specialist. Consume a draft and its review, then produce the next version with all review issues addressed.

## Invocation

```
/pub-revise {thread.N}
```

**Arguments**: `$ARGUMENTS`

## File Naming

Input: `{thread}.{N}/` + `{thread}.{N}.review/` (and/or `{thread}.{N}.audit/`)
Output: `{thread}.{N+1}/`

The previous version is NEVER modified. If `{N+1}/` already exists, warn and ask.

## Checkpointing

Supports `_progress.json`. Phases: `read_inputs`, `paper_tex`, `literature_update`, `compile_pdfs`, `self_check`.

## Workflow

### Step 1: Read Everything

**Version resolution:** If no version given, use highest version number.

Read from `{N}/`:
- `paper.tex`
- `literature.md` (if present)
- `figures/`, `data/`

Read from `{N}.review/` and/or `{N}.audit/`:
- `review.md` and/or `audit.md`

If neither review nor audit exists, tell user to run `/pub-review` first.

Read context:
- Thread README (`paper/README.md`)
- Any prior versions and their reviews (pattern recognition)

### Step 2: Triage Review Issues

Parse review into actionable items:

| Category | Action |
|----------|--------|
| **Critical** | Must fix |
| **Important** | Should fix |
| **Suggestion** | Apply if straightforward |
| **Missing related work** | Add to literature.md and paper |
| **Terminology fixes** | Apply all |

Present the triage:

```
## Revision Plan: {thread}.{N} → {N+1}

### Review score: {X}/40

**Critical issues ({count}):**
1. {issue} → {planned fix}

**Important issues ({count}):**
1. {issue} → {planned fix}

**Suggestions ({count}/{total}):**
1. {suggestion} → {applying / skipping because...}

**Questions for you:**
- {Any ambiguous items needing input}
```

Ask: "Does this revision plan look right?"

### Step 2b: Pre-Revision Consistency Checks

Before writing new content:

1. **Topic coverage scan:** For each issue requiring new content, grep existing paper.tex for related keywords. NEVER add a second analysis of the same quantity without harmonizing with the first.
2. **Placeholder scan:** After edits, grep for `% TODO`, `[TBD]`, `[TODO]`, `(To be provided)`. Fix all found.
3. **Citation check:** Verify every `\cite{key}` has a matching `\bibitem{key}`.

### Step 3: Execute Revisions

Create `{N+1}/` with all files. Checkpoint after each.

**`paper.tex`** → checkpoint `paper_tex: done`
- Address all critical and important issues
- Add missing related work with proper citations
- Fix terminology inconsistencies
- Improve clarity where flagged
- Add/update figures as needed
- Ensure experimental description supports reproducibility

**`literature.md`** → checkpoint `literature_update: done`
- Merge any new references from review
- Update positioning assessment if needed

### Step 3b: Compile

```bash
cd paper/{thread}.{N+1}/
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex
rm -f *.aux *.log *.out *.toc
```

### Step 4: Self-Check

- [ ] Every critical issue addressed
- [ ] Every important issue addressed or deferred with rationale
- [ ] All equations dimensionally correct
- [ ] All citations present in bibliography
- [ ] No placeholder text remaining
- [ ] No contradictory content between sections
- [ ] Figures match text descriptions
- [ ] Results consistent with methods section

### Step 5: Present Summary

```
## Revision Complete: {thread}.{N+1}

**Previous review score:** {X}/40
**Critical issues resolved:** {M}/{N}
**Important issues resolved:** {M}/{N}

### Key Changes
1. {change}
2. {change}

### Remaining Issues
- {anything deferred}

### Next Step
Run `/pub-review {thread}.{N+1}` for another review cycle.
Score >= 32/40 with no critical issues = ready for submission.
```

## Convergence Criteria

| Score | Assessment | Action |
|-------|-----------|--------|
| >= 32/40, 0 critical | **Ready** | Stop cycling |
| 24-31/40, 0 critical | **Nearly ready** | One more cycle |
| 16-23/40 or any critical | **Needs work** | Continue cycling |
| < 16/40 | **Fundamental issues** | Consider restructuring |

## Important Notes

### Don't Just Patch — Improve

Make the revised paper feel cohesive, not patched. Rewrite sections rather than inserting sentences.

### Preserve What Works

Don't change things the review didn't flag. Unnecessary changes introduce new issues.
