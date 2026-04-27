---
name: "Publication Skills"
description: "Research paper drafting, review, revision, and figure generation -- loaded when working on technical papers, white papers, or conference submissions"
domain: pub
type: skill
user-invocable: false
---

# Publications Domain

This project produces research papers in `paper/`. All publication work follows a structured draft-review-revise cycle with immutable version history and formal scoring.

## State Machine

```
EMPTY --> DRAFTED --> REVIEWED --> REVISED --> REVIEWED --> ... --> READY
```

Commands map to state transitions:

| Transition | Command | Input | Output |
|------------|---------|-------|--------|
| `EMPTY --> DRAFTED` | [[pub-draft]] | User interview + literature search | `{thread}.1/` |
| `DRAFTED --> REVIEWED` | [[pub-review]] | Version `{N}/` | `{N}.review/review.md` |
| `REVIEWED --> REVISED` | [[pub-revise]] | `{N}/` + `{N}.review/` | `{N+1}/` |
| `REVISED --> REVIEWED` | [[pub-review]] | Version `{N+1}/` | `{N+1}.review/review.md` |
| `READY --> AUDITED` | [[pub-audit]] | Version `{N}/` | `{N}.audit/audit.md` |
| any state | [[pub-figures]] | Paper `.tex` | Missing figure scripts in `figures/` |
| portfolio | [[pub]] | Thread name or all | Assessment + parallel agent launch |

Convergence criterion: review score >= 32/40 with 0 critical issues.

## Naming Convention

```
{thread}.{N}
```

- **thread**: the paper topic name (e.g., `group-moe`)
- **N**: integer starting at 1, incremented by `/pub-revise`

Example: `group-moe.3` is the 3rd revision of the Group-MoE paper.

## Directory Layout

For this project, the thread name is `group-moe`. Paper versions live in `paper/`:

```
paper/
  outline.md                     # Working outline (paper_outline.md in docs/)
  group-moe.1/                   # Draft version 1
    paper.tex                    # LaTeX paper (standard article class)
    paper.pdf                    # Compiled PDF
    literature.md                # Literature review notes (internal)
    figures/                     # Python-generated figures (.py --> .pdf/.png)
    data/                        # Supporting scripts and results
  group-moe.1.review/            # Review (read-only sibling)
    review.md                    # Scored review report (markdown)
  group-moe.2/                   # Revision incorporating review
    ...                          # Same structure
```

### Format Convention

- **LaTeX** (`.tex` --> `.pdf`): paper uses standard `\documentclass{article}` with academic packages (geometry, amsmath, booktabs, graphicx, hyperref). NOT `sphere-patent.sty`
- **Markdown** (`.md`): internal working documents -- `literature.md`, `review.md`
- **Python figures** (`.py` --> `.pdf` + `.png`): raw matplotlib for data plots
- Compile: `pdflatex paper.tex` (run twice for refs)

### Key Principles

1. **Literature review first.** `/pub-draft` conducts a literature search before writing. Understand the landscape and position the contribution.
2. **Immutable versions.** Previous versions are never modified. The version history IS the revision trail.
3. **Separation of concerns.** Review is read-only (produces `{N}.review/`). Revision is separate (consumes `{N}/` + `{N}.review/`, produces `{N+1}/`).
4. **Converge, then submit.** Cycle until review score >= 32/40 with 0 critical issues.

## Commands

| Command | Description |
|---------|-------------|
| [[pub]] | Portfolio orchestrator: assess state of all papers, launch parallel agents |
| [[pub-draft]] | Interview + literature search + first-draft paper.tex (creates version 1) |
| [[pub-review]] | Read-only critic: 8-dimension scored review report |
| [[pub-revise]] | Consume draft + review, produce next version with all issues addressed |
| [[pub-audit]] | Fact-check: verify citations, numbers, equations, reproducibility claims |
| [[pub-figures]] | Batch-generate missing figures for a paper |

## Checkpointing Protocol

Long-running skills (`/pub-draft`, `/pub-revise`, `/pub-figures`) write `_progress.json` in the output directory. On retry, completed phases are skipped.

| Skill | Checkpointed Phases |
|-------|-------------------|
| `/pub-draft` | `interview`, `literature_search`, `paper_tex`, `compile_pdfs` |
| `/pub-revise` | `read_inputs`, `paper_tex`, `literature_update`, `compile_pdfs`, `self_check` |
| `/pub-figures` | `analysis`, then per-figure: `fig_N_script`, `fig_N_run`, `fig_N_verify` |
| `/pub-review` | Not checkpointed (single output file) |
| `/pub-audit` | Not checkpointed (single output file) |

## Key References

- `docs/paper_outline.md` -- comprehensive paper outline with all experimental results
- `docs/arithmetic_experiment.md` -- S_2 complement transfer results
- `docs/ternary_experiment.md` -- S_3 complement transfer results
- `data/` -- JSON files with all experimental outputs (gitignored)
- `scripts/` -- experiment scripts (source of all reported results)
