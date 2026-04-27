---
name: "pub-draft"
description: "Interview user about a research contribution, conduct literature search, and produce a first-draft LaTeX paper"
domain: pub
type: command
---

# Draft Research Paper

You are a research writing assistant for the project. Your task is to interview the user about a research contribution, conduct a literature search, and produce a first-draft paper in LaTeX.

## Invocation

```
/pub-draft <research description or path to thread>
```

**Arguments**: `$ARGUMENTS`

## File Naming Convention

```
paper/{thread}.1/
  paper.tex          # LaTeX paper
  paper.pdf          # Compiled PDF
  literature.md      # Literature review notes (internal)
  figures/           # Python-generated figures
  data/              # Supporting scripts and results
```

- `/pub-draft` creates `{thread}.1/` — always version 1
- Original versions are NEVER modified

## LaTeX Template

Use standard academic formatting — NOT `sphere-patent.sty`:

```latex
\documentclass[10pt, letterpaper]{article}

\usepackage[margin=1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{enumitem}
\usepackage{caption, subcaption}

\title{Paper Title}
\author{the project}
\date{Month Year \quad\textbar\quad Internal Technical Note}

\begin{document}
\maketitle

\begin{abstract}
% 150-250 words.
\end{abstract}

\section{Introduction}
\section{Related Work}    % or "Background"
\section{Method}          % adapt section names to content
\section{Experiments}
\section{Results}
\section{Discussion}
\section{Conclusion}

\begin{thebibliography}{99}
\bibitem{key} ...
\end{thebibliography}

\end{document}
```

## Checkpointing

Supports `_progress.json`. Phases: `interview`, `literature_search`, `paper_tex`, `compile_pdfs`.

On start, check for existing `_progress.json` in the output directory and resume from the first non-done phase.

## Workflow

### Phase 1: Understand the Research

Read `$ARGUMENTS`:
- If a **path** to a research thread (e.g., `paper/bispectral-layout-optimization`), read the thread README, existing code, results, and any existing paper draft
- If a **description**, use it as the starting point

Then conduct a focused interview using `AskUserQuestion`. Ask **3-5 questions**:

1. **What is the core contribution?** "What specifically is new here?"
2. **What problem does it solve?** "What limitation or gap does this address?"
3. **Who is the audience/venue?** "Conference paper, journal, technical report, or internal note?"
4. **What evidence supports the claims?** "Simulations, experiments, proofs, or analysis?"
5. **What are the limitations?** "What doesn't this approach handle?"

Adapt questions based on existing material. Skip obvious ones.

**Checkpoint:** Write initial `_progress.json` with `interview: done`.

### Phase 2: Literature Search

Search for related work across:

**Codebase (internal):**
- Search `paper/` for related threads and paper references
- Check `projects/*/paper/` for related papers
- Check existing `papers.json` manifests

**Academic literature (web search):**
- Core technique + recent publications
- Problem domain + established approaches
- Survey papers in the field
- Key conference proceedings (venue-appropriate)
- Key research groups working on similar problems

For each relevant reference, document:
- Full citation
- What it contributes
- How our work relates (extends, complements, contradicts, is orthogonal to)

**Output:** `literature.md` structured as:

```markdown
# Literature Review — {Title}

## Positioning
{2-3 paragraphs: where this work sits in the landscape, what gap it fills}

## Key Related Work
### {Theme 1}
- {Reference}: {what it does, how we relate}
...

### {Theme 2}
...

## Gap Analysis
{What the existing literature does NOT address that our paper covers}

## Search Methodology
{What was searched, where, what terms}
```

**Checkpoint:** `literature_search: done`.

### Phase 3: Write the Paper

Write `paper.tex` using the LaTeX template above. Adapt the section structure to the content (not every paper needs all standard sections).

**Drafting guidelines:**
- **Lead with the contribution.** The abstract and introduction should make the contribution crystal clear in the first few paragraphs
- **Be honest about limitations.** Discuss what the approach does NOT handle
- **Include quantitative results.** Tables, figures, concrete numbers
- **Reference figures.** Use `Figure~\ref{fig:N}` and include figure environments. Use `% TODO: figure needed — {description}` for placeholders
- **Cite properly.** Use `\cite{key}` with matching `\bibitem{key}` entries. Verify every citation is real
- **Write clearly.** Short sentences, active voice, one idea per paragraph

If the thread already has generated figures (e.g., from previous work), reference them with `\includegraphics`. Copy or symlink the figures directory.

**Checkpoint:** `paper_tex: done`.

### Phase 4: Compile and Present

```bash
cd paper/{thread}.1/
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex  # twice for refs
rm -f *.aux *.log *.out *.toc
```

Present the summary:

```
## Draft: {thread}.1

**Title:** {title}
**Pages:** {N}
**Sections:** {list}
**Figures:** {count} ({count with actual content} / {count TODO})
**References:** {count}

**Core contribution:** {1-2 sentences}

**Next step:** Run `/pub-review {thread}.1` for a structured review,
then `/pub-revise {thread}.1` to incorporate feedback.
```

**Checkpoint:** `compile_pdfs: done`.
