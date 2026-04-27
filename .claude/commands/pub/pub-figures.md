---
name: "pub-figures"
description: "Identify and generate missing figures for a research paper"
domain: pub
type: command
---

# Generate Paper Figures

Identify missing or placeholder figures in a research paper and batch-generate Python scripts to create them.

## Invocation

```
/pub-figures {thread.N}
```

**Arguments**: `$ARGUMENTS`

## Locate the Paper

Same resolution as `/pub-review`: `paper/{thread}.{N}/`

## Workflow

### Step 1: Identify Missing Figures

Read `paper.tex` and find:
1. `% TODO: figure` or `% TODO: Figure` comments
2. `\ref{fig:...}` references without corresponding `\begin{figure}` environments
3. `\begin{figure}` environments with placeholder content
4. Concepts in the text that would clearly benefit from a figure

Report:
```
Found {N} figures to generate:
1. fig_{name} — {description} ({TODO comment / missing ref / suggested})
...
```

### Step 2: Generate Figure Scripts

For each figure, create a Python script in `{thread}.{N}/figures/`:

**For block/system diagrams:** Use `patent_figures.py`:
```python
import sys; sys.path.insert(0, '/path/to/docs/templates')
from patent_figures import PatentFigure
fig = PatentFigure("Title", palette="color")
# ... blocks, arrows, labels
fig.save("figN_name.pdf")
```

**For data/result plots:** Use raw matplotlib:
```python
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
})

fig, ax = plt.subplots(figsize=(4.5, 3))
# ... plot data
fig.savefig('figN_name.pdf')
fig.savefig('figN_name.png')
```

**Style guidelines (papers, not patents):**
- Always use color (no B&W requirement)
- Use colorblind-friendly palettes
- No reference numerals (patent convention — not needed for papers)
- Clear axis labels with units
- Legends inside or outside the plot, not in captions
- Serif font to match LaTeX body text
- High DPI (300+) for crisp rendering

### Step 3: Run and Verify

Execute each script and verify the output:
```bash
cd paper/{thread}.{N}/figures/
python3 figN_name.py
```

Check that both `.pdf` and `.png` are produced. View the `.png` to verify correctness.

### Step 4: Update Paper

For each generated figure, update `paper.tex`:
- Replace `% TODO` comments with `\includegraphics` environments
- Add `\caption` and `\label` matching the text references
- Recompile the PDF

Report:
```
Generated {N} figures:
- figures/fig1_name.{pdf,png} — {description}
...

Updated paper.tex with figure environments. Recompile with:
  cd paper/{thread}.{N}/
  pdflatex paper.tex && pdflatex paper.tex
```

## Checkpointing

Supports `_progress.json`. Phase `analysis` first, then per-figure: `fig_N_script`, `fig_N_run`, `fig_N_verify`.
