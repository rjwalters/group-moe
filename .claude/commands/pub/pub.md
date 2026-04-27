---
name: "pub"
description: "Assess state of all research publications and launch parallel review/revise/figure agents"
domain: pub
type: command
---

# Publication Portfolio Orchestrator

Assess the state of all research paper drafts and coordinate next actions.

## Invocation

```
/pub              # assess all threads
/pub {thread}     # assess a specific thread
```

**Arguments**: `$ARGUMENTS`

## Workflow

### Step 1: Discover Threads

Scan `paper/` for all threads with paper directories. Also read `README.md` for the thread listing.

### Step 2: Assess State

For each thread, determine its state:

| State | Condition |
|-------|-----------|
| **NO PAPER** | Thread exists but no `paper/` subdirectory |
| **WORKING DRAFT** | `paper/paper.tex` exists at thread root (unversioned) |
| **DRAFTED** | `{thread}.{N}/paper.tex` exists, no `{N}.review/` |
| **REVIEWED** | `{N}.review/review.md` exists, no `{N+1}/` |
| **REVISED** | `{N+1}/` exists, no `{N+1}.review/` |
| **READY** | Latest review score >= 32/40, 0 critical issues |
| **AUDITED** | `{N}.audit/audit.md` exists and clean |

For REVIEWED states, read the review score and critical issue count.

### Step 3: Present Assessment

```
## Publication Portfolio

| Thread | State | Version | Score | Next Action |
|--------|-------|---------|-------|-------------|
| {name} | {state} | {N} | {X}/40 | `/pub-{action} {thread}.{N}` |
...

### Recommended Actions
1. {thread}: {action} — {reason}
...
```

### Step 4: Execute (if requested)

If the user asks to proceed, launch the appropriate commands:
- DRAFTED → run `/pub-review`
- REVIEWED → run `/pub-revise`
- REVISED → run `/pub-review`
- READY → run `/pub-audit`
- WORKING DRAFT → suggest `/pub-draft` to create versioned structure

Multiple independent reviews can run in parallel (via Agent tool).
