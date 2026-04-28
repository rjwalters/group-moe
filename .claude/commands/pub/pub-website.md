---
name: "pub-website"
description: "Create a blog post and research paper entry on rjwalters.info for a completed research paper"
domain: pub
type: command
---

# Publish to Website

Create a blog post and research paper entry on rjwalters.info for a completed research paper. This is the final step in the publication workflow — run after `/pub-audit` passes.

## Invocation

```
/pub-website <thread.N>
```

**Arguments**: `$ARGUMENTS`

## Prerequisites

- The paper must have a clean audit (0 critical issues)
- The paper LaTeX source must exist at `paper/{thread}.{N}/paper.tex`
- The rjwalters.info repo must be at `../rjwalters.info`

## Website Structure

The website at `../rjwalters.info` is a React/Vite/TypeScript site deployed to Cloudflare Pages.

### Blog Posts

- **Location**: `src/blog/posts/YYYY-MM-DD-slug.tsx`
- **Format**: TSX component, single file
- **Registry**: `src/blog/posts/posts.ts` (metadata array)
- **Loader**: `src/blog/posts/index.ts` (dynamic import map)
- **Style**: Follow STYLE_GUIDE.md — conversational, specific, story-driven
- **Links**: Internal via `<Link to="/research/...">`, external via `<a href="..." target="_blank">`

### Research Papers

- **Location**: `src/research/papers/{paper-id}/` (directory with section components)
- **Format**: TSX components per section (index.tsx, Introduction.tsx, Methods.tsx, etc.)
- **Registry**: `src/research/papers/papers.ts` (metadata with ResearchPaper interface)
- **Loader**: `src/research/papers/index.ts` (dynamic import map)
- **Math**: Use `react-katex` (`InlineMath`, `BlockMath`) — import `katex/dist/katex.min.css`
- **Figures**: Copy to `public/research/{paper-id}/`, reference in components

### ResearchPaper Interface

```typescript
{
  id: string,           // e.g., "2026-group-moe"
  title: string,
  authors: string[],
  venue: string,        // e.g., "Preprint"
  year: number,
  date: string,         // ISO format
  abstract: string,
  tags?: string[],
  links?: {
    pdf?: string,
    arxiv?: string,
    github?: string,
  }
}
```

## Workflow

### Step 1: Read the Paper

Read `paper/{thread}.{N}/paper.tex` and `paper/{thread}.{N}/literature.md` to understand the content.

### Step 2: Create Blog Post

Write a blog post TSX file following the pattern of existing posts (see `2026-04-24-ordering-is-not-invariant.tsx`). The blog post should:

- **Open with a question or hook** — not "We present a paper about..."
- **Explain the key idea in plain language** — what problem, what insight, why it matters
- **Include 1-2 concrete results** — the most compelling findings
- **Be honest about limitations** — don't oversell
- **Link to the full paper** on the site and code on GitHub
- **Follow STYLE_GUIDE.md** — confident voice, specific details, natural rhythm
- **Length**: 500-800 words, 4-6 sections

### Step 3: Create Research Paper Entry

Convert the LaTeX paper to TSX section components. Follow the pattern of `2026-latent-symmetries/`:

1. Create directory `src/research/papers/{paper-id}/`
2. Create section components: `index.tsx`, `TableOfContents.tsx`, `Introduction.tsx`, etc.
3. Convert LaTeX math to `react-katex` components
4. Convert tables to HTML `<table>` with Tailwind classes
5. Convert `\cite{key}` to numbered superscript links
6. Copy figures to `public/research/{paper-id}/`

### Step 4: Update Registries

Add entries to:
- `src/blog/posts/posts.ts` — blog post metadata
- `src/blog/posts/index.ts` — dynamic import
- `src/research/papers/papers.ts` — paper metadata
- `src/research/papers/index.ts` — dynamic import

### Step 5: Build and Verify

```bash
cd ../rjwalters.info
pnpm dev  # Start dev server
# Visit http://localhost:5173/blog/{post-id}
# Visit http://localhost:5173/research/{paper-id}
```

Verify:
- Blog post renders, links work, style matches existing posts
- Research paper renders, math displays, figures load, references link
- No TypeScript or build errors

### Step 6: Commit and Deploy

```bash
cd ../rjwalters.info
git add src/blog/posts/ src/research/papers/ public/research/
git commit -m "Add {paper title} blog post and research paper"
pnpm publish  # Build + deploy to Cloudflare Pages
```

## Style Notes

- Blog posts are **conversational** — a friend explaining their research over coffee
- Research papers on the site are **complete but web-native** — not a PDF dump, but structured components with web typography
- Both follow STYLE_GUIDE.md for voice and tone
- Use semantic HTML: `<section>`, `<h2>`, `<p>`, `<em>`, `<strong>`
- Links: blue-400 hover:blue-300 for both internal and external
