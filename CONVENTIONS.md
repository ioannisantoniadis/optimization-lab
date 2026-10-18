# Authoring conventions for optimization-lab

This file is the single source of truth for how the chapters in `docs/` are written, so
that chapters written at different times read as one book. Read
`docs/appendix-notation.qmd` first (it defines every symbol), then this file. It is
adapted from the conventions of the sibling books (`rl-for-llms`, `loss-functions-lab`,
`data-lab`) and from the Part II rules of `objectives-book`'s SPEC (§9, §12, §21.1), which
this book is meant to feed.

Not part of the rendered book.

## What this book is, in one paragraph

An introduction to how optimization algorithms work and why. The reader should finish
each section able to state the method's update rule, say which property of the objective
it exploits (slope, curvature, convexity, structure, or none), predict when it will fail,
and recognize it inside a larger system. The `optimlab` package exists so that every
figure and every number in the book is computed by a real implementation, but **the book
is about optimization, not about the package.** Code is evidence, folded away by default.

## Standing rules

1. **Explain the mechanism, not the API.** Every method gets its idea in plain words,
   its update rule as math, the derivation step that makes it work, its settings, and its
   failure modes. A reader who never opens the code must still understand the method.
2. **Name algorithms, not functions.** Prose says "BFGS", "particle swarm", "the
   Lanczos algorithm", never `bfgs`, `optimlab.optimizers.particle_swarm` or
   `lanczos_eigenvalues`. The package is named in exactly two places: the
   implementation line at the end of each method section (below), and
   `docs/appendix-the-code.qmd`.
3. **No project-history references.** No roadmap phases, no "this repo", no "this
   phase", no test file paths in prose, no bug stories told as bug stories. A past
   failure is worth telling only as a lesson about the method (for example, why a
   primal-residual-only stopping rule is unsafe for ADMM), in the method's own terms.
4. **Every number is computed.** Numbers quoted in prose come from inline expressions
   (`` `{python} f"{value:.3g}"` ``) or from tables emitted by a code cell, never typed
   by hand. Bare `print` output is avoided: use an inline expression or a Markdown table
   (`optimlab.viz.markdown_table`). Never put an inline expression inside `$...$`:
   Quarto escapes its output as Markdown text, which breaks the LaTeX. Write
   `$a =$ `` `{python} a` `` instead.
5. **Every figure answers a question the text asks.** Before adding one, finish the
   sentence "this figure makes visible that ...". Captions state the lesson, not just
   what is plotted.

## Chapter file and first line

`docs/chapters/NN-slug.qmd`, no YAML front matter. The first line is the chapter heading
with its Quarto ID, which is what cross-references target:

```markdown
# Gradient Methods {#sec-gradient-methods}
```

The map is `{#sec-map .unnumbered}`, so chapter numbers run 1–9.

## The chapter template

1. **Opening, no heading.** The problem in plain language, and which limitation of the
   previous chapter motivates this one.
2. **`## The problem`**: what is being minimized, under which assumptions, and what
   "solving it" means here (an exact answer, a rate, a guarantee, or a good guess).
3. **One `##` section per method**, in this order inside the section (use `###`
   subheadings only when the section is long):
   - the idea, in plain words, before any notation;
   - the **definition box**: the update rule or optimality condition, exactly;
     ```markdown
     ::: {.callout-note title="Definition — <Name>"}
     $$ ... $$
     :::
     ```
     Multi-step algorithms (simplex pivot, Nelder–Mead, a genetic-algorithm
     generation, EM, ADMM) use a numbered list of steps inside an
     `::: {.callout-note title="Algorithm — <Name>"}` box instead;
   - **why it works**: the derivation step that the method rests on (Taylor expansion,
     smoothness bound, secant condition, conjugacy, Jensen's inequality, ...). Never
     skip a *conceptual* step;
   - **settings that matter**: each hyperparameter and what it trades off;
   - **behavior and failure modes**, with a computed figure where one helps;
   - the **implementation line**, last:
     ```markdown
     [Implementation: [`optimizers/quasi_newton.py`]({{< var src >}}/optimizers/quasi_newton.py) ·
     Checked by: [`test_optimizers.py`]({{< var tests >}}/test_optimizers.py)]{.impl}
     ```
4. **`## What these examples cannot show`**: what the low-dimensional, cheap, noiseless
   examples do not settle (scale, noise, wall-clock, generalization).
5. **`## Connections`**: 2–4 other chapters, each with *why* it is related.

Theorems use `::: {.callout-tip title="Theorem — <Name>"}`. A modeling caveat uses
`callout-important`. Collapsed callouts never hold essential content.

Length: roughly 2,000–4,000 words of prose per chapter (equations, code and figures
excluded). Economy still applies: if a paragraph could be deleted without loss, delete
it.

## Evidence labels

Every empirical claim is one of three kinds, and the prose says which:

- **Mathematical fact**: derived here or proved in a cited source. Stated plainly.
- **Replicated finding**: reported by several independent groups. "Several groups
  report ...", with citations.
- **Recent or contested**: one paper or an open debate. "The paper reports ...", with
  the counter-evidence (e.g. sharp vs. flat minima).

A toy result is never stated as a general result.

## Math, code and citation mechanics

- Inline `$...$`, display `$$...$$`, standard LaTeX only. Use only the symbols in
  `docs/appendix-notation.qmd`; a new symbol goes there in the same change. Math is never
  written inside code backticks.
- **Code cells execute at render time** (unlike the sibling books, which use static
  figure scripts) and are folded by default (`code-fold: true`). Keep cells short and
  readable; lines under 100 characters. Use readable series names in figures
  (`"Gradient descent"`, not `"gradient_descent"`).
- Cross-references by Quarto ID: `@sec-gradient-methods`, `@fig-ill-conditioned`. Never
  type a chapter number by hand.
- Citations: `[@key]` against `docs/references.bib`. Never fabricate an entry.

## Figures

- Built with the `optimlab.viz` figure builders, which share `optimlab/viz/theme.py`
  (see CLAUDE.md for the palette rules). A one-off figure in a chapter still goes
  through `layout_template` and the theme's colors.
- Give every figure a `#| label: fig-...` and a caption that states the lesson.
- Render and look at every new or changed figure before committing.

## Rendering and the freeze

`freeze: auto` re-executes a chapter only when its `.qmd` changes. After changing
`src/optimlab/`, re-render the affected chapters. `scripts/check_freeze.py` compares a
fresh render's computed text (printed output, tables and inline numbers) against the
committed `docs/_freeze/`. Re-render once per finished chapter, not on every edit: each
re-freeze adds the Plotly payloads to git history.
