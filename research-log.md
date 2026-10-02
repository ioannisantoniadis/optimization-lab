# Research log

Sources consulted and what was checked in each. `ROADMAP.md` records design decisions; this file
records source verification. Add an entry whenever a new source is used.

## 2026-10-02: audit and remediation pass

- **Filter normalization** (Ch 7, `highdim/loss_landscape.py`): the implementation rescaled whole
  layers, which Li et al. 2018 (arXiv 1712.09913, §4 and appendix) call *layer* normalization.
  Filter normalization rescales each filter, for a fully connected layer "the weights that generate
  one neuron". Re-implemented per neuron (a column of `W` here); bias directions zeroed, matching
  the default `ignore='biasbn'` in the authors' code (`tomgoldstein/loss-landscape`,
  `net_plotter.py`). Test updated; Chapter 7 re-executed.
- **Sharp minima** (Ch 7): the reparametrization objection is Dinh et al. 2017 (arXiv 1703.04933),
  now cited; Keskar et al. is the original claim.
- **Bibliography identifiers**: DOIs confirmed against Crossref (original publications, not
  reprints) and arXiv ids against abstract pages. Bellman (RAND report, 1957) and Kalman (1960)
  have no DOI. `reddi2018convergence` removed (never cited).
- Confirmed in the audit: Dauphin et al. 2014 (saddle points, Bray & Dean), Sagun et al. 2016
  (bulk + outliers), Jacot et al. 2018, Du et al. 2019 (metadata), and all six arXiv titles.
- **Docs execution**: `freeze: auto` re-executes a chapter only when its `.qmd` changes (Quarto
  docs, *Code execution*); README corrected.
