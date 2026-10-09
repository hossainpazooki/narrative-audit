# reference/

The artefacts that produced the figures in the published paper. `verify.py`
compares what the pipeline builds against these.

`per_paper/` holds the metric files as computed for the paper. `pgfplots/`
holds the plot-ready series.

The `readability_*.csv` files here carry only `year` and `neurips`, the two
columns the paper's figures read, and only the 39 years NeurIPS covers
(1987-2025). Upstream those files spanned 1987-2027 with four further corpora,
because the year index was the union across five corpora and ICLR 2026 and
PubMed ahead-of-print records extend past 2025. Those rows and columns are
empty for NeurIPS and are dropped rather than left here to imply this release
covers them. No value was altered.

Nothing in this directory is an input to the pipeline. To work from the shipped
metrics instead of scraping, copy `reference/per_paper/` into `data/per_paper/`
and run `python main.py --from aggregate`.
