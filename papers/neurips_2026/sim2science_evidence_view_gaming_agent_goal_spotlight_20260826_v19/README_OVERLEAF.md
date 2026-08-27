# Sim2Science evidence-view gaming — v19

Compile with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The submission body is five pages; references begin on page 6. All figures are
editable TikZ/PGFPlots inputs under `figures/`. The Tabler robot and lock icons
are bundled with their MIT license and source note.

Run the manuscript claim verifier from this directory:

```bash
python3 verify_manuscript_claims.py
```

The verifier reads large immutable experiment artifacts from the surrounding
`verisigs` repository rather than duplicating them in the Overleaf payload.
The two compact analyses added during the final result pass are under
`repro_analysis/`.

The defense-aware enumeration is explicitly retrospective and truth-aware. It
exhausts the frozen 12-view bank, not unseen numerical representations, and is
not presented as an executable attack or a security guarantee.
