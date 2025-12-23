# README Figures

Place paper figures and derived charts used by the root README in this directory.

Expected files:

- `riscore-method-overview.svg`: high-level RISCORE prompting overview (paper Figure 2).
- `context-reconstruction-pipeline.svg`: context reconstruction / augmentation pipeline (paper Figure 3).
- `fs-vs-riscore-comparison.svg`: side-by-side comparison of standard FS vs RISCORE on the same riddle (paper Figure 1). **User-supplied** — export from the paper or redraw and place here. If exported from draw.io, re-export as a plain SVG (File → Export As → SVG → uncheck "Include a copy of my diagram") to avoid embedding the full diagram XML (~600 KB overhead).
- `riscore-vs-fs-sim-4shot.svg`: grouped bar chart comparing FS Sim vs RISCOREm at 4-shot across five models. **Generated** from Table 1 (semantically-similar block), Panagiotopoulos et al. COLING 2025 — this is derived data, not a paper figure.

Keep image files reasonably compressed for GitHub rendering. If figures are
exported from the paper PDF, preserve the original caption meaning and cite the
paper in the root README.
