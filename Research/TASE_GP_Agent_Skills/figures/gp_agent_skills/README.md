# Editable GP / Agent / Skill figures

Paper: **PACT-GP: Genetic Programming for Skill-Based Agent Coordination in
Multi-Site Inventory Control**. PACT-GP means Peer-Agent Coordination with
Three-tree Genetic Programming. This title is also embedded in the PPTX and all
six PDF metadata records.

`gp_agent_skills_figures.pptx` is the master deliverable. All rectangles, tree
nodes, text and connectors are native editable PowerPoint objects. There are no
embedded figure screenshots, SVG pictures, or external assets.

## Figures

| Slide | Standalone vector export | Content |
| --- | --- | --- |
| 1 | `architecture.pdf` | Offline GP individual → identical local policy copies → deterministic Agent Skills → peer messages |
| 2 | `negotiation.pdf` | Three-site, two-round competition; conditional reservations of 8 and 2 |
| 3 | `gp_search.pdf` | PACT-GP initialization, study budget of at least 200 individuals / 50 generations, seeded evaluation, constrained selection, genetic variation, validation and frozen test |
| 4 | `gp_individual.pdf` | Three illustrative valid expressions, not initialized seed or trained optimum |
| 5 | `timing.pdf` | Pre-demand negotiation versus full / partial / cancelled post-demand execution |
| 6 | `fixed_evaluator.pdf` | Forecast-slack gain proxy, candidate quantity grid and illustrative local message schema |

Orange denotes offline GP; blue denotes local Agents / Skills; green denotes the
fixed negotiation protocol; purple denotes actual physical execution. The deck
uses English and Arial, a 16:9 canvas, and no LLM in training or deployment.
Text and geometry can be edited individually in PowerPoint's Selection Pane.
Shape names identify their slide and, for GP nodes, their tree and symbol.

The request and offer roles are selected by the fixed layer from forecast slack.
The three pictured trees are illustrative valid expressions, not initialized
seed or trained optimum. Actual initialization in the implementation is
`(maximum(0, F-I-PIPE), DEFICIT, SURPLUS)`. F1 and F2 are exposed
forecasts; I is local on-hand inventory; P is the local pipeline total. Donor and
receiver marginal values use their own h and b coefficients. The evaluator is a
heuristic proxy, not an expected-cost guarantee. The message schema is illustrative,
not a claim about a particular code serialization. Candidate quantities scale the
remaining minimum of offer and request by {0, 0.25, 0.5, 0.75, 1}.
Conservative request and offer bounds do not exceed forecast deficit and
forecast surplus, respectively. Thus the proxy stays on its linear branches:
for positive quantity, `G(q) = (h_donor + b_receiver - c) q - K`.
A profitable pair prefers the largest feasible quantity; the grid does not
provide an interior or nonlinear risk-optimal quantity search. All scoring and
protocol rules remain fixed. GP fitness evaluates site service constraints
first, rather than a network-aggregate service constraint, and then ranks
admissible policies by cost.

In the negotiation example, initially D offers 10, A requests 6, and B requests 8.
This is explicitly an illustrative **K = 0** example, not the simulator's
default **K = 10** setting. Its linear gain coefficient is 1.5 per unit, so
`G(q) = 1.5q` for positive quantity.
Round 1 reserves D–B = 8 (gain 12); A's candidate q = 6 has gain 9 and is not
mutual best. Round 2 reserves D–A = 2 (gain 3). D reserves 10 in total without
overpromise. These gains are illustrative, internally consistent candidate scores,
not fitted experimental results; promises are distinct from realized flow.

## Regenerate and validate

The listed dependencies are already available in the supplied Python 3.11
interpreter; no installation was needed. PowerPoint COM 16.0 is available locally.
From this folder:

```powershell
& 'C:\Users\xumen\AppData\Local\Programs\Python\Python311\python.exe' .\generate_figures.py --preview
& 'C:\Users\xumen\AppData\Local\Programs\Python\Python311\python.exe' .\generate_figures.py --check
```

`generate_figures.py` constructs the native deck with `python-pptx` and invokes
`export_powerpoint.ps1`. PowerPoint opens a read-only, untitled copy of the master
for each single-slide PDF export; the master PPTX is never modified by the exporter.
Only these task-owned copies are closed. Because PowerPoint may share an
existing application server even when COM activation requests a new application,
the exporter releases COM references without calling `Application.Quit`; it
never closes pre-existing user presentations or terminates PowerPoint processes.
PDFs use native
PowerPoint vector/text output, not a separately redrawn approximation. The
optional `*_preview.png` files are inspection renders only, never the editable
deliverable or LaTeX figures.

`--check` validates model bounds, font-metric text fit, expected labels, the exact
native GP-node counts, PPTX XML native shape / connector / text counts, absence
of PPTX media and pictures, one-page PDF dimensions, vector paths, absence of
PDF raster images, complete visible text from every textbox, text bounds and
inter-textbox overlap.
The console reports per-slide counts. `--no-export` can generate / check only
the PPTX on a machine without PowerPoint; existing PDFs are not regenerated in
that mode. The default full generation requires Microsoft PowerPoint on Windows.

## Verified generated output

The six-slide master contains **234 native editable objects**: 156 shapes /
textboxes and 78 connectors, with 209 native text runs. Its ZIP/XML contains
zero picture objects and no media files. The GP examples have 25 native tree
nodes (9 order, 9 request, 7 offer).

| Figure | Shapes / textboxes | Connectors | PDF vector paths | PDF raster images |
| --- | ---: | ---: | ---: | ---: |
| architecture | 31 | 11 | 28 | 0 |
| negotiation | 23 | 6 | 19 | 0 |
| gp_search | 17 | 14 | 26 | 0 |
| gp_individual | 41 | 23 | 56 | 0 |
| timing | 25 | 23 | 36 | 0 |
| fixed_evaluator | 19 | 1 | 12 | 0 |

All six vector PDFs were exported successfully by local Microsoft PowerPoint
16.0, rendered for visual inspection, and passed text completeness, bounds and
overlap checks. No render/export fallback was needed. PowerPoint theme effects
are explicitly disabled to prevent raster shadow layers. Unicode bidirectional
arrow glyphs that PowerPoint treats as emoji are avoided; peer arrows are native
connectors instead. Re-exporting on other Office versions or without Arial may
change font metrics; rerun `--check` after regenerating.

## Two-column article readability

All visible text is at least **18 pt in the native 16-inch-wide slide/PDF**.
At a **7.125-inch two-column article width**, the minimum printed text is
**8.016 pt**. This is checked from the actual exported PDF font sizes, not only
the source model. Tiny slide footer text has been removed, labels have been
shortened where necessary, and textboxes accommodate native Arial leading.
All six updated figures were rendered and visually inspected.
The three `max` root circles are wider than the terminal / arithmetic nodes;
node text wrapping is explicitly disabled. Validation checks every GP-node
label renders as one complete PDF text span inside its native node.

Use a full-width `figure*` and `\includegraphics[width=\textwidth]{...}`.
Do not place these dense diagrams in a single article column. At a different
width, the minimum printed text is `18 × width_in_inches / 16` pt; at 7 inches
it is 7.875 pt. The generator's `PAPER_WIDTH` constant controls the readability
check. The sixth export, `fixed_evaluator.pdf`, is a required deliverable.

For LaTeX, use the six named `.pdf` exports directly with `\includegraphics`.
They preserve the full consistent slide canvas without screenshot rasterization.
