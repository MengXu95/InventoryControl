# Three-Tree GP / Agent / Skill Methods Draft

This is an independent research branch of the earlier ConsensusCode-Evolve
draft. The active manuscript studies **three-tree GP with fixed peer
negotiation**, not LLM policy generation.

## Files

- `main.tex`: active article, problem definition, timing, evaluation protocol,
  and explicit limitations.
- `method.tex`: GP / Agent / Skill relationship, three-tree representation,
  local information boundary, fixed gain, mutual matching, reservations,
  conditional dispatch, and offline GP search.
- `references.bib`: bibliography copied from the earlier manuscript.
- `figures/gp_agent_skills/`: editable PowerPoint deck, figure generator, and
  vector exports used by the active manuscript.
- `provenance/llm_source_snapshot.tex` and `provenance/llm_source_notes.md`:
  inherited source material, retained only to document the copy. They are not
  the active manuscript and should not be compiled as the GP version.
- `..\..\MTGP_Agent_Skills\`: independent executable implementation.

The original `Research\TASE_GAI_ConsensusCodeEvolve` and
`MTGP_niching_replenish_transship` folders are not edited by this work.

## Algorithm contract

1. Each site is an agent with local observation, neighbor messages, a
   reservation ledger, and actual-execution memory.
2. A single shared GP individual has exactly three expressions: order,
   request, and offer. The site state, not its identity, conditions decisions.
3. Skills provide the interface and fixed output checks around these
   expressions; GP is the offline search procedure, not an online supervisor.
4. The fixed protocol uses forecast-slack gain, finite quantity candidates,
   mutual partner choices, and reservations. There is no central matching
   optimizer and no evolved fourth gate/value tree.
5. Orders and conditional promises are chosen before current demand.
   Actual transfers occur after due arrivals and demand and are rechecked
   against donor stock and receiver capacity.
6. Planned, executed, partial, and cancelled commitments are not conflated.
   Only actual movements enter transfer history and physical costs.
7. Gain is a heuristic, service constraints are empirical, and no global
   optimality, privacy, consensus-convergence, or performance superiority is
   claimed.

The implemented conservative caps bound requests by forecast deficit and
current storage room, and offers by forecast surplus and pre-demand
availability including due arrivals. Under these caps the fixed gain is
linear in quantity. A positive eligible pair therefore selects the largest
grid quantity; the grid is **not** an interior risk-optimization algorithm.
GP controls the advertised quantities and hence the available matching
choices.

## Run the copied implementation

From the repository root, in the configured Python environment:

```powershell
python -m unittest discover -s MTGP_Agent_Skills\tests -v
python -m MTGP_Agent_Skills smoke --seed 0 --output-dir MTGP_Agent_Skills\outputs\my_smoke
python -m MTGP_Agent_Skills evaluate --policy MTGP_Agent_Skills\outputs\my_smoke\best_policy.json --config MTGP_Agent_Skills\outputs\my_smoke\config.json --seeds 2001,2002 --output-dir MTGP_Agent_Skills\outputs\my_replay
```

Output directories must be new or empty. `smoke` still uses the complete
64-period scenario, but only six individuals and one replacement
generation. Full `train` defaults to 20 individuals and five replacement
generations. There are exactly three expressions; the evaluator and
matching protocol are not additional evolved trees.

## Build the manuscript

Generate the figure exports first using the instructions in
`figures\gp_agent_skills\README.md`. Then, from this directory:

```powershell
.\build.ps1
```

The build script checks that all six diagram PDFs exist, stops on any failed
LaTeX/BibTeX command, and produces `main.pdf`. The equivalent manual sequence is:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The build uses the installed IEEEtran class and standard algorithm packages.

## Interpretation of results

Unit tests and a small GP smoke run establish implementation behavior, not a
publishable comparison. Full independent search repetitions, matched-budget
baselines, and industrial calibration remain future experiments. Future
actual demand copied into historical forecast slots must be identified as
oracle information, never silently reported as a causal forecast.
