# ConsensusCode-Evolve Paper Draft

This folder contains a LaTeX draft for a research article on simulator-grounded generative code search for consensus-aware multi-site inventory automation.

## Paper Framing

The manuscript should be framed as a normal research article. The motivation comes from the inventory-control problem: multi-site replenishment and transshipment require policies that are cost-aware, feasible, interpretable, and sensitive to donor risk and receiver urgency.

Key points to keep:

- The method studies multi-site replenishment and transshipment decisions for inventory automation.
- Each site is treated as a local inventory agent, and transshipment is filtered by a consensus-seeking gate.
- An LLM generates candidate policy code offline, while the simulator evaluates cost, service, transshipment, and fairness measures.
- The final policies are constrained Python functions, not online LLM controllers, so deployment can be lightweight and inspectable.
- Generated policies are accepted only after validation and simulator evaluation under inventory dynamics, capacity limits, lead times, and transshipment feasibility.

Risk to manage:

- The paper should not look like pure prompt engineering. The algorithmic contribution must be the closed-loop code-evolution workflow, safety validator, consensus gate, archive selection, and simulator-grounded evaluation.
- Claims must be based on experiments. The current implementation is a prototype; the manuscript marks missing experimental details with `[AUTHOR CHECK: ...]`.

## Suggested Title

Primary title:

**ConsensusCode-Evolve: Token-Efficient Generative Code Search for Consensus-Aware Multi-Agent Inventory Automation**

Alternative titles:

1. **Simulator-Grounded Generative Code Evolution for Consensus-Aware Multi-Site Inventory Control**
2. **Generative Policy-Code Search for Consensus-Aware Replenishment and Transshipment Automation**
3. **Token-Efficient LLM Policy Evolution for Collaborative Multi-Site Inventory Automation**

## Build

From this folder:

```powershell
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The draft uses `IEEEtran.cls`. If the class is missing, install the IEEEtran LaTeX package through TeX Live Manager.
