# T-ASE Special Issue Draft: ConsensusCode-Evolve

This folder contains a LaTeX draft for the IEEE Transactions on Automation Science and Engineering special issue on **Generative AI Theories, Methods and Applications for Industrial Automation**.

## Fit To The Special Issue

The idea is suitable if framed as an automation-oriented GAI method rather than a general LLM-for-code paper.

Strong fit:

- **GAI-driven plant-wide operation, dynamic production scheduling and intelligent decision-making**: the work studies multi-site replenishment and transshipment decisions for inventory automation.
- **Industrial task agents and cooperative control**: each site is treated as an autonomous inventory agent, and transshipment is filtered by a consensus-seeking protocol.
- **GAI-enabled multi-objective optimization**: an LLM generates candidate policy code, while the simulator evaluates cost, service, transshipment, and fairness measures.
- **Lightweight deployment and interpretability**: the final policies are constrained Python functions, not online LLM controllers, so deployment can be lightweight and inspectable.
- **Integration with physical mechanisms**: generated policies are accepted only after simulator-based validation under inventory dynamics, capacity limits, lead times, and transshipment feasibility.

Risk to manage:

- The special issue emphasizes industrial automation and manufacturing. The paper should explicitly connect multi-site inventory control to discrete manufacturing supply operations, production-line replenishment, spare-parts support, or plant-wide material flow.
- The paper should not look like pure prompt engineering. The algorithmic contribution must be the closed-loop code-evolution framework, safety validator, consensus gate, archive selection, and simulator-grounded evaluation.
- Claims must be based on experiments. The current implementation is a prototype; the draft marks performance claims as hypotheses until full experiments are run.

## Suggested Title

Primary title:

**ConsensusCode-Evolve: Token-Efficient Generative Code Search for Consensus-Aware Multi-Agent Inventory Automation**

Alternative titles:

1. **Simulator-Grounded Generative Code Evolution for Consensus-Aware Multi-Site Inventory Control**
2. **Generative Industrial Task Agents for Consensus-Aware Replenishment and Transshipment Automation**
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
