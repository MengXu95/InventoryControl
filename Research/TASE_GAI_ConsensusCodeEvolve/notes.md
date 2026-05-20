# Notes For Revision

## Research Framing

The topic should be positioned as an inventory-control and industrial-automation method, not as a general LLM-for-code study. The strongest framing is:

> lightweight generative policy-code search for consensus-aware material-flow decision-making.

The method should emphasize simulator grounding, physical constraints, safe generated code, interpretable deployment, and collaborative control.

## What Must Be Added Before Final Manuscript

- Full experiments over multiple seeds and scenarios.
- Comparison with `MTGP_niching_replenish_transship`, hand-designed policies, and ideally one conventional inventory baseline.
- Ablation on consensus gate components.
- Ablation on LLM token budget and number of generated candidates.
- A clear industrial case narrative, such as multi-plant spare-parts inventory, production-line material replenishment, or multi-warehouse support for discrete manufacturing.
- Stronger discussion of automation deployment: generated policy is executed locally without the LLM.

## Avoid Overclaiming

Do not claim the method outperforms baselines until experiments are complete. Current text intentionally says "planned", "hypothesis", and "prototype".

## Method Name Review

Do not rename `ConsensusCode-Evolve` in the manuscript until the final method name is approved.

`AI4Tech` is not recommended as the method name for this paper. A Crossref check found existing 2024 uses of `AI4Tech`, including "AI4Tech: X-AI Enabling X-Tech...". The name is also too broad for this paper: it sounds like a general project, platform, lab, or company rather than a specific method for simulator-grounded multi-site inventory policy search.

Candidate shorter names:

- `ConCode`: Consensus Code Search. Fits the consensus/code idea; possible weakness is that it does not mention inventory.
- `InvCode`: Inventory Code Search. Short and domain-specific; possible weakness is that consensus is not visible.
- `CoDE-Inv`: Consensus-Driven Evolution for Inventory. Connects consensus, evolution, and inventory; possible weakness is mixed capitalization.
- `Cosearch`: Consensus Search. Easy to remember; possible weakness is broad outside inventory.
- `ConEvo`: Consensus Evolution. Short and close to current name; possible weakness is that code generation is implicit.
- `TransCode`: Transshipment Code Search. Strongly tied to the transfer decision; possible weakness is less clear about replenishment.
- `InvEvolve`: Inventory Policy Evolution. Clear and broad enough for the method; possible weakness is less specific about consensus.
- `CodeStock`: Code Search for Stock Control. Memorable; possible weakness is informal.
- `MAICode`: Multi-Agent Inventory Code Search. Specific to the paper; possible weakness is acronym pronunciation.
- `ConStock`: Consensus-Aware Stock Control. Clear link to consensus and inventory; possible weakness is less explicit about code evolution.
