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
