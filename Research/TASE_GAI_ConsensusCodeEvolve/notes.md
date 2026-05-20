# Notes For Revision

## Suitability Judgment

The topic is suitable for the T-ASE special issue if positioned as an industrial automation method for discrete manufacturing/supply operations. The strongest framing is not "LLM writes code" alone, but:

> lightweight generative industrial task agents for consensus-aware material-flow decision-making.

The method should emphasize simulator grounding, physical constraints, safe generated code, interpretable deployment, and collaborative control.

## What Must Be Added Before Submission

- Full experiments over multiple seeds and scenarios.
- Comparison with `MTGP_niching_replenish_transship`, hand-designed policies, and ideally one conventional inventory baseline.
- Ablation on consensus gate components.
- Ablation on LLM token budget and number of generated candidates.
- A clear industrial case narrative, such as multi-plant spare-parts inventory, production-line material replenishment, or multi-warehouse support for discrete manufacturing.
- Stronger discussion of automation deployment: generated policy is executed locally without the LLM.

## Avoid Overclaiming

Do not claim the method outperforms baselines until experiments are complete. Current text intentionally says "planned", "hypothesis", and "prototype".
