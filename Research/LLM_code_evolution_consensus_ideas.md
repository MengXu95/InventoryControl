# LLM Code Evolution Ideas for Autonomous Multi-Agent Consensus-Seeking

## Core Angle

Use an LLM not as an expensive online inventory controller, but as an **offline policy-code evolution engine**. The LLM proposes small Python policy functions, the simulator evaluates them cheaply, and a consensus layer forces transshipment decisions to satisfy both local site interests and global inventory performance.

This gives a research direction that combines:

- LLM-based program search,
- autonomous multi-agent consensus-seeking,
- interpretable Python inventory policies,
- simulator-grounded evaluation with low token usage.

## Short Literature Review: LLMs That Generate, Improve, Or Evolve Code

| Work | What It Shows | Useful Lesson For This Project |
| --- | --- | --- |
| **AlphaCode**: Li et al., *Competition-Level Code Generation with AlphaCode*, Science, 2022 | Large language models can generate many candidate programs and use filtering/clustering to select solutions for programming competitions. | Generate many small policy-code candidates, then select using the inventory simulator rather than trusting the LLM. |
| **CodeRL**: Le et al., *CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning*, NeurIPS, 2022 | Code generation improves when execution feedback and test results guide learning or reranking. | Treat simulation cost and feasibility checks as execution feedback for policy-code search. |
| **Self-Refine**: Madaan et al., *Self-Refine: Iterative Refinement with Self-Feedback*, NeurIPS, 2023 | LLM outputs can be iteratively improved using feedback without model fine-tuning. | Use compact simulator feedback to ask the LLM for targeted policy repairs. |
| **Reflexion**: Shinn et al., *Reflexion: Language Agents with Verbal Reinforcement Learning*, NeurIPS, 2023 | Agents can improve by storing verbal reflections over failed trials. | Store short reflections such as “policy over-transships from high holding-cost site during demand spikes.” |
| **Voyager**: Wang et al., *Voyager: An Open-Ended Embodied Agent with Large Language Models*, 2023 | An LLM can write executable skill code and accumulate a reusable skill library. | Build a library of reusable inventory-policy code motifs: reorder-up-to, risk gate, donor gate, fairness penalty. |
| **Eureka**: Ma et al., *Eureka: Human-Level Reward Design via Coding Large Language Models*, ICLR, 2024 | LLMs can write reward functions that are evaluated in simulation and iteratively improved. | Let the LLM evolve consensus penalties or robustness objectives, not only direct action policies. |
| **FunSearch**: Romera-Paredes et al., *Mathematical Discoveries from Program Search with Large Language Models*, Nature, 2024 | LLM-generated programs plus an automatic evaluator can discover high-performing algorithms. | The closest template: ask the LLM for Python policy code, score it in `InvOptEnv`, keep the best, and mutate from an archive. |
| **OPRO**: Yang et al., *Large Language Models as Optimizers*, 2023 | LLMs can propose improved solutions from a history of previous solutions and scores. | Feed the LLM only top policy summaries and scores, then ask for the next candidate. |
| **PromptBreeder**: Fernando et al., *PromptBreeder: Self-Referential Self-Improvement via Prompt Evolution*, 2023 | Evolutionary loops can improve prompts themselves. | Evolve not only policy code but also the prompt template used to generate policy code. |

Older genetic improvement work, such as GenProg and broader genetic programming literature, also matters because it gives a non-LLM foundation: mutate executable programs, evaluate with tests, keep better variants. The new opportunity is that LLMs can make larger semantic edits than random mutation while still being grounded by simulator scores.

## Proposed Fancy Idea: ConsensusCode-Evolve

### One-Sentence Idea

**ConsensusCode-Evolve** uses an LLM to evolve small Python replenishment and transshipment policy functions, while a multi-agent consensus protocol filters transshipment actions before simulator evaluation.

## Policy Representation

Instead of GP trees, ask the LLM to generate constrained Python functions:

```python
def replenishment_policy(site_state, global_state):
    return order_quantity

def transshipment_proposal(source_state, target_state, global_state):
    return proposed_quantity

def consensus_gate(source_state, target_state, proposed_quantity, history):
    return accepted_quantity
```

The code is still interpretable because it is normal Python with named variables. It can be inspected, simplified, and compared with GP trees.

## Consensus-Seeking Mechanism

Each site is treated as a local agent:

- A potential donor computes its **willingness to give**.
- A potential receiver computes its **urgency to receive**.
- The network computes a **global benefit score**.
- The consensus gate accepts, scales, or rejects the proposed transfer.

Example structure:

```text
accepted = proposed_quantity
           * donor_willingness
           * receiver_urgency
           * fairness_factor
```

This makes transshipment a negotiated action, not only a central command.

## Token-Efficient Evolution Loop

The key is to avoid using many tokens. The LLM should not see full trajectories or full logs.

```text
1. Start from 5-10 seed policies.
2. Run each policy in the simulator across small scenarios/seeds.
3. Summarize each policy in 5-8 numbers:
   total cost, lost-sales cost, order cost, transshipment cost,
   stockout count, rejected transfer count, fairness debt, recovery time.
4. Send only the top-k policy code snippets and compact score table to the LLM.
5. Ask for 3 mutations: conservative, consensus-focused, and resilience-focused.
6. Validate generated code with AST restrictions and simulator smoke tests.
7. Keep an archive of high-performing and behaviorally diverse policies.
```

The LLM is used only between batches. The simulator does the heavy work.

## Safety And Reproducibility Constraints

Generated code should be restricted to a tiny safe subset:

- no imports,
- no file access,
- no network access,
- no random calls inside policy functions,
- no loops except bounded simple loops if needed,
- only arithmetic, `min`, `max`, `abs`, and simple conditionals.

Before evaluation, parse with Python `ast` and reject unsafe nodes. This keeps the method publishable and reproducible.

## How It Connects To `MTGP_niching_replenish_transship`

The current GP method can remain the main baseline. The LLM method can be a new comparison family:

| Component | Current MTGP | ConsensusCode-Evolve |
| --- | --- | --- |
| Policy form | GP trees | constrained Python functions |
| Search operator | crossover/mutation | LLM semantic mutation + archive selection |
| Feedback | simulator cost | simulator cost + compact verbal reflection |
| Transshipment | direct pairwise quantity | proposal plus consensus gate |
| Interpretability | symbolic tree | readable code |

## Main Research Hypothesis

LLM-evolved Python policies can discover higher-level transshipment logic than GP trees under the same simulator budget, especially when consensus features such as donor willingness, receiver urgency, and fairness debt are explicitly represented.

This is an untested hypothesis, not a claim.

## Suggested Experiments

### Experiment 1: LLM Policy Code vs GP Tree

Compare:

- `MTGP_niching_replenish_transship`,
- hand-designed consensus policy,
- LLM-evolved Python policy without consensus gate,
- LLM-evolved Python policy with consensus gate.

### Experiment 2: Token Budget Ablation

Test whether the method works under small LLM budgets:

| Setting | LLM Calls | Candidates Per Call |
| --- | --- | --- |
| Tiny | 5 | 3 |
| Small | 10 | 5 |
| Medium | 20 | 5 |

Report cost improvement per LLM call, not only final cost.

### Experiment 3: Consensus Ablation

Compare:

- no consensus,
- donor willingness only,
- donor + receiver agreement,
- donor + receiver + fairness debt,
- donor + receiver + fairness + network benefit.

## Possible Paper Contribution

### Working Title

**Token-Efficient LLM Code Evolution for Consensus-Aware Multi-Site Inventory Control**

### Contributions To Test

1. A constrained Python policy representation for LLM-evolved replenishment and transshipment decisions.
2. A simulator-grounded LLM code-evolution loop that uses compact numeric feedback rather than long trajectory prompts.
3. A consensus-seeking transshipment protocol with donor willingness, receiver urgency, and fairness debt.
4. A comparison against symbolic GP policies on multi-site inventory scenarios.

## Minimal Implementation Plan

1. Create a new folder, for example `LLM_policy_evolution/`.
2. Define the safe policy API: `replenishment_policy`, `transshipment_proposal`, and `consensus_gate`.
3. Add an adapter that lets `InvOptEnv` evaluate Python policy functions instead of GP trees.
4. Add AST validation for generated code.
5. Store candidates and scores in `Research/llm_policy_runs/` or a new result folder.
6. Start with manual seed policies before calling an LLM.

## Why This Is Fancy But Still Practical

The LLM is not asked to control inventory online. It only writes small candidate policies offline. The simulator remains the judge, and consensus-seeking is encoded in explicit variables and gates. This keeps token cost low, makes the method reproducible, and gives a strong conceptual bridge from autonomous multi-agent consensus-seeking to executable inventory-control policy design.