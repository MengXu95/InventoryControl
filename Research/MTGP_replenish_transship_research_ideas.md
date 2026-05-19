# Research Ideas for Improving `MTGP_niching_replenish_transship`

## Core Direction

Turn the current two-tree MTGP replenishment/transshipment method into a **Rehearsal-Consensus MTGP**: an interpretable symbolic policy learner where each site behaves like a local agent, transshipment becomes a consensus decision, and candidate policies are trained to survive future demand trajectories rather than only minimize average one-rollout cost.

## What The Three Reference Papers Suggest

| Paper | Main Useful Message | How It Transfers To This Project |
| --- | --- | --- |
| `Agentic LLMs in the Supply Chain: Towards Autonomous Multi-Agent Consensus-Seeking` | Supply-chain decisions often require consensus among self-interested actors; inventory levels, delivery quantities, order frequency, and capacity allocation are natural consensus problems. | Treat each inventory site as a semi-autonomous agent. Transshipment should consider donor willingness, receiver urgency, fairness, and network benefit instead of only a direct pairwise quantity. |
| `From Topology to Trajectory: LLM-Driven World Models For Supply Chain Resilience` | Good planning needs physical grounding, latent trajectory rehearsal, reflection-in-action, and retrospective reflection-on-action under disruption. | Add short future rollouts or a learned surrogate world model to GP fitness, and use hindsight cost/regret to improve niching. |
| `Rethinking Supply Chain Planning: A Generative Paradigm` | Planning should connect high-level strategic intent with adaptive execution, rather than remain a static optimization routine. | Add strategic-intent variables such as service focus, risk tolerance, robustness priority, or transshipment aversion to scenario design and GP terminals. |

## Current Method Snapshot

The current `MTGP_niching_replenish_transship` method is a strong baseline because it is simple and interpretable:

- It evolves two symbolic GP trees: one replenishment tree and one transshipment tree.
- It evaluates three cost components: transshipment cost, holding/lost-sales cost, and order cost.
- The simulator state contains inventory level, holding/lost-sales cost, capacity, production capacity, fixed/per-unit order cost, pipeline, short forecast, and pairwise transshipment information.

The main research limitations are:

- Transshipment is a direct pairwise action, not a consensus process among sites.
- Fitness mostly rewards average cost, with limited pressure for resilience under unusual demand trajectories.
- Niching is available but currently not used, and it does not explicitly preserve different recovery strategies.
- The terminal set is local and short-horizon; it does not expose system-level imbalance, fairness, stress, or topology information.

## Flagship Idea: RC-MTGP

**Rehearsal-Consensus Multi-Tree Genetic Programming** keeps the current two-tree structure but adds two layers:

| Layer | What It Does | Where It Fits |
| --- | --- | --- |
| Consensus layer | Converts local inventory pressure into pairwise agreement signals before transshipment. | Add terminals to the transshipment state and optionally add an action gate in the simulator. |
| Rehearsal layer | Simulates short imagined futures and penalizes actions that look cheap now but cause later shortage or instability. | Add an optional `run_with_rehearsal` or new fitness evaluator in `GPFC.py`. |

The paper story is clean: symbolic GP policies become physically grounded, consensus-aware, and future-rehearsing, while staying interpretable.

## Idea 1: Consensus-Gated Transshipment

### Motivation

The consensus paper suggests that inventory and delivery decisions often require agreement among actors with different objectives. In the current simulator, a single transshipment tree directly outputs a transfer quantity. That is convenient, but it hides whether the donor site can afford to give inventory and whether the receiver site urgently needs it.

### Proposed Method

Before applying the transshipment tree, compute local agreement signals:

- **Donor willingness**: how safe the source site feels about giving inventory away.
- **Receiver urgency**: how much the destination site needs inventory.
- **Fairness debt**: whether one site has repeatedly acted as donor.
- **Network benefit**: expected reduction in lost sales minus transshipment cost.

Then gate the GP output:

```text
raw_transfer = GP_transshipment(pair_state)
consensus_score = min(donor_willingness, receiver_urgency)
accepted_transfer = raw_transfer * consensus_score
```

### Candidate Terminals

| Terminal | Meaning |
| --- | --- |
| `BAL` | inventory imbalance between two sites |
| `RISK1`, `RISK2` | shortage risk estimates for each site |
| `HELP12`, `HELP21` | cumulative historical transfer support between the pair |
| `AGREE` | pairwise agreement score |
| `NETB` | estimated network benefit of transfer |

### Expected Benefit

This should reduce harmful transshipment where one site is drained too early. It also makes the method easier to frame as decentralized coordination rather than central forced movement.

### First Experiment

Compare four variants:

- Baseline: current `MTGP_niching_replenish_transship`.
- Variant A: add agreement terminals only.
- Variant B: add fixed consensus gate only.
- Variant C: add both agreement terminals and consensus gate.

## Idea 2: Latent Trajectory Rehearsal Fitness

### Motivation

The world-model paper's most useful idea is rehearsal before commitment. Inventory control has delayed consequences because replenishment has lead time and transshipment changes future shortage risk. A policy that looks cheap now can create expensive lost sales later.

### Proposed Method

During fitness evaluation, branch into short imagined futures at selected periods:

```text
At period t:
        execute GP action
        sample K possible demand continuations for t+1 ... t+H
        simulate H future periods under the same policy
        add future risk penalty to fitness
```

A first version can use the existing demand generator, so no neural model is needed. Later, train a lightweight surrogate world model:

```text
z_t = encode(inventory, pipeline, forecast, imbalance)
z_{t+1} = world_model(z_t, replenishment_action, transshipment_action)
```

### Suggested Fitness

```text
fitness = average_cost
                                + lambda_shortage * expected_future_shortage
                                + lambda_instability * future_cost_variance
                                + lambda_recovery * time_to_recover_network_balance
```

### Expected Benefit

This should improve robustness on stress scenarios and high holding-cost asymmetry scenarios, especially when a locally cheap transshipment creates delayed shortage.

## Idea 3: Reflection-On-Action Niching

### Motivation

The current method has niching infrastructure, but `USE_Niching = False`. Standard behavior niching may preserve superficial decision differences. Inspired by reflection-on-action, niching can focus on how policies behave around high-regret moments.

### Proposed Method

After each rollout, identify critical periods:

- high lost-sales cost,
- large network imbalance,
- expensive transshipment,
- inventory collapse after previous shipment.

Then characterize the policy by its behavior around those periods:

```text
[order_before_shock, transship_before_shock,
 order_after_shock, transship_after_shock,
 recovery_time, max_shortage]
```

Use this vector for phenotypic niching instead of treating all normal decision states equally.

### Expected Benefit

The population should preserve meaningful recovery styles:

- conservative stock buffering,
- aggressive transshipment rescue,
- balanced replenishment/transshipment,
- low-order-cost delayed recovery.

### Implementation Hook

Use `InvOptEnv.run_to_get_decision(individual)` as the starting point, then add future-cost labels to decision situations and update the replenishment/transshipment phenotype characterization files.

## Idea 4: Strategic-Intent Conditioning

### Motivation

The generative planning paper frames planning as connecting human strategic intent with adaptive execution. In this repository, each run currently optimizes the same cost structure. Real managers may want different behavior under different operating modes.

### Proposed Method

Add intent variables to `ScenarioDesign_replenish_transship.py` and pass them into GP states:

| Terminal | Meaning |
| --- | --- |
| `SLT` | service-level target |
| `RISK` | risk tolerance |
| `TAV` | transshipment aversion |
| `BUDG` | ordering budget pressure |
| `ROB` | robustness priority |

Example modes:

| Mode | Behavior |
| --- | --- |
| Lean | minimize holding/order cost, tolerate more service risk |
| Service | strongly penalize lost sales |
| Resilience | penalize cost variance and slow recovery |
| Collaboration | penalize unfair repeated donor/receiver patterns |

### Expected Benefit

The method can produce controllable symbolic policy families instead of one fixed policy. This is useful for paper writing because it connects optimization to planning intent.

## Idea 5: Topology-To-Trajectory Features

### Motivation

The current simulator supports 2-site and 3-site transshipment explicitly. For larger multi-site systems, transshipment should depend on network topology and global imbalance, not only pairwise local states.

### Proposed Method

For each pair, add graph/system features:

- total system inventory,
- total expected demand,
- site surplus rank,
- site shortage rank,
- pair distance or transfer friction,
- network imbalance entropy,
- nearby surplus and nearby shortage pressure.

For the current 3-site setting, start with:

```text
system_inventory = sum(INL_i)
system_forecast = sum(FC1_i + FC2_i)
imbalance_entropy = std(INL_i - expected_demand_i)
surplus_rank_i, surplus_rank_j
```

### Expected Benefit

This makes the method more scalable and helps avoid locally sensible but globally poor inventory movements.

## Recommended Research Priority

| Priority | Idea | Why |
| --- | --- | --- |
| 1 | Consensus-gated transshipment | Strong conceptual novelty and easy to implement incrementally. |
| 2 | Reflection-on-action niching | Uses existing niching machinery and gives operationally meaningful diversity. |
| 3 | Trajectory rehearsal fitness | Strongest resilience story, but increases evaluation cost. |
| 4 | Strategic-intent conditioning | Good second-stage extension after the method improves baseline cost. |
| 5 | Topology-to-trajectory features | Important for scaling beyond small networks. |

## Suggested Paper Concept

### Working Title

**Rehearsal-Consensus Genetic Programming for Multi-Site Inventory Replenishment and Transshipment**

### Research Question

Can symbolic GP policies become more robust and realistic for multi-site inventory control by combining consensus-aware transshipment with future trajectory rehearsal?

### Contributions To Test

1. A consensus-aware transshipment mechanism that models donor willingness, receiver urgency, and fairness debt.
2. A trajectory rehearsal fitness term that improves robustness under demand uncertainty and lead-time delay.
3. A reflection-on-action niching method that preserves diverse crisis-response behaviors.
4. An interpretable symbolic policy representation that remains inspectable compared with black-box DRL.

### Baselines

- Current `MTGP_niching_replenish_transship`.
- `MTGP_niching_replenish_transship_price` / `InventoryGP`, if pricing/RFQ comparison is relevant.
- `MTGP_niching` for older replenishment/transshipment behavior.
- DRL/PPO and sS policy if runtime budget allows.

### Metrics

- Total average cost.
- Holding/lost-sales cost.
- Order cost.
- Transshipment cost.
- Service level or stockout frequency.
- Transshipment frequency.
- Fairness of inventory support between sites.
- Recovery time after shock.
- Robustness gap between normal and stress-test scenarios.

## Minimal First Implementation

Start with **Consensus-Gated Transshipment**, because it is the clearest paper idea and lowest-risk simulator change:

1. Copy `MTGP_niching_replenish_transship` to a new variant, for example `MTGP_niching_replenish_transship_consensus`.
2. Add donor willingness, receiver urgency, and fairness debt to `Inventory_simulator_replenish_transship.py`.
3. Add matching terminals to `multi_tree.py`.
4. Keep GP hyperparameters unchanged for the first comparison.
5. Run small scenarios first:

```powershell
python main.py sN2h_1_5b2 0 MTGP_niching_replenish_transship
python main.py sN3h_1_5_10b2 0 MTGP_niching_replenish_transship
```

Do not claim improvement until comparing several seeds and several holding-cost asymmetry scenarios.

## Suggested Abstract Seed

Multi-location inventory systems require replenishment and transshipment decisions that are both locally responsive and globally coordinated. Existing GP-based symbolic policies provide interpretability, but they often optimize average simulated cost without explicitly modeling consensus among sites or resilience under future demand trajectories. We propose Rehearsal-Consensus MTGP, an interpretable multi-tree genetic programming framework for joint replenishment and transshipment. The method augments symbolic policy learning with consensus-aware state signals, short-horizon trajectory rehearsal, and reflection-based behavioral niching. This design encourages policies that not only reduce inventory cost, but also recover from demand shocks and avoid unfair or unstable transshipment patterns. Experiments on synthetic and real-demand multi-site inventory scenarios can evaluate total cost, service behavior, recovery time, and robustness across seeds.

## What The Three Reference Papers Suggest

| Paper | Useful Research Signal | Relevance To This Project |
| --- | --- | --- |
| `Agentic LLMs in the Supply Chain: Towards Autonomous Multi-Agent Consensus-Seeking` | Supply-chain decisions often require consensus among self-interested actors; inventory levels, delivery quantities, order frequency, and capacity allocation are natural consensus problems. | Transshipment is already a coordination action, but the current GP tree directly outputs transfers without modeling negotiation, fairness, or agreement. |
| `From Topology to Trajectory: LLM-Driven World Models For Supply Chain Resilience` | Good planning needs physical grounding, latent trajectory rehearsal, reflection-in-action, and retrospective reflection-on-action under black-swan disruptions. | Current MTGP evaluates policies directly in the simulator, but does not rehearse counterfactual futures or learn from high-cost trajectory segments. |
| `Rethinking Supply Chain Planning: A Generative Paradigm` | Planning should become an interactive cognitive process that connects strategic intent with adaptive execution, rather than a static optimization routine. | Current GP has no explicit interface for service-level preference, risk appetite, shortage tolerance, or scenario intent. These could become policy-conditioning inputs or experiment controls. |

## Main Proposal: Rehearsal-Consensus MTGP

The most promising publishable idea is **Rehearsal-Consensus MTGP**, a symbolic GP method that keeps the interpretability of MTGP but adds two missing capabilities:

1. **Consensus-aware transshipment**: treat each retailer as an agent with local preferences, and evolve transshipment rules that must satisfy a lightweight agreement mechanism.
2. **Trajectory rehearsal**: evaluate candidate policies not only on the sampled demand path, but also on short simulated future rollouts that expose hidden shortage risk and delayed pipeline effects.

This gives a neat paper story: symbolic policies are no longer merely reactive rules; they become physically grounded, consensus-aware, future-rehearsing decision rules for multi-site inventory control.

## Idea 1: Consensus-Aware Transshipment Tree

### Motivation

The transshipment tree currently outputs a signed quantity for each pair of sites. For two sites, positive means ship from site 0 to site 1; negative means ship from site 1 to site 0. This is operationally simple, but it assumes a central planner can force the movement. The consensus paper suggests a more realistic framing: each site has its own objective and must agree to the transfer.

### Mechanism

Instead of evolving only one transshipment quantity, define an agreement score around each possible transfer:

- Donor willingness: how safe it is for the source site to give inventory away.
- Receiver urgency: how much the destination site needs inventory.
- Fairness debt: whether one site has repeatedly helped the other without reciprocal benefit.
- Network benefit: expected reduction in total lost sales minus transshipment cost.

The evolved transshipment tree can still output a quantity, but the simulator only accepts it when it passes an agreement gate:

```text
accepted_transfer = proposed_transfer
    if donor_willingness > threshold and receiver_urgency > threshold
    else 0
```

The thresholds can be fixed, evolved as ephemeral constants, or generated by a third small symbolic tree.

### New State Terminals

For transshipment, add terminals such as:

| Terminal | Meaning |
| --- | --- |
| `BAL` | inventory imbalance between two sites |
| `RISK1`, `RISK2` | shortage risk estimates for each site |
| `HELP12`, `HELP21` | cumulative historical transfer support between the pair |
| `AGREE` | current pairwise agreement score |
| `NETB` | estimated total-cost benefit of the proposed transfer |

### Why It Is Novel Here

This makes transshipment more than routing inventory. It becomes a symbolic consensus policy. Compared with normal MTGP, the contribution is not only better cost; it is a more realistic coordination mechanism for decentralized multi-site inventory systems.

### First Experiment

Compare:

- Baseline: current `MTGP_niching_replenish_transship`.
- Variant A: add agreement features but no gate.
- Variant B: add agreement gate with fixed thresholds.
- Variant C: evolve the agreement threshold.

Recommended small smoke scenarios:

```powershell
python main.py sN2h_1_5b2 0 MTGP_niching_replenish_transship
python main.py sN3h_1_5_10b2 0 MTGP_niching_replenish_transship
```

## Idea 2: Latent Trajectory Rehearsal Fitness

### Motivation

The world-model paper argues that planners fail when they only react to the current state and lack a grounded model of future trajectories. Inventory control has exactly this problem: an action may look good now but create lost sales several periods later because of lead time and pipeline delay.

### Mechanism

During fitness evaluation, when a GP individual reaches a decision state, run a few cheap future rollouts before scoring the individual. The rollouts do not need an LLM or neural world model at first. A lightweight empirical world model is enough:

```text
current state -> sample K short demand futures -> simulate H periods -> estimate trajectory risk
```

Then add trajectory-risk penalties to fitness:

```text
fitness = average_cost
        + lambda_shortage * future_shortage_risk
        + lambda_instability * transfer_oscillation_risk
        + lambda_capacity * future_capacity_violation_risk
```

### Fancy Version

Train a small surrogate model from simulator rollouts to predict the next latent state:

```text
z_t = encode(inventory, pipeline, forecast, pairwise imbalance)
z_{t+1} = world_model(z_t, replenishment_action, transshipment_action)
```

Then use the surrogate to rehearse many futures cheaply during GP evaluation. This mirrors the world-model paper's “latent trajectory rehearsal,” but in a symbolic inventory-control setting.

### Why It Is Novel Here

Most GP inventory policies are evaluated by direct rollout only. Adding rehearsal turns GP from a reactive rule learner into a lookahead symbolic planner while preserving interpretability.

### First Experiment

Start with the non-neural version:

- `K = 3` demand futures.
- `H = 4` future periods.
- Penalty only for future lost sales.

If this improves test robustness, then try the learned latent world model.

## Idea 3: Reflection-On-Action Niching

### Motivation

The current method has niching infrastructure, but `USE_Niching = False`. Existing niching is based on phenotypic characterisation of decisions. The world-model paper suggests a stronger idea: after seeing the future outcome, retrospectively identify which earlier decisions caused the bad trajectory.

### Mechanism

After each rollout, collect high-regret moments:

```text
regret_t = future_cost_after_t - local_cost_at_t
```

Then characterize individuals by what they do in high-regret states, not all states equally. This makes niching focus on meaningful behavior differences:

- shortage rescue behavior,
- over-order recovery behavior,
- emergency transshipment behavior,
- pipeline-delay handling behavior.

### Expected Benefit

The population should maintain diverse crisis-handling strategies instead of many superficially different trees that behave similarly in normal states.

### Implementation Hook

Use `InvOptEnv.run_to_get_decision(individual)` as the starting point, but add future cost labels to decision situations. Then update `niching/ReplenishmentPhenoCharacterisation.py` and `niching/TransshipmentPhenoCharacterisation.py` to weight high-regret decisions more heavily.

## Idea 4: Generative Strategic-Intent Conditioning

### Motivation

The generative planning paper frames planning as connecting human strategic intent with adaptive execution. In this project, every run currently optimizes the same cost structure. But real inventory managers may prefer different behaviors:

- aggressive service level,
- conservative ordering cost,
- low transshipment frequency,
- balanced fairness across sites,
- resilience under demand spikes.

### Mechanism

Add scenario-level intent variables to the GP terminal set:

| Terminal | Meaning |
| --- | --- |
| `SLT` | service-level target |
| `RISK` | risk tolerance |
| `TFRQ` | target transshipment frequency |
| `BUDG` | ordering budget pressure |
| `ROB` | robustness priority |

These values can be set in `ScenarioDesign_replenish_transship.py` and passed into each retailer/pair state. Then a single evolved policy can adapt to different strategic settings.

### Paper Angle

This supports a strong title such as:

> Intent-Conditioned Symbolic Policies for Multi-Site Inventory Replenishment and Transshipment

The contribution is not just lower cost. It is controllable symbolic planning.

## Idea 5: Topology-To-Trajectory Transshipment Features

### Motivation

The current simulator supports two-site and three-site transshipments, but the state is pairwise and mostly local. For three or more sites, transshipment should depend on the network topology and future trajectory of imbalance, not just one pair.

### Mechanism

Add graph-level features:

- site centrality,
- total system inventory,
- total expected demand,
- site surplus rank,
- site shortage rank,
- pair distance or transfer friction,
- network-level imbalance entropy.

The transshipment tree then sees both pairwise state and network pressure. This helps avoid locally sensible but globally poor transfers.

### First Implementation

For the existing three-site scenario, compute:

```text
system_inventory = sum(INL_i)
system_forecast = sum(FC1_i + FC2_i)
imbalance_entropy = std(INL_i - expected_demand_i)
surplus_rank_i, surplus_rank_j
```

Add these to the transshipment state and primitive terminal set.

## Recommended Priority

| Priority | Idea | Why First |
| --- | --- | --- |
| 1 | Reflection-on-action niching | Uses existing niching infrastructure and fits GP naturally. |
| 2 | Consensus-aware transshipment | Strong conceptual novelty and easy to explain in a paper. |
| 3 | Trajectory rehearsal fitness | Likely improves robustness but increases evaluation cost. |
| 4 | Strategic-intent conditioning | Good for a second paper or extension after baseline improvement. |
| 5 | Topology-to-trajectory features | Important for scaling beyond small networks. |

## A Strong Paper Concept

### Working Title

**Rehearsal-Consensus Genetic Programming for Multi-Site Inventory Replenishment and Transshipment**

### Core Research Question

Can symbolic GP policies become more robust and realistic for multi-site inventory control by combining consensus-aware transshipment with future trajectory rehearsal?

### Claimed Contributions To Test

1. A consensus-aware transshipment mechanism that models donor willingness, receiver urgency, and fairness debt.
2. A trajectory rehearsal fitness term that improves robustness under demand uncertainty and lead-time delay.
3. A reflection-on-action niching method that preserves diverse crisis-response behaviors.
4. An interpretable symbolic policy representation that remains inspectable compared with black-box DRL.

### Baselines

- `MTGP_niching_replenish_transship`.
- `MTGP_niching_replenish_transship_price` / `InventoryGP`, if pricing/RFQ comparison is relevant.
- `MTGP_niching` without the new transshipment-specific mechanisms.
- DRL/PPO and sS policy if runtime budget allows.

### Metrics

- Total average cost.
- Holding/lost-sales cost.
- Order cost.
- Transshipment cost.
- Service level or stockout frequency.
- Transshipment frequency.
- Fairness of inventory support between sites.
- Robustness gap between training scenarios and stress-test scenarios.

## Minimal Next Step

The smallest meaningful implementation is **Reflection-On-Action Niching**:

1. Enable `USE_Niching = True` in a copied experiment variant, not the baseline.
2. Label decision situations by future cost/regret.
3. Weight phenotypic distance toward high-regret periods.
4. Compare against current `MTGP_niching_replenish_transship` on `sN2h_1_5b2` and `sN3h_1_5_10b2`.

This is a clean first experiment because it changes only selection/diversity pressure, not the simulator physics.# Research Ideas for Improving `MTGP_niching_replenish_transship`

## One-Sentence Direction

Turn the current two-tree MTGP replenishment/transshipment method into a **consensus-aware, rehearsal-based symbolic policy learner**: each retailer remains interpretable through GP rules, but the rules are trained with imagined future trajectories, local negotiation signals, and resilience-oriented behavioral diversity.

## Papers Reviewed

The ideas below are based on the three reference papers in `Research/Reference`:

| Paper | Main Useful Message | How It Transfers To This Project |
|---|---|---|
| `Agentic LLMs in the Supply Chain: Towards Autonomous Multi-Agent Consensus-Seeking` | Supply-chain decisions often require consensus among self-interested actors; automated agents can negotiate order quantities, delivery timing, and capacity allocation. | Treat each inventory site as a semi-autonomous agent and make transshipment a consensus/coordination decision, not only a numeric pairwise action. |
| `From Topology to Trajectory: LLM-Driven World Models For Supply Chain Resilience` | Long-horizon planning improves when agents rehearse future trajectories in a grounded world model and use reflection-in-action plus reflection-on-action. | Add surrogate trajectory rehearsal and hindsight feedback to GP fitness/niching, so policies are selected for future recoverability, not only immediate average cost. |
| `Rethinking Supply Chain Planning: A Generative Paradigm` | Planning should be an interactive, integrated cognitive process that turns strategic intent into structured analytical workflows. | Let a high-level strategic intent, such as cost-minimization, service protection, or resilience, shape scenario generation, objective weighting, and policy interpretation. |

## Current Method: What Is Strong And What Is Missing

The current `MTGP_niching_replenish_transship` method is a clean and useful baseline:

- It evolves two symbolic GP policies: one replenishment tree and one transshipment tree.
- It evaluates three costs: transshipment cost, holding/lost-sales cost, and order cost.
- Its state uses inventory level, holding/lost-sales cost, capacity, pipeline, short forecast, and pairwise transshipment information.
- It is interpretable and easy to compare against existing MTGP/InventoryGP variants.

The main research limitations are:

- Transshipment is treated as a direct pairwise quantity, but not as a consensus process among sites.
- Fitness is mostly average cost over sampled demand, with limited pressure for resilience under unusual trajectories.
- Niching is optional and currently behavior-focused, but it does not explicitly preserve different recovery strategies.
- The terminal set is local and short-horizon; it does not expose system-level imbalance, fairness, stress, or topology information.

## Flagship Idea: RC-MTGP, Rehearsal-Consensus Multi-Tree GP

### Core Claim

**RC-MTGP learns interpretable replenishment and transshipment policies that coordinate decentralized sites through consensus signals and future-trajectory rehearsal.**

Instead of asking only, "What quantity should I order or ship now?", the method asks:

1. What does each site want locally?
2. What does the network need globally?
3. Which candidate action still looks good after rehearsing several possible future demand trajectories?

### Mechanism

Keep the two-tree structure, but enrich training with two extra layers:

| Layer | What It Does | Where It Fits |
|---|---|---|
| Consensus layer | Converts local inventory pressure into pairwise agreement signals before transshipment. | Add terminals to the transshipment tree and/or a lightweight pre-action calculation in the simulator. |
| Rehearsal layer | Simulates short imagined futures for candidate actions and penalizes actions that look cheap now but cause future shortage or instability. | Add a new fitness evaluator or an optional `run_with_rehearsal` mode. |

### New State Signals

Add terminals that are still interpretable:

| Terminal | Meaning | Why It Helps |
|---|---|---|
| `NIMB` | Network inventory imbalance, such as max inventory pressure minus min inventory pressure. | Encourages transshipment only when the network is genuinely uneven. |
| `LREQ1`, `LREQ2` | Local required inventory: forecast plus pipeline-adjusted safety need. | Gives each site a clearer local demand request. |
| `GREQ` | Total network requirement across all sites. | Helps avoid myopic movement that fixes one site but hurts the system. |
| `FAIR` | Accumulated fairness debt or historical net shipped amount for a site/pair. | Prevents one site from becoming a permanent donor. |
| `RISK1`, `RISK2` | Shortage risk proxy from inventory, pipeline, forecast, and lost-sales cost. | Makes transshipment sensitive to service risk, not only inventory level. |
| `STRESS` | Scenario stress score, such as demand percentile or forecast shock. | Helps evolve policies that change behavior under disruption. |

This is a good paper angle because the terminals are not black-box features. They can be explained as supply-chain coordination variables.

## Fancy Idea 1: Consensus-Gated Transshipment

### Motivation

The consensus paper argues that inventory and delivery decisions often need agreement among actors with different goals. In the current simulator, a single transshipment tree directly outputs a quantity. That is simple, but it hides the idea that site A and site B may disagree.

### Proposed Method

Before applying the transshipment tree, compute two local willingness scores:

- Donor willingness: how safe site A feels about giving inventory.
- Receiver urgency: how badly site B needs inventory.

Then gate the transshipment action:

```text
raw_transship = GP_transship(pair_state)
consensus_score = min(donor_willingness, receiver_urgency)
final_transship = raw_transship * consensus_score
```

The willingness scores can be hand-designed first, then evolved later.

### Expected Benefit

This should reduce harmful transshipment where one site is stripped of inventory too early. It may improve lost-sales cost and make learned rules easier to defend as "coordinated" rather than purely reactive.

### First Experiment

Compare:

- Baseline: current `MTGP_niching_replenish_transship`.
- Variant A: hand-designed consensus gate.
- Variant B: GP terminal set includes consensus variables but no gate.
- Variant C: both gate and terminals.

## Fancy Idea 2: Latent Trajectory Rehearsal For GP Fitness

### Motivation

The world-model paper's useful idea is not necessarily "use an LLM". The useful idea is **rehearsal before commitment**: evaluate whether an action creates a bad future trajectory.

### Proposed Method

During fitness evaluation, for selected periods, branch into several short imagined futures:

```text
At period t:
  execute GP action
  sample K possible demand continuations for t+1 ... t+H
  estimate recovery cost under the same policy
  add rehearsal penalty to fitness
```

The first version does not need a neural world model. Use the existing demand generator as a grounded simulator. Later, a learned surrogate can replace repeated simulation to reduce cost.

### Suggested Fitness

```text
fitness = average_cost
        + lambda_risk * expected_future_shortage
        + lambda_instability * variance_of_future_cost
        + lambda_recovery * time_to_recover_network_balance
```

### Expected Benefit

This encourages policies that are robust, not just cheap on average. It is especially promising for high holding-cost asymmetry and 3-site scenarios.

## Fancy Idea 3: Reflection-On-Action Niching

### Motivation

The current niching machinery can preserve behavioral diversity, but the behavior characterization can be more meaningful. Inspired by reflection-on-action, characterize policies by how they recover after bad decisions or demand shocks.

### Proposed Method

After each rollout, identify "critical periods":

- high lost-sales cost,
- large network imbalance,
- expensive transshipment,
- inventory collapse after a previous shipment.

Then build a phenotype vector from what the policy did around those periods:

```text
[order_before_shock, transship_before_shock, order_after_shock,
 transship_after_shock, recovery_time, max_shortage]
```

Use this vector for niching instead of, or in addition to, ordinary decision-point behavior.

### Expected Benefit

The population should keep multiple recovery styles:

- conservative stock buffering,
- aggressive transshipment rescue,
- balanced replenishment/transshipment,
- low-order-cost delayed recovery.

This is more publishable than generic niching because the diversity has an operational meaning.

## Fancy Idea 4: Strategic-Intent Conditioning

### Motivation

The generative planning paper frames planning as translating strategic intent into structured workflows. For this project, strategic intent can become a controllable experimental condition.

### Proposed Method

Add an experiment-level intent vector:

```text
intent = [cost_focus, service_focus, resilience_focus, transshipment_aversion]
```

Then use it in one of three ways:

- As terminals available to both GP trees.
- As dynamic weights for the three cost objectives.
- As scenario generator settings that create different demand stress profiles.

### Example Modes

| Mode | Objective Behavior |
|---|---|
| Lean mode | Minimize holding and order cost, tolerate some service risk. |
| Service mode | Strongly penalize lost sales and stockout recovery time. |
| Resilience mode | Penalize high variance and slow recovery under shocks. |
| Collaboration mode | Penalize unfair repeated donor/receiver patterns. |

### Expected Benefit

One evolved method can produce policy families, not just one policy. This gives a strong paper story: interpretable symbolic planning under strategic intent.

## Fancy Idea 5: Topology-Aware Transshipment For More Than Three Sites

### Motivation

The current simulator handles 2-site and 3-site cases explicitly. The world-model paper emphasizes moving from topology to trajectory. A natural extension is to make transshipment aware of network topology and scalable to more locations.

### Proposed Method

Represent sites as a graph. For each pair, add graph-aware terminals:

- pair distance or lane cost,
- site degree/centrality,
- upstream/downstream role,
- neighborhood shortage pressure,
- total nearby surplus.

Then replace hard-coded 2-site/3-site transshipment application with a general pairwise flow loop plus feasibility repair.

### Expected Benefit

This makes the method look less like a small simulator trick and more like a general multi-location inventory-control method.

## Fancy Idea 6: Symbolic Policy With Test-Time Micro-Adaptation

### Motivation

The world-model paper discusses test-time policy evolution. For GP, fully changing tree structure online may be unstable, but adapting small constants is feasible.

### Proposed Method

Keep tree structure fixed after training, but allow a few scalar knobs to adapt during testing:

```text
order_scale
transship_scale
shortage_risk_threshold
consensus_gate_temperature
```

Use a small rolling-window update rule after each episode or after every few periods.

### Expected Benefit

This preserves interpretability while giving the policy a lightweight ability to respond to demand-regime shifts.

## Fancy Idea 7: LLM-As-Research-Orchestrator, Not LLM-As-Controller

### Motivation

Using an LLM directly as a controller may be expensive and hard to reproduce. A more suitable role is to generate scenarios, name behavioral patterns, and write explanations for symbolic policies.

### Proposed Method

Use LLM support offline to:

- generate stress-test scenario descriptions,
- translate strategic intent into simulator parameter sweeps,
- summarize evolved GP trees in supply-chain language,
- classify policies into archetypes such as buffer-first, ship-first, or balanced recovery.

### Expected Benefit

The method remains reproducible because the optimizer is still GP and the simulator is deterministic under seeds. The LLM improves research workflow and interpretability rather than becoming an uncontrolled black box.

## Recommended First Paper Contribution

The strongest publishable path is:

> **Rehearsal-Consensus MTGP for Multi-Location Inventory Replenishment and Transshipment**

### Main Contribution

An interpretable multi-tree GP framework that combines:

1. consensus-aware transshipment features,
2. trajectory rehearsal during fitness evaluation,
3. reflection-based niching for recovery-behavior diversity.

### Why It Is Novel Enough

It is not just "add more terminals". The method changes the learning target from static average-cost minimization to coordinated and resilient trajectory control, while keeping symbolic policy interpretability.

### Baselines

- Current `MTGP_niching_replenish_transship`.
- `MTGP_niching_replenish_transship_price` / `InventoryGP` if using the broader inventory-control framing.
- `MTGP_niching` if comparing against older replenishment/transshipment behavior.
- sS policy and DRL/PPO where available.

### Evaluation Metrics

| Metric | Reason |
|---|---|
| Total cost | Main optimization target. |
| Holding/lost-sales cost | Service-quality pressure. |
| Transshipment cost | Coordination cost. |
| Order cost | Replenishment efficiency. |
| Stockout frequency | More interpretable than cost alone. |
| Recovery time after shock | Measures resilience. |
| Fairness debt | Measures whether one site is overused as donor. |
| Cost variance across scenarios | Measures robustness. |

## Minimal Implementation Roadmap

### Stage 1: Low-Risk Feature Variant

Add consensus/risk/topology terminals only. No new fitness yet.

Expected changed files:

- `MTGP_niching_replenish_transship/Inventory_simulator_replenish_transship.py`
- `MTGP_niching_replenish_transship/multi_tree.py`

### Stage 2: Rehearsal Fitness Variant

Add optional short-horizon future rehearsal in evaluation.

Expected changed files:

- `MTGP_niching_replenish_transship/GPFC.py`
- `MTGP_niching_replenish_transship/Inventory_simulator_replenish_transship.py`

### Stage 3: Reflection Niching Variant

Change phenotype characterization to focus on shock/recovery behavior.

Expected changed files:

- `MTGP_niching_replenish_transship/niching/niching.py`
- `MTGP_niching_replenish_transship/niching/*PhenoCharacterisation.py`

## Small Validation Plan

Start with cheap synthetic scenarios:

```powershell
python main.py sN2h_1_5b2 0 MTGP_niching_replenish_transship
python main.py sN3h_1_5_10b2 0 MTGP_niching_replenish_transship
```

Then compare across seeds:

```powershell
python main.py sN2h_1_5b2 0 MTGP_niching_replenish_transship
python main.py sN2h_1_5b2 1 MTGP_niching_replenish_transship
python main.py sN2h_1_5b2 2 MTGP_niching_replenish_transship
```

For the paper, avoid claiming improvement until testing against at least 3-5 seeds and several holding-cost asymmetry scenarios.

## Suggested Paper Abstract Seed

Multi-location inventory systems require replenishment and transshipment decisions that are both locally responsive and globally coordinated. Existing GP-based symbolic policies provide interpretability, but they often optimize average simulated cost without explicitly modeling consensus among sites or resilience under future demand trajectories. We propose Rehearsal-Consensus MTGP, an interpretable multi-tree genetic programming framework for joint replenishment and transshipment. The method augments symbolic policy learning with consensus-aware state signals, short-horizon trajectory rehearsal, and reflection-based behavioral niching. This design encourages policies that not only reduce inventory cost, but also recover from demand shocks and avoid unfair or unstable transshipment patterns. Experiments on synthetic and real-demand multi-site inventory scenarios can evaluate total cost, service behavior, recovery time, and robustness across seeds.

## My Recommendation

Implement **Fancy Idea 1 + Fancy Idea 2** first. Consensus-gated transshipment is easy to explain and relatively cheap to test. Rehearsal fitness gives the work a stronger research identity. Reflection niching is attractive, but it should come after the first two pieces work, because it changes the evolutionary dynamics more deeply.