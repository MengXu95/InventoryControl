# LLM Policy Evolution

This folder implements the minimal `ConsensusCode-Evolve` workflow from `Research/LLM_code_evolution_consensus_ideas.md`.

Run a small seed-policy evaluation with:

```powershell
python main.py sN2h_1_5b2 0 LLM_policy_evolution
```

The evaluator:

1. loads candidate Python policies from `LLM_policy_evolution/seed_policies/`,
2. validates them with a restrictive AST whitelist,
3. adapts them to `MTGP_niching_replenish_transship.InvOptEnv`,
4. evaluates replenishment, transshipment proposal, and consensus gate functions,
5. writes scores and archived candidate source files to `Research/llm_policy_runs/scenario_<dataset>/`.

Candidate policy files must define exactly these functions:

```python
def replenishment_policy(site_state, global_state):
    return order_quantity

def transshipment_proposal(source_state, target_state, global_state):
    return proposed_quantity

def consensus_gate(source_state, target_state, proposed_quantity, history):
    return accepted_quantity
```

Generated policies are intentionally limited: no imports, no file access, no random calls, no attributes, and only `abs`, `min`, `max`, and `round` function calls.

Create the next LLM mutation prompt with:

```powershell
python main.py sN2h_1_5b2 0 LLM_policy_evolution_next
```

This writes `llm_mutation_prompt.md` and a `generated_candidates/` folder under `Research/llm_policy_runs/scenario_<dataset>/round_<n>/`. Put LLM-generated `.py` files into that folder, then run the same command again to validate and evaluate them.

Alternatively, paste the complete LLM answer into `llm_response.md` inside the round folder. The next run will extract fenced Python blocks into candidate files automatically before evaluation.