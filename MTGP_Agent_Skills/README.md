# PACT-GP: Three-Tree GP + Fixed Peer Negotiation

**PACT-GP** means **Peer-Agent Coordination with Three-tree Genetic
Programming**. The name evokes a pact between peer sites: GP evolves their
local decision skills, while a fixed protocol confirms conditional support
commitments. The paper title is **PACT-GP: Genetic Programming for Skill-Based
Agent Coordination in Multi-Site Inventory Control**.

Independent executable research variant, **not** an alias for the original
`MTGP_niching_replenish_transship` two-tree joint-state policy. The original
package, shared utilities, root entry point, and manuscript directory are
unchanged. The 21 copied Python files are preserved without algorithm changes in
`legacy/` as provenance only (moving normalized some trailing blank lines);
**never run/import them**. Active code imports only this
package, third-party libraries, and read-only `Utils` scenario/demand utilities.
There is no LLM, centralized runtime policy, global matching solver, niching,
brood recombination, or claim of superior results.

## Run (repository root, Windows PowerShell)

The verified interpreter is CPython 3.11.9:

```powershell
$py = 'C:\Users\xumen\AppData\Local\Programs\Python\Python311\python.exe'
& $py -m unittest discover -s MTGP_Agent_Skills/tests -v
& $py -m compileall -q MTGP_Agent_Skills

# Full 64-period demand horizon; small search, not a benchmark-quality run.
& $py -m MTGP_Agent_Skills smoke --seed 0 --output-dir MTGP_Agent_Skills/outputs/site_service_seed0

# Reload canonical policy and exact saved simulator settings on held-out seeds.
& $py -m MTGP_Agent_Skills evaluate `
  --policy MTGP_Agent_Skills/outputs/site_service_seed0/best_policy.json `
  --config MTGP_Agent_Skills/outputs/site_service_seed0/config.json `
  --seeds 2001,2002 --output-dir MTGP_Agent_Skills/outputs/site_service_replay_seed0

# Study-budget search, still serial. Set all three seed batches explicitly.
& $py -m MTGP_Agent_Skills train --scenario sN2h_1_5b2 --seed 17 `
  --population 200 --generations 50 --train-seeds 11,12,13 `
  --validation-seeds 1001,1002,1003 --test-seeds 2001,2002,2003 `
  --output-dir MTGP_Agent_Skills\outputs\pact_gp_train_seed17

# Generic N and an undirected path; default topology is complete.
& $py -m MTGP_Agent_Skills smoke --scenario sN5h_1_2_3_4_5b2 `
  --topology 0-1,1-2,2-3,3-4 --rounds 3 `
  --output-dir MTGP_Agent_Skills/outputs/five_sites
```

Unqualified `python -m MTGP_Agent_Skills ...` is equivalent when that interpreter
is active. No install is necessary in the verified environment. `requirements.txt`
records the used versions of NumPy, DEAP, and pandas (pandas is imported by the
shared scenario utility). Tests use only `unittest`; no pytest dependency.
The module entry point preserves `freeze_support()` and the main guard.
Training is deliberately single-process.

Output directories must be empty to prevent accidental overwrite. Default:
`MTGP_Agent_Skills/outputs/<train|smoke|evaluate>/`, never old Research archives.
`--periods k` executes a prefix **without changing full-episode demand generation**.
`--topology none` or `--rounds 0` disables peer promises for a controlled ablation.
CLI evaluation uses a complete topology unless a saved config or CLI topology
is provided; use `--config` for exact replay. Explicit CLI settings override
saved settings and are recorded in the new output config.

### Complete CLI surface and defaults

| Flags | Commands | Default without replay config |
|---|---|---|
| `--output-dir PATH` | all | package `outputs/<command>`; must be empty |
| `--scenario NAME` | all | `sN2h_1_5b2` |
| `--rounds R` | all | 3; zero disables promises |
| `--periods T` | all | scenario horizon, normally 64 |
| `--topology complete\|none\|0-1,1-2,...` | all | complete, undirected |
| `--service-target X` | all | 0.95 |
| `--allow-oracle-forecasts` | all | false |
| `--no-trace` | all | false (traces enabled) |
| `--seed S` | train/smoke | 0; search RNG, not an episode seed |
| `--population P` | train/smoke | 200 / 6 |
| `--generations G` | train/smoke | 50 / 1; excludes generation 0 |
| `--train-seeds a,b,...` | train/smoke | 11,12 |
| `--validation-seeds a,b,...` | train/smoke | 1001,1002 |
| `--test-seeds a,b,...` | train/smoke | 2001,2002 |
| `--policy PATH` | evaluate | required canonical JSON |
| `--config PATH` | evaluate | optional saved simulator settings/scenario |
| `--seeds a,b,...` | evaluate | 2001,2002, even when a config is supplied |

`--help`/`-h` is available on the module and each subcommand. There are no CLI
flags for LT/L, variation probabilities, tournament size, or demand seed offset;
these are shared-scenario/Python-API settings, recorded in `config.json`.
The Python API defaults to `demand_seed_offset=1000`. A saved replay config
supplies scenario/settings, not evaluation seeds or a raw stream snapshot.

For the default scenario, N=2, T=64, L=LT=2, initial stocks `[10,10]`,
holding `[1,5]`, lost-sales costs `[2,10]`, capacity `[100,100]`, fixed order
cost `[20,20]`, unit order cost `[1,1]`, transfer unit cost 1 and fixed cost 10.
The demand-mean scale is 20. Recorded production capacity `[20,20]` is unused.
The name suffix `b2` sets the lost-sales multiplier, not LT.

## Architecture and information boundary

`policy.py` implements one shared individual containing **exactly three DEAP
trees in order `(order, request, offer)`**. Every site has its own `Agent`
(`skills.py`), immutable local observation, and memory. Each tree drives just
its corresponding fixed bounded skill. A site never reads another site's
inventory, pipeline, raw forecasts, realized future demand, or any global total.
Runtime interactions consist only of neighbor messages and fixed confirmations.
The offline simulator may hold all states, generate demand, deliver mail, enforce
physical feasibility, compute fitness, and write central traces; those capabilities
are **not** given to the agents.

Messages contain routing ID, fixed role, remaining request/offer, net forecast
slack, holding/lost-sales costs, and reserved inbound/outbound quantities. Routing
IDs are used only for delivery and deterministic ties, never GP features.
Transfer costs are homogeneous public parameters, as in the benchmark.
Memory features are previous-period executed inbound/outbound and lost quantity;
history records only actual positive transfers, including partial executions.
Cancelled/planned transfers do not enter executed history. Summaries reset each
period, including periods without transfers/losses.

All sites evaluate their order tree once at period start and only the tree
corresponding to their fixed request/offer role once (neither for idle sites).
The individual still always contains all three trees. GP features, role and
initial skill quantity remain frozen throughout negotiation: trees are **not**
re-evaluated each round. Reservations change only advertised remaining quantity
and advertised slack, not the original local observation or GP terminals.
Specifically, `remaining = initial_quantity - reserved_out - reserved_in` and
`message_rho = observation_rho - reserved_out + reserved_in`.
Reservations reset at the next period; failed promises are not reallocated.

**No privacy guarantee:** slack, costs, messages, and reservation summaries can
reveal operational information. There is no encryption, differential privacy,
strategic misreporting model, incentive compatibility, authentication, or security
isolation from arbitrary Python introspection. Central experiment traces contain
states and actual demands for auditing and must not be mistaken for private logs.

## Mathematical contract (manuscript alignment)

All actions and stocks are continuous quantities; Poisson demands remain integer.
At the **start** of period t, site i observes on-hand `I_i >= 0`, all outstanding
orders `P_i` (length `LT`, slot 0 due now), exposed mean forecasts `f_i[0:L]`,
storage capacity `C_i`, local costs and own execution memory. Define

```
F_i   = sum(f_i)
rho_i = I_i + sum(P_i) - F_i
Phi_i(rho) = h_i * max(rho, 0) + b_i * max(-rho, 0).
```

The order selected this period is not outstanding yet and is **excluded** from
this period's role/slack calculation. Roles are fixed for the entire period:
donor if `rho > eps`, receiver if `rho < -eps`, idle otherwise;
`eps = 1e-8`. No same-period receive-and-forward is possible.
With the raw three tree outputs `z_order, z_request, z_offer`, skills apply:

```
order   = min(C, max(0, z_order))
ROOM    = max(0, C - I)
DEFICIT = max(0, -rho)
AVAIL   = min(C, I + P[0])
SURPLUS = max(0, rho)
request = min(max(0, z_request), DEFICIT, ROOM)  [receiver only, else 0]
offer   = min(max(0, z_offer), SURPLUS, AVAIL)   [donor only, else 0].
```

Order capacity equals the benchmark's final **per-order storage-capacity cap**;
it is not `C-I-sum(P)` and can cause arrival overflow/discard. This intentionally
preserves benchmark order semantics. `production_capacity` remains unused and is
neither a physical cap nor a terminal. Request storage headroom is conservatively
measured before demand, and offer availability includes only on-hand plus due-now
arrivals, not later pipeline or the new order. Demand can invalidate any offer:
these are conditional promises, not guarantees.

For a donor i and receiver j, using only their messages, evaluate:

```
A_i(q) = Phi_i(rho_i-q) - Phi_i(rho_i)
B_j(q) = Phi_j(rho_j) - Phi_j(rho_j+q)
G_ij(q) = B_j(q) - A_i(q) - (c*q + K*1(q>0)).
```

This is a **deterministic forecast-slack heuristic**, not a calibrated expected
inventory cost, demand risk bound, service guarantee, or optimality result.
Donor `A` can be negative when sending reduces its surplus holding proxy.
Reservations reduce advertised donor slack by outbound promises and increase
receiver slack by inbound promises.

### Fixed decentralized matching

In each of at most `R=3` rounds (configurable):

1. Deliver each site's remaining quantity, reservation summary and updated slack
   to its topology neighbors.
2. Each endpoint independently considers its own message with each received
   opposite-role message. For each pair let
   `limit = min(remaining_offer, remaining_request)`. Candidate quantities are
   `{0,.25,.5,.75,1} * limit`, rounded to **6 decimal places**, clipped to
   `[0,limit]` and de-duplicated within `eps=1e-8`. Quantities `<= eps` are ignored.
   The implementation is `min(limit, round(fraction*limit, 6))`, in ascending
   fraction order, keeping the first candidate in each tolerance neighborhood.
   Python's binary floating-point `round` uses ties-to-even. Clipping after
   rounding can restore an unrounded upper bound; it is not an integer lattice.
3. Retain only candidates with `G > eps`. Both endpoints use the **same** ranking:
   descending gain, ascending donor ID, ascending receiver ID, descending quantity.
   Floating gains are compared exactly (no approximate equality comparator);
   tolerance is used only for quantity/positive-gain eligibility.
4. Each site chooses its best incident edge. Only **mutual identical choices**
   confirm. Each site therefore participates in at most one match this round.
   There is no central globally-best-edge or maximum-weight matching step.
5. Both local reservation ledgers debit the same planned quantity; subsequent
   rounds advertise the remaining quantities and updated slack. Stop early if
   no pair confirms. Roles do not change.

The simulator orchestrates mail delivery and checks mutuality; it does not choose
partners on behalf of sites. Finite rounds and topology may leave positive-gain
opportunities unmatched; this is not described as globally optimal matching.

An important consequence of the chosen surplus/deficit caps: every eligible q
stays on the same linear branches of both proxies, so (up to floating arithmetic)
`A=-h_donor*q`, `B=b_receiver*q`, and
`G=(h_donor+b_receiver-c)*q-K`. With nonnegative costs, a positive-gain pair
therefore prefers its largest available grid candidate. The five-fraction
enumeration is retained exactly for the fixed evaluator contract, not claimed
to discover an interior nonlinear risk optimum. Tree quantities and successive
reservations still change edge limits and partner rankings.

### Timing and actual execution

1. Observe local pre-arrival/pre-demand state; select orders and negotiate all
   conditional promises. **No actual current demand is read during these stages.**
2. Environment applies due arrivals: `I = min(C, I + P[0])`, counting any discarded
   overflow. Subtract current demand, temporarily permitting a signed unmet-demand
   deficit, as in the baseline's post-demand transshipment accounting.
3. Execute approved promises in **(round, donor ID, receiver ID)** order. Recheck
   donor free stock `max(0,I_d)` and receiver room `max(0,C_r-I_r)`:
   `q_actual = min(q_plan, donor_free, receiver_room)` (values `<= eps` become 0).
   Debit shared physical ledgers and update inventories after each execution.
   A receiver can clear its unmet current demand before storing up to C; transfers
   cannot make a donor negative. No re-matching/re-optimization occurs.
4. Charge `c*q_actual + K` only when `q_actual > 0`. Then account for lost sales
   `max(0,-I)` and reset negative stock to zero; charge holding on nonnegative stock.
   Charge order cost `unit_order*order + fixed_order*1(order>0)`.
5. Shift pipeline and append the new order, advance forecasts and period.
   **An order chosen in t arrives in t+LT**, including LT=1. This reproduces the
   original simulator's previous-action staging while exposing its otherwise
   omitted most recent outstanding order. No phantom/double-counted pipeline.

Execution priority is reproducible but can favor earlier rounds/lower IDs; it is
**not** claimed unbiased. Ledger accounting distinguishes approved versus actual
transfers: `approved_count/qty`, `executed_count/qty`, zero-execution
`cancelled_count`, total unfulfilled `cancelled_qty` (including partial shortfalls),
`partial_count`, partial executed `partial_qty`, and `partial_shortfall_qty`.
Thus `approved_qty = executed_qty + cancelled_qty`.
Order quantities and initial request/offer budgets are **not rounded**. Actual
executions are **not rounded** beyond treating quantities `<= eps` as zero.
An execution is classified partial only if `actual < planned-eps`; a smaller
positive shortfall can therefore be included in `cancelled_qty` for a record
classified full. All unexecuted quantities are retained numerically.

## Terminals, arithmetic, and search

Exact terminal dictionaries (no site IDs or other-site values):

| Module | Terminal names |
|---|---|
| All three | `I, PIPE, DUE, F, FC1, FC2, CAP, H, B, RHO, LAST_IN, LAST_OUT, LAST_LOST` |
| Order only | `ORDER_FIXED, ORDER_UNIT` |
| Request only | `ROOM, DEFICIT, TC, TK` |
| Offer only | `AVAIL, SURPLUS, TC, TK` |

`PIPE=sum(P)`, `DUE=P[0]`, `FC1=f[0]`, `FC2=f[1]` (0 if absent);
`H=h`, `B=b`, `TC=c`, `TK=K`. All L exposed means enter F, but only the first two
have individual terminals. All tree inputs and primitive results are saturated
to `[-1e6,1e6]`. Primitives are binary `add, sub, mul, div, maximum, minimum`;
`div(a,b)=1` when `abs(b)<=1e-8`, else saturated `a/b`.
Initializer/mutator constants are `{0,.25,.5,1,2}`; the validated loader also
accepts nonnegative finite constants up to 1e6. Negative values can be represented
using subtraction. NaN/nonfinite inputs or outputs raise `NumericalPolicyError`;
there is **no silent action fallback**. Successful metrics report zero numerical
failures; a failed evaluation aborts with an exception rather than fabricated cost.
The +/-1e6 saturation belongs only to GP arithmetic/terminal inputs. Physical
stocks, forecasts, slack calculations, skill caps, messages and proxy gains use
their original finite values; they are not normalized or saturated by that bound.

Default search (`train`): population 200, 50 replacement generations plus generation
0, tournament size 3 with replacement, 1 elite, mutually exclusive crossover .8,
mutation .15, reproduction .05. Formal study runs should use **at least 200
individuals and 50 replacement generations**; larger budgets can be configured.
Smaller explicit budgets remain available for debugging and tests.
`smoke` deliberately retains population 6 and 1 generation for functional
verification only; its output is not a formal study result.
Random trees use DEAP half-and-half depths 0–2. Mutation replaces one subtree
with grow depth 0–2. Crossover exchanges subtrees of **one homologous module**,
never order↔request↔offer. A one-node tree cannot cross under DEAP's operator.
Children exceeding height 5 or 63 nodes per tree revert to their corresponding
parent; maximum individual size is 189 nodes. No adaptive hyperparameters.

One initial seed template is
`(maximum(0.0, sub(F, add(I, PIPE))), DEFICIT, SURPLUS)`; all remaining individuals
are random. GP evaluates centralized complete simulator episodes,
not distributed learning. Ranking minimizes the lexicographic tuple
`(worst episode service violation, mean period total cost, total nodes)`,
where episode violation is the **sum of site-level fill shortfalls** and the
service target defaults to .95 for **every site**. This conservative
worst-episode constraint ranking is not a chance constraint or out-of-sample
service guarantee.

For each seed/episode e, cost `C_e` is summed over **all sites and all elapsed
periods**, then `cost_e=C_e/T_e` (no division by number of sites). Let `D_ei` be
site i's demand over the episode and `L_ei` its lost units **after actual
transfers**, with all per-period losses reset correctly. The exact definition is:

```
fill_ei = 1 - L_ei/D_ei                       if D_ei > 0, else 1
v_ei    = max(0, target - fill_ei)
v_e     = sum_i v_ei
V       = max_e v_e
fitness = (V, (1/M)*sum_e C_e/T_e, total_tree_nodes)  # lexicographic minimum
```

There is **no averaging over sites**, no demand weighting of site shortfalls,
and no replacement by a single worst site or mean episode violation. `V` may
exceed 1 when several sites violate the target; it is not a probability.
A large well-served site's demand cannot hide a neglected site. In particular,
site demands `[100,1]` with losses `[0,1]` have network fill `100/101 > .95`
but episode violation `.95`, not zero. A zero-demand site has fill 1 and
violation zero.

Each episode's `site_metrics` lists `{site, demand_qty, lost_qty, fill_rate,
service_violation}`. Episode `service_violation` is their sum. Batch
`worst_service_violation` is its maximum over episodes. Batch
`mean_site_fill_rates` is the equal-episode-weight mean for each site, in site-ID
order. The separate diagnostic `network_fill_rate` is `1-sum_i L_ei/sum_i D_ei`
(1 for zero network demand); batch `mean_network_fill_rate` averages these
episode diagnostics, **not** pooled units over episodes. Network fill is never
a service constraint. Ambiguous old top-level `fill_rate`/`mean_fill_rate`
fields have been removed. Exact fitness ties use the tuple of canonical tree
strings as a deterministic final ordering, not site IDs.

Every competing individual and elite uses the **same fixed scenario/seed batch**;
there is no rotation or stale cross-batch HallOfFame. Identical trees cache their
fixed-batch fitness. Unique generation champions are compared on one distinct
validation batch; its lexicographic winner is selected. Only then is that policy
scored on held-out test seeds. Training, validation, and test seeds must be
nonempty, unique within each split, and mutually disjoint. Defaults:
train `11,12`, validation `1001,1002`, test `2001,2002`. Reusing test outcomes to
tune settings would invalidate the held-out interpretation.

## Demand provenance and fairness caveats

The adapter reads `Utils.ScenarioDesign_replenish_transship` and
`Utils.inventory_core.build_demand_model` without modifying either. The baseline
constructs a generator and resets once before `run`, so its first episode uses
**`provided seed + 1000`**. This implementation generates directly at that
effective seed, with exactly the same NumPy global-RNG uniform forecast and
Poisson draw order/array sizes, then restores the caller's RNG state.
It does not switch to `default_rng`. Tests compare exact arrays against the
baseline demand generator's reset. The baseline synthetic episode has 65 demand
records, of which 64 are simulated; the final record is not exposed as an
actual. Default horizon remains 64; cost divides by **elapsed simulated periods**,
not a nominal horizon when running a prefix.

The scenario suffix `b` is a lost-sales multiplier, **not LT**; shared synthetic
scenarios set `LT=L=2`. The generic environment supports N>=2, arbitrary valid
undirected topology, configurable positive integer LT/L via the Python API,
subject to sufficient forecast samples. Shared synthetic utilities only generate
`epi_len+3` means; insufficient forecast coverage is rejected rather than padded.

**Historical data warning:** shared `demand_test` (including `teckwah_test`)
constructs forecasts directly from realized future demand. This is **not causal**.
It is rejected by default. `--allow-oracle-forecasts` explicitly opts in and is
recorded in config/metrics; sufficient forecast records must still exist (use a
shorter `--periods` if needed). No silent zero/last-value padding is performed.
No causal real-data forecast model is supplied. Current/future realized synthetic
demands remain environment-private; exposed synthetic forecasts are Poisson means.

The original 400×50 algorithm, joint-state terminals, logistic action map, incomplete
pipeline observation, cost-only selection, and stale cross-batch HallOfFame are
**not** reproduced. Representation, protocol, information boundary and search all
differ; this is an independent method implementation, not a single-factor empirical
comparison. No full training study or superiority/service-risk claim is implied.

## Artifacts and validation scope

Training writes:

- `config.json`: all settings, search hyperparameters, three seed batches,
  per-seed scenario parameters, Python/dependency versions; seed rotation is false.
  Schema is `peer-experiment-v2`, with service-constraint marker
  `max_episode_sum_site_fill_shortfall_v1`.
- `best_policy.json`: strict canonical JSON, fixed module names, exact terminal/
  arithmetic/bloat schema. Loading validates a restricted expression AST before
  DEAP parsing; no pickle or arbitrary Python expressions.
- `generation_NNN_policy.json`, `history.json`, `training_evaluations.json`,
  `validation_candidates.json`: generation champions, all unique training
  evaluations, and validation candidate selection.
- `metrics.json`: separate training/validation/held-out test summaries plus each
  episode's costs, per-site demand/lost/fill/violation, sum-of-site violation,
  explicitly labeled network diagnostic, elapsed periods, seed/stream seed and
  transfer ledger. The service-constraint marker is included in summaries and
  episodes.
- `traces_train.json`, `traces_validation.json`, `traces_test.json`: observations,
  orders, per-round messages/choices/confirmations, arrivals, current demands,
  planned/executed transfers, losses and ending pipelines/stocks. `--no-trace`
  suppresses these bulky files.

Evaluation writes `config.json`, `policy.json`, `metrics.json`, and `traces.json`.
Outputs contain no wall-clock timestamps; repeated seeded runs with identical
dependencies produce byte-identical policy/config/metrics/traces in the tests.
Runtime CLI errors are explicit; a partially written output directory should be
kept for diagnosis or replaced with a new empty directory for retry.

**Objective-version boundary:** earlier `peer-experiment-v1` output directories
used network-aggregate service and are obsolete evidence for the site-level
constraint. They are retained, not overwritten. CLI `--config` rejects those
configs (and mismatched service markers) instead of silently replaying them
under a different objective. Canonical policy JSON remains
`three-tree-peer-v1` because representation/primitives/caps are unchanged; an old
policy can be explicitly re-scored with current scenario/settings without its old
experiment config. No old fitness or selection outcome is carried into a new run.

Focused tests cover competing receivers and reservation budgets, mutual ties,
topology restrictions, generic five sites, role/storage/availability/order caps,
partial/cancelled execution and executed-only memory, post-demand loss reset,
pre-demand observation independence from actual future streams, exact demand
stream provenance, LT/horizon/denominator, historical oracle opt-in, three-tree
operators/bloat, canonical loading/rejection, seed repeatability, disjoint splits,
constraint-first ranking and training→reload→CLI evaluation equality.

The 27-test suite additionally verifies all 64 steps of the
unmodified `sN2h_1_5b2` scenario for **each** default training, validation and
held-out seed. It compares demands/forecasts against the shared generator after
its original reset and compares varying-order arrivals, complete pipelines,
inventory, lost quantities, capacity discard, costs and final outstanding orders
against an independent reference of the old LT-1-pipeline/previous-action staging.
The same new continuous order sequence is supplied to both paths, with transfers
disabled to isolate timing; this does not assert equality with the old GP action
mapping or a different transfer policy. Tests also check frozen per-period GP
features versus updated messages, unpooled episode aggregation, and that the
chosen policy is saved before any held-out simulation occurs.
Site-service regressions cover network fill above target with a neglected site,
zero-demand sites and an all-zero-demand episode, and two episodes where the
maximum **sum** of site shortfalls differs from their mean, the worst single
site, and the sum of per-site maxima across episodes. Config compatibility tests
prevent accidental reuse of old network-service experiment evidence.
The study-budget regression verifies matching Python-API and CLI defaults of
200 individuals / 50 generations, while retaining the 6/1 smoke configuration.
Checking these defaults does not imply that a full study run has completed.

### Fresh site-service validation results

The earlier site-level validation recorded **26 tests passed**, `compileall`
passed, and a new 64-period smoke plus canonical-policy reload completed.
Artifacts are in:

```
MTGP_Agent_Skills/outputs/site_service_seed0/
MTGP_Agent_Skills/outputs/site_service_replay_seed0/
```

Both configs use `peer-experiment-v2` and
`max_episode_sum_site_fill_shortfall_v1`. Scenario `sN2h_1_5b2`, search seed 0,
population 6, one replacement generation plus generation 0, R=3, complete
topology, target .95, all 64 periods; train seeds 11,12, validation 1001,1002,
held-out 2001,2002. Demand stream seeds remain supplied seeds +1000.

| Split | V (site constraint) | Mean cost/period | Mean site 0 fill | Mean site 1 fill | Mean network fill (diagnostic) |
|---|---:|---:|---:|---:|---:|
| Training | 0 | 675.925748 | 0.970149 | 0.973188 | 0.971677 |
| Validation | 0 | 711.019948 | 0.978784 | 0.963613 | 0.970720 |
| Held-out | 0 | 718.085373 | 0.983138 | 0.967600 | 0.974859 |

The selected three-tree policy has 11 total nodes:

```
order:   maximum(maximum(FC1, RHO), DUE)
request: maximum(F, LAST_OUT)
offer:   maximum(1.0, CAP)
```

Every individual held-out episode/site meets .95 in this small batch, not just
the site means. Held-out site fills are `[0.988811800, 0.950980392]` for seed
2001 and `[0.977464789, 0.984218952]` for seed 2002. Per-site demand, lost units
and violations are present in `metrics.json`; actual losses are after transfers.
Reload reproduced the held-out **metrics and complete traces exactly**.
This freshly recomputed run happens to select the same trees and cost/network
diagnostics as an earlier network-service smoke; its site-level objective and
evidence were recomputed, not inherited from those obsolete artifacts.
These are smoke checks, not a full study, superiority result, or general service
guarantee. Conservative caps and their explicitly linear proxy gain are unchanged.
