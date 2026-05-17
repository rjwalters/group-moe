# Paper 2 Experiment Log

Running ledger of every experiment, its purpose, status, and result. Drives the paper's evidence base — each row should map to a sentence (or a figure) in the manuscript. Updated as experiments complete.

## Status legend

- ✅ **done** — result in hand, summarized below
- 🟢 **running** — cluster job in flight; expected completion noted
- 📋 **queued** — staged; will auto-launch
- 💡 **planned** — discussed, not staged
- ❌ **abandoned** — tried and dropped (with reason)

## Negative result (Paper Section: "When Does Selective Equivariance Fail?")

Establishes that Group-MoE doesn't help on tasks where it shouldn't — provides the contrast against MD17.

| # | Experiment | Status | Result | File |
|---|---|---|---|---|
| 1 | SchNet baseline on QM9 U0 | ✅ | 16.7 meV test MAE | `data/qm9/schnet_baseline_v5/` |
| 2 | SchNet+GroupMoE v1 on QM9 | ✅ | 657 meV val plateau (terminated e133) | killed; v1 notes only |
| 3 | SchNet+GroupMoE v2a (lr=1e-5) on QM9 | ✅ | ~92 meV val; MoE active | `data/qm9/v2a/` |
| 4 | SchNet+GroupMoE v2b (lb_weight=0) on QM9 | ✅ | ~29 meV val; router collapsed (pt=0.99) — effectively pure SchNet | `data/qm9/v2b/` |
| 5 | SchNet+GroupMoE v2c (norm reducer) on QM9 | ✅ | 186 meV val (terminated) — worse than v2a | `data/qm9/v2c/` |
| 6 | ViSNet baseline on QM9 U0 | ✅ | 8.7 meV val (reference for "full equivariance") | `data/qm9/visnet_baseline_v4/` |
| 7 | Per-atom ‖v‖ CV on ViSNet | ✅ | CV=0.18 (uniform); predicts MoE-in-ViSNet won't gain on QM9 | `scripts/analyze_visnet_v4.py` |

**Section narrative:** On QM9 (invariant scalar output, sp3-dominated organic molecules), every Group-MoE variant we tested is either (a) net-negative when the MoE is active, or (b) effectively bypassed when the router collapses. Diagnosis: scalar bottleneck on an invariant-output task means the MoE block has no irreducible contribution to make. The ViSNet ‖v‖ CV analysis suggests this generalizes beyond SchNet — even fully-equivariant backbones do per-atom equivariant work roughly uniformly on QM9.

## Positive result (Paper Section: "Selective Equivariance for Force Prediction")

Establishes Group-MoE works on MD17 force prediction with the same architecture that failed on QM9.

| # | Experiment | Status | Result | File |
|---|---|---|---|---|
| 8 | SchNet baseline, ethanol (seed=0) | ✅ | F=0.316, E=0.071 kcal/mol{/Å} | `data/md17/results/schnet_md17_ethanol_alc2/` |
| 9 | SchNet+GroupMoE, ethanol (seed=0) | ✅ | F=**0.272** (−14%), E=0.071 (tie) | `data/md17/results/groupmoe_md17_ethanol_alc8/` |
| 10 | SchNet baseline, aspirin (seed=0) | ✅ | F=0.932, E=0.344 | `data/md17/results/schnet_md17_aspirin_alc2/` |
| 11 | SchNet+GroupMoE, aspirin (seed=0) | ✅ | F=**0.727** (−22%), E=0.247 (−28%) | `data/md17/results/groupmoe_md17_aspirin_alc6/` |
| 12 | SchNet baseline, ethanol seeds 1, 2 | 🟢 | n=3 variance bars; ~done by 2026-05-16 evening | `data/md17/results/schnet_md17_ethanol_alc2_s{1,2}/` |
| 13 | SchNet baseline, aspirin seeds 1, 2 | 🟢 | n=3 variance bars; ~done by 2026-05-16 evening | `data/md17/results/schnet_md17_aspirin_alc2_s{1,2}/` |
| 14 | SchNet+GroupMoE, ethanol seeds 1, 2 | 🟢 | n=3 variance bars; ~done by 2026-05-16 evening | `data/md17/results/groupmoe_md17_ethanol_alc8_s{1,2}/` |
| 15 | SchNet+GroupMoE, aspirin seeds 1, 2 | 🟢 | n=3 variance bars; ~done by 2026-05-16 evening | `data/md17/results/groupmoe_md17_aspirin_alc6_s{1,2}/` |

**Section narrative (writable now):** Identical model + recipe + train/val/test split. Only architectural difference is the inserted MoE block at moe_position=2/4 in SchNet. Force MAE drops 14% on ethanol and 22% on aspirin. Energy MAE drops 28% on aspirin (ties on ethanol). The larger gain on aspirin (21 atoms, more chemical heterogeneity) supports the "router-needs-signal" hypothesis.

## Breadth (Paper Section: "Across the MD17 Suite")

Testing whether the result is specific to ethanol/aspirin or transfers across the standard benchmark.

| # | Experiment | Status | Result | File |
|---|---|---|---|---|
| 16 | SchNet + GroupMoE, uracil (heterocycle, N+O) | 📋 | wave 2 — auto-launches after wave 1 | — |
| 17 | SchNet + GroupMoE, naphthalene (pure aromatic) | 📋 | wave 2 | — |
| 18 | SchNet + GroupMoE, toluene (aromatic + methyl) | 📋 | wave 2 | — |
| 19 | SchNet + GroupMoE, benzene (smallest aromatic) | 📋 | wave 3 — auto-launches after wave 2 | — |
| 20 | SchNet + GroupMoE, malonaldehyde (small, polar) | 📋 | wave 3 | — |
| 21 | SchNet + GroupMoE, salicylic acid (aspirin's parent) | 📋 | wave 3 | — |

**Section target:** "Group-MoE consistently improves SchNet across the 8-molecule MD17 benchmark. Improvements range from X% to Y%, with the strongest gains on chemically heterogeneous molecules (aspirin, salicylic acid) and weakest on simple aromatics (benzene)." (Hypothesis — measure once results land.)

## Mechanism (Paper Section: "What Does the Router Learn?")

The chemistry-detection story. This is the interpretability figure.

| # | Experiment | Status | Result | File |
|---|---|---|---|---|
| 22 | Per-atom routing analysis on aspirin checkpoint | ✅ | Position-specific routing >90% modal on most atoms; chemically equivalent atoms route identically | `scripts/analyze_md17_routing.py`, `routing_aspirin.png` |
| 23 | Per-atom routing analysis on ethanol checkpoint | ✅ | 3+2+1 H split (methyl/methylene/hydroxyl) recovered without supervision | `routing_ethanol.png` |
| 24 | Chemistry labeling via bond topology | ✅ | Automated per-position chemical-role assignment | `scripts/identify_md17_chemistry.py` |
| 25 | Cross-molecule transfer: aspirin-trained on ethanol | ✅ | Equivalence-class structure transfers (methyl Hs grouped, methylene Hs grouped, Cs grouped); specific labels don't (different bijection chemistry→expert per training run) | `routing_on_ethanol.json` |
| 26 | Cross-molecule transfer: ethanol-trained on aspirin | ✅ | Asymmetric: ethanol→aspirin **collapses** to default expert (octahedral) for chemistries ethanol training didn't cover (sp² C, =O, etc.). The methyl-H rule transfers because both molecules have methyl Hs. | `routing_on_aspirin.json` |

**Section narrative (figure-driven):** Stacked-bar routing distribution per atom position, with chemistry labels. On ethanol, the 3 methyl H's, 2 methylene H's, and OH H form three visually distinct routing groups — the model unsupervisedly recovered the 3+2+1 chemical equivalence classes. On aspirin, the same pattern at finer grain: 4 ring-CH atoms, 2 ipso ring C's, 2 carboxyl C's, 3 distinct O species, 3 methyl H's all form internally consistent routing groups. Caveat: expert names (tetrahedral/octahedral/planar) are arbitrary embeddings the model learned; the assertion is about *consistency*, not literal point-group identification.

**Cross-molecule findings (experiments 25, 26):** The router learns chemistry as **equivalence relations** (which atoms are interchangeable), not as fixed labels. (a) When the source training chemistry covers the target (aspirin → ethanol), the *partition* transfers — methyl Hs group together, methylene Hs group together, methyl-H → pass-through transfers cleanly. The expert *labels* differ between runs (each training run picks a different bijection chemistry → expert). (b) When the target has chemistry the source training didn't cover (ethanol → aspirin), the model collapses everything it can't discriminate to a default expert. Working theory: the model learns the equivalence relations its training data forced it to learn. Implication for scaling to proteins: training data must span the chemistries seen at inference.

## Ablations and controls (Paper Section: "What's Doing the Work?")

| # | Experiment | Status | Purpose | File |
|---|---|---|---|---|
| 27 | Random-router control on aspirin | 💡 | Distinguish "routing matters" from "extra params matter". If gain persists with randomized router → params; if gain vanishes → routing | — |
| 28 | Single-expert (no routing) on aspirin | 💡 | Tests pass-through-vs-expert binary alone, no expert specialization | — |
| 29 | Frozen-router (post-warmup) on aspirin | 💡 | Tests whether ongoing routing updates matter vs early specialization | — |

**Status:** Discussed but not staged. Random-router (#27) is the highest-leverage causal-attribution experiment. Requires a code change (training script flag to randomize router argmax) before queueing.

## Scale (Paper Section: "Beyond MD17")

| # | Experiment | Status | Purpose | File |
|---|---|---|---|---|
| 30 | MD22 (42–370 atom biomolecules) | 💡 | Tests whether the result holds at scale where chemical diversity is much higher | — |
| 31 | Per-residue routing on a protein dataset | 💡 | The AlphaFold-direction experiment. Big infra burden, big payoff if it works. | — |

**Status:** Planned but deferred until MD17 + ablations are nailed down.

## Infrastructure / negative results

| # | Experiment | Status | Result | File |
|---|---|---|---|---|
| 32 | Mac MPS feasibility (50ep ethanol GroupMoE probe) | ✅ | Works but autograd-graph leak grows wall time 49s→700+s/epoch; not viable for 1000ep runs | `data/md17/results/probe_mps_ethanol_50ep/` |
| 33 | alc-cluster setup (alc-2/4/6/8) | ✅ | RTX 4090 nodes via Tailscale; setup recipe in `reference_alc_cluster.md`; alc-4 lost mid-experiment (Tailscale stale); alc-8 brought online to replace | — |

## Open questions for paper

- **Robustness of routing patterns across seeds.** Does each seed pick a different mapping (chemistry → expert) but the *groupings* (which atoms share an expert) remain stable? Tests once wave 1 finishes.
- **Cross-distribution transfer.** Routing on the same molecule the model trained on doesn't tell us about generalization. Cross-molecule routing (#25, #26) addresses this.
- **What does "tetrahedral" mean to the model?** The labels are arbitrary post-hoc names. A version where experts are constrained to literal irreducible representations of named point groups would be a different (potentially weaker) model — worth discussing in related work.

## Compute spend (running tally)

| Source | Spent | Per |
|---|---|---|
| Lambda Labs (Paper 2 prep, QM9 work) | ~$150 | $7 SchNet baseline, $50+ ViSNet baseline, $40 GroupMoE v2 sweep, smaller |
| alc-cluster (MD17 wave 1+2+3) | $0 (in-house) | — |
| Mac MPS (probes only) | $0 (electricity) | — |

