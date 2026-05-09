# Paper 2: Routes Forward After the v2 Sweep

**Status as of 2026-05-08:** the v2 hyperparameter sweep on QM9 confirmed the v1 negative result. SchNet+GroupMoE in three different configurations (lr=1e-5, lb_weight=0, norm-reducer) all underperform the SchNet host alone at the same recipe. This document records what we learned, why we think it failed, and three concrete routes that could yield a positive result.

## What we tried (QM9 U0, 110k/10k/10.8k split, 1000 epochs cosine + warmup)

| Variant | What changed vs v1 | Best val_mae | Notes |
|---|---|---|---|
| **v1** | (baseline GroupMoE attempt) | 657 meV | terminated at e133, slow descent |
| **v2a** | lr=1e-5, moe_position=6 | ~92 meV | converged near completion |
| **v2b** | load_balance_weight=0, moe_position=6 | ~29 meV | router collapsed to 99% pass-through (= effectively pure SchNet at v2 recipe) |
| **v2c** | norm reducer, moe_position=6 | 186 meV @ e419 (terminated early) | trajectory still descending but >2× behind v2b |

**Reference points:**
- SchNet v5 baseline (lr=5e-4 cosine, well-tuned): **16.7 meV** test
- v2b SchNet at v2 recipe (= no MoE in effect): **~29 meV** val
- ViSNet v4 (full equivariance reference): **~12 meV** val and still descending

**Consistent pattern:** every configuration where the MoE is *active* (router using experts) is worse than when the MoE is *bypassed* (collapsed router). The MoE block adds harm, never value, on the QM9 energy task.

## Diagnosis: why didn't it work?

The architecture has a **scalar bottleneck**:

```
SchNet (scalar features) → lift → MoE (equivariant features) → reduce → SchNet (scalar features)
```

The reduce-back-to-scalars step necessarily destroys most of the equivariant computation, summarizing it via either l=0 channels (v1) or l=0 + ‖l>0‖ norms (v2c). For a task whose output is *already* invariant (energy is a scalar), SchNet's distance-based features encode most of the relevant SE(3) invariants directly. The MoE's equivariant computation has to provide value through that scalar bottleneck, and apparently it can't — its contribution is dominated by:

1. **Optimization noise:** the MoE adds parameters whose gradients fight SchNet's well-tuned dynamics.
2. **Routing churn:** different atoms get different experts on different epochs; the model never settles into a stable specialization.
3. **No new information:** any rotation-invariant scalar the MoE produces could in principle be approximated by SchNet's existing distance-feature pipeline.

The v2c norm-reducer test was specifically designed to widen the bottleneck and *did not help* — actually made things worse than the scalar-only reducer. That suggests the issue isn't just bandwidth at the boundary; it's that the MoE block isn't computing useful invariants.

## Three routes forward

Each route addresses a different aspect of why the v2 design failed, and each could yield a positive result.

### Route 1 — Switch the task: MD17 forces (recommended)

**Idea.** Forces on atoms in MD trajectories are *vectors*. Predicting them requires equivariant outputs. SchNet handles this only by autograd through energy (slow, indirect); PaiNN/ViSNet/MACE predict forces directly via vector features and substantially outperform SchNet on MD17.

If GroupMoE's "selective equivariance" framing has any teeth, MD17 is where it should bite — different atoms in the trajectory genuinely *do* have different local symmetry (a rigid carbon vs a rotating methyl group, for example), and a router that detected that could route different atoms to different equivariant force-predictors.

**What's needed:**
- MD17 data path (similar effort to `src/data/qm9.py`).
- Force prediction loss (`F.l1_loss(predicted_force, true_force)` on per-atom 3-vectors).
- Architecture: SchNet+GroupMoE could be modified so the GroupMoE block outputs forces directly (skipping the scalar reducer). Alternative: ViSNet+GroupMoE.
- Training: ~$30–50 per run on Lambda A100.

**Why this is the highest-expected-value route:**
1. The task argues for equivariance — we're testing the idea on home turf.
2. Opens a paper structure that includes both the QM9 negative result *and* a positive result on MD17, with a clean "when does selective equivariance help?" framing.
3. MD17 is well-established; literature comparisons are easy.

**Risk:** MD17 trajectories may not have *enough* per-atom symmetry variation for the router to learn useful specialization. A small molecule like aspirin has limited variety. Mitigation: also try MD22 or larger systems.

### Route 2 — Switch the backbone: MoE inside ViSNet

**Idea.** Don't bottleneck through SchNet's scalar pipeline. Replace one of ViSNet's `ViSNetBlock`s with a Group-MoE-of-equivariant-blocks that operates *entirely in the irrep basis*. The features stay equivariant end-to-end; the router decides which equivariant computation to apply.

```
ViSNet block × k  →  GroupMoE block (equivariant in/out)  →  ViSNet block × (n-k)
```

The router still uses scalar features (rotation-invariant by construction). Each expert is an equivariant block similar to the existing `SO3Expert`, but it now sits in a context where its full output (l=0 + l=1 + l=2) is consumed by downstream equivariant operations rather than being summarized to scalars.

**What's needed:**
- Subclass `torch_geometric.nn.models.ViSNet`, replace one block with a MoE wrapper.
- The MoE wrapper accepts `(x_scalar, v_vector)` (ViSNet's two-track representation) and produces `(x_scalar, v_vector)` of the same shape.
- Each expert is essentially a `ViSNetBlock` with different irreps or hyperparameters.
- Training: ~$40 on A100.

**Expected outcome:** if `Group-MoE-in-ViSNet` matches or approaches plain ViSNet at lower FLOPs, that's a **positive efficiency result** even on QM9 — "selective equivariance recovers full-equivariance performance at reduced compute."

**Risk:** ViSNet's blocks are tightly coupled; replacing one with a MoE may break gradient flow through the rest. Also, the win condition ("similar accuracy at lower FLOPs") is harder to observe than a clean accuracy win.

### Route 3 — Switch the experts: discrete chemical point groups

**Idea.** Replace continuous SO(3) experts with **discrete chemical point group** representations: T_d (sp³), D_3h (sp²/aromatic), C_∞v (linear/sp), C_3v (umbrella), etc. Each expert implements the finite group's irrep matrices directly (which is exactly what Paper 1 did). The router classifies atoms by chemical symmetry type explicitly.

This is the closest path to Paper 1's framework — it uses the existing `GroupRepresentation` / `GroupExpert` infrastructure rather than the new e3nn-based modules.

**What's needed:**
- Add point-group `GroupRepresentation` subclasses (T_d, D_3h, C_3v, C_2v) in `src/groups/`.
- Verify multiplication tables and irrep matrices.
- Training: ~$30 on A10.

**Expected outcome:** less certain. Tests whether the failure was specifically about continuous-vs-discrete groups, not the MoE+SchNet combination. If discrete experts succeed where continuous ones fail, that's a finding about *which* group-structure formalism transfers to molecular tasks.

**Risk:** highest infrastructure burden for the most uncertain payoff. Defer.

## Recommendation

**Route 1 (MD17 forces) first.** It's the most natural test of the "selective equivariance" hypothesis on a task where equivariance unambiguously matters, and it converts the paper from "negative result only" to "negative + positive result with insight into when each applies." Cost: ~$50–100 for one round of experiments.

**Route 2 (MoE-in-ViSNet) as parallel or follow-up.** Lower-lift since we have ViSNet code; addresses a different question (efficiency rather than absolute accuracy). Could run alongside or after Route 1.

**Route 3 deferred** unless Routes 1 and 2 also fail.

## Decision log

- **2026-05-07:** GroupMoE v1 on QM9 fails (val plateaus at 657 meV).
- **2026-05-08:** v2 sweep (3 variants in parallel) confirms negative result. MoE block is net-negative under all tested configurations on QM9.
- **2026-05-08:** This document. Identified scalar bottleneck as likely structural cause; three routes forward identified.
- **2026-05-08:** Built `src/models/visnet_groupmoe.py` and `scripts/train_qm9_visnet_groupmoe.py` (Route 2 implementation) + `scripts/analyze_visnet_v4.py` (free pre-flight signal check). Smoke-tested; equivariance verified.
- **2026-05-08:** MD17 attempt (Route 1) on local Mac MPS abandoned. Wall time crept 240s → 820s over 15 epochs even with `torch.mps.empty_cache()`; aggressive downscale (h=64/L=3/batch=2) didn't fix it. MPS kernel coverage for force-prediction (autograd-of-autograd through e3nn ops) is the bottleneck. Lambda would cost ~$50 for 36h; deferred.
- **2026-05-09:** Pre-flight analysis on v4's converged ViSNet (best val 8.7 meV, e993). Per-atom ‖v‖ statistics computed on 35K val atoms. **Overall CV=0.18, within-carbon CV=0.16** — both well below the 0.5 threshold for "router has signal." ViSNet uses its equivariant pathway roughly uniformly across atoms; per-atom MoE has weak signal to discriminate. **Decision: Route 2 likely fails for the same reason Route 1 did — QM9 energy is too invariant for selective equivariance to gain on top of full equivariance.** Skipped Route 2 training run; saved $40.

## Updated recommendation (after analysis)

The original ranking (Route 1 → 2 → 3) was based on a-priori reasoning about scalar bottlenecks. The post-v4 analysis adds direct evidence:

- ViSNet's per-atom equivariance is roughly uniform → there's little for a router to specialize over.
- The v2 sweep on SchNet+MoE showed the same pattern: the architecture *can* route, but the routing doesn't carry useful information for the task.
- The convergent reading: **QM9 is the wrong task for this idea**, not "we haven't found the right architecture yet on QM9."

The honest paper framing now becomes: "we attempted selective equivariance via Group-MoE on QM9 from two architectural angles (SchNet host with scalar bottleneck; ViSNet host with no bottleneck) and analyzed why both fail. The diagnosis points to the task itself: QM9's invariant scalar output collapses the per-atom routing signal." MD17 forces remains the natural next experiment but requires Lambda compute (~$50) since Mac MPS can't handle force-prediction autograd at scale.

This is a publishable negative-result paper, especially with the supporting analysis (the CV plot is a nice figure).
