# Literature Review — Group-MoE

## Positioning

Group-MoE sits at the intersection of three active research areas: geometric deep learning, mixture-of-experts architectures, and learned symmetry detection. It proposes a novel combination: using group representations in the irrep basis as expert modules with learned routing — making symmetry an architectural option rather than a mandatory constraint or an emergent property.

The closest related work is MatrixNet (NeurIPS 2024), which also applies group representations in neural networks but learns the representations from data rather than using fixed irreps. Group-MoE is complementary: we fix the irrep structure (guaranteeing exact composition) and learn when to apply it (via routing).

A naming collision exists with Kang et al. (2025) "Mixture of Group Experts," which uses "group" to mean expert grouping for diversity, not group-theoretic representations. Our work is conceptually distinct — we must clarify this in the paper.

## Key Related Work

### Geometric Deep Learning
- **Cohen & Welling (2016)**: Group equivariant convolutional networks (G-CNNs). Founded the field. Uses G-convolutions that share weights across group orbits. Key difference: symmetry is enforced in every layer. Group-MoE makes it optional via routing.
- **Bronstein et al. (2021)**: "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges." The blueprint paper. Unifies CNNs, GNNs, transformers under a common equivariance framework. Group-MoE uses the same irrep machinery but wraps it in an expert module.
- **Weiler & Cesa (2019)**: General E(2)-equivariant steerable CNNs. Steerable convolutions using irrep decomposition — the same mathematical tool Group-MoE uses for the expert transform.

### Mixture of Experts
- **Shazeer et al. (2017)**: Outrageously large neural networks (MoE). Original sparse MoE with gating. Experts are generic feedforward blocks.
- **Fedus, Zoph & Shazeer (2022)**: Switch Transformers. Simplified MoE with top-1 routing and load-balancing loss. Our routing and balance loss follow this design. Key difference: their experts are generic MLPs; ours are group representations.
- **Kang et al. (2025)**: "Mixture of Group Experts for Learning Invariant Representations." Uses group sparse regularization on routing inputs for expert diversity. Despite the name, this is about MoE expert diversity, NOT group-theoretic representations. Naming collision — must distinguish in our paper.

### Learned Group Representations
- **MatrixNet (Laird, Hsu, Bapat, R. Walters, NeurIPS 2024)**: Learns matrix representations of group elements from data. Key architecture: matrix block outputs invertible matrices for generators; compositions are defined by matrix products. Achieves compositional generalization to longer word lengths. Highly relevant — our complementary approach: we fix the representations (guaranteeing exact irrep structure) and learn routing; they learn representations but fix the application.
- **Dehmamy et al. (2021)**: Automatic symmetry discovery via Lie algebra. L-conv discovers continuous symmetries from data.
- **Benton et al. (2020)**: "Learning Invariances in Neural Networks." Meta-learn which invariances to enforce.

### Compositional Generalization & Permutation Equivariance
- **Gordon, Lopez-Paz, Baroni & Bouchacourt (ICLR 2020)**: "Permutation Equivariant Models for Compositional Generalization in Language." Hypothesizes language compositionality is group-equivariance. Tests on SCAN tasks. Directly relevant — they build equivariant seq2seq models; we provide group structure as optional expert modules.
- **Lake & Baroni (2018)**: Generalization without systematicity. Demonstrates that standard seq2seq fails at compositional generalization.

### Equivariance in Transformers
- **Fuchs et al. (2020)**: SE(3)-Transformers for molecular property prediction.
- **Liao & Smidt (2023)**: Equiformer — equivariant graph transformers.
- **These are domain-specific** (molecules, point clouds). Group-MoE is domain-agnostic.

## Gap Analysis

The existing literature addresses either:
1. **Rigid equivariance** (every layer is equivariant — Cohen, Bronstein, Weiler) — no flexibility
2. **Learned equivariance from data** (MatrixNet, Dehmamy, Benton) — no irrep guarantees
3. **Standard MoE with generic experts** (Shazeer, Fedus) — no algebraic structure
4. **Equivariant models for specific tasks** (Gordon, Fuchs, Liao) — domain-bound

Group-MoE fills the gap: **fixed algebraic structure (irrep matrices) with learned dispatch (routing)**. The model gets exact composition guarantees from the group theory AND flexibility from the router. No prior work combines these.

## Search Methodology

Web searches performed:
- "mixture of experts group equivariance irreducible representations neural networks 2024 2025"
- "learned symmetry detection routing neural networks group theory 2024 2025"
- "compositional generalization group representations permutation equivariance 2024 2025"
- Targeted searches for specific papers: arXiv:2504.09265, arXiv:2501.09571
- Reference verification: Cohen & Welling 2016, Bronstein et al. 2021, Switch Transformers
