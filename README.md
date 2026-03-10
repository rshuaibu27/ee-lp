# EE-LP: Pairwise Early-Exit Graph Neural Networks for Link Prediction

This repository accompanies my L65 Geometric Deep Learning mini-project submission.

## Overview

Standard GNN-based link prediction applies the same computational depth to every candidate pair, regardless of difficulty. EE-LP addresses this by introducing **pairwise early-exit decisions**: at each message-passing layer, a lightweight confidence network inspects the concatenated pair representation $[h_u^l \| h_v^l]$ and produces a Neural Adaptive Step $\tau_{uv}^l \in [0,1]$. When $\tau \to 0$, the pair's representation freezes — encoding an early exit at that layer. Different pairs exit at different depths.

EE-LP is built on the **SAS-GNN** backbone (Di Francesco et al., 2026), whose weight-sharing and stability guarantees make it well-suited to variable-depth inference. The model is trained end-to-end with binary cross-entropy loss only — no auxiliary budget penalty.

Evaluated under the **HeaRT benchmarking protocol** (hard negatives, leakage-free splits), EE-LP consistently outperforms GCN, GraphSAGE, and GAT across all metrics on Cora and CiteSeer.

## Results (Hits@20, HeaRT protocol)

| Dataset   | GCN   | GraphSAGE | GAT   | **EE-LP** |
|-----------|-------|-----------|-------|-----------|
| Cora      | 0.095 | 0.082     | 0.144 | **0.288** |
| CiteSeer  | 0.111 | 0.117     | 0.166 | **0.252** |

## Project Structure
```
ee-lp/
├── models/
│   ├── ee_lp.py          # Core model: SASConv, PairConfidenceNet, EELP, baselines
│   └── __init__.py
├── data/
│   ├── dataset.py        # Cora/CiteSeer loader with HeaRT protocol
│   ├── ogbl_collab.py    # ogbl-collab loader
│   └── __init__.py
├── train.py              # Training loop, evaluation (Hits@K, MRR)
├── experiments.py        # Full experiment runner (all models, datasets, plots)
└── requirements.txt
```

## Setup
```bash
pip install torch>=2.0.0
pip install torch-geometric>=2.4.0
pip install ogb numpy matplotlib tqdm
```

## Running Experiments
```bash
# Cora and CiteSeer (recommended, runs in ~2 hours on a T4 GPU)
python experiments.py --datasets cora citeseer

# ogbl-collab (requires ~4 hours on a T4 GPU)
python experiments.py --datasets collab

# All datasets
python experiments.py --datasets all
```

Results are saved to `results/` as JSON and PDF/PNG plots.

## Reproducing in Google Colab

1. Open [Google Colab](https://colab.research.google.com) and set runtime to **GPU (T4)**
2. Run the following setup:
```python
!git clone https://github.com/rshuaibu27/ee-lp.git
%cd ee-lp
!pip install torch-geometric ogb tqdm matplotlib -q
```

3. Then run experiments:
```python
!python experiments.py --datasets cora citeseer
```

## Key Design Decisions

**Why SAS-GNN as backbone?** Its weight-sharing means constant parameter count regardless of depth — essential for variable-depth inference. Its stability guarantees (Theorem 3.1 in the base paper) ensure intermediate representations remain meaningful at any exit point.

**Why no budget loss?** Following Di Francesco et al. (2026), exit decisions are driven purely by task confidence. Budget-aware training causes premature exits on tasks requiring deep exploration — which link prediction is.

**Why HeaRT evaluation?** Standard link prediction benchmarks suffer from target leakage and trivially easy negatives. HeaRT corrects both, giving a more honest picture of model performance.

## References

- Di Francesco et al. (2026). *Early-Exit Graph Neural Networks*. arXiv:2505.18088
- Kang et al. (2023). *Evaluating GNNs for Link Prediction: Current Pitfalls and New Benchmarking*. NeurIPS 2023
- Hu et al. (2020). *Open Graph Benchmark*. NeurIPS 2020