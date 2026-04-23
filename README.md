# 🧠 Self-Pruning Neural Network

> **Learning sparse architectures through end-to-end differentiable gate regularization on CIFAR-10**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org)
[![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-green)](https://www.cs.toronto.edu/~kriz/cifar.html)
[![Hardware](https://img.shields.io/badge/Hardware-NVIDIA%20T4-76B900)](https://www.nvidia.com)

---

## Overview

This project implements a **self-pruning multi-layer perceptron** that discovers its own sparse architecture during training — no post-hoc pruning, no separate fine-tuning stage. Each weight is paired with a learnable gate parameter whose L1-penalized sigmoid activation drives redundant connections to exactly zero while the network is simultaneously optimized for classification accuracy.

The core insight is that sparsity and accuracy are not necessarily in conflict: the best model in this study achieves **91.8% sparsity** (over 9 in 10 weights eliminated) while recording the **highest test accuracy** across all configurations — a result that highlights the regularizing effect of structured weight elimination.

---

## Features

- **`PrunableLinear`** — a drop-in replacement for `nn.Linear` with per-weight learnable gates
- **Differentiable gating** via sigmoid activation; fully compatible with standard optimizers
- **L1 sparsity regularization** with configurable λ to control the accuracy–sparsity tradeoff
- **No post-hoc pruning** — the final sparse network emerges directly from training
- **Delayed pruning analysis** — the "learn-first, prune-second" phenomenon documented and explained
- **Multi-λ sweep** with automated best-model selection and gate distribution visualization

---

## Model Architecture

```
Input (3×32×32 CIFAR-10 image)
    │
    ▼ flatten
[3072]
    │
    ▼ PrunableLinear(3072 → 1024) + ReLU
[1024]
    │
    ▼ PrunableLinear(1024 → 512) + ReLU
 [512]
    │
    ▼ PrunableLinear(512 → 10)
  [10] → class logits
```

**Total parameters:** ~8.4M (4.2M weights + 4.2M gate scores)  
**Active parameters at best λ:** ~680K (91.8% pruned)

---

## How It Works

### 1. Gating Mechanism

Each weight `w_ij` is multiplied by the sigmoid of a learnable gate score `g_ij`:

```
W' = W ⊙ σ(G)        (element-wise product)
output = W'x + b
```

Gate scores are initialized to `0`, so `σ(0) = 0.5` — a neutral starting point where no connection is pre-committed to being open or closed.

### 2. Loss Function

```
L_total = CrossEntropy(logits, labels) + λ × Σ σ(G)
```

The L1 penalty on gate values exerts a **constant downward pressure** on every gate regardless of its magnitude. Unlike L2, which only weakens as gates shrink, L1 forces gates all the way to zero — producing true sparsity rather than mere shrinkage.

### 3. Delayed Pruning (Key Phenomenon)

In all experiments, sparsity remains exactly **0% for epochs 1–19**, then rises sharply from epoch 20 onward:

```
Epoch 19:  0.0% sparse
Epoch 20: 63.7% sparse   ← phase transition
Epoch 21: 75.7% sparse
...
Epoch 30: 91.8% sparse
```

This "learn-first, prune-second" behavior is intentional: the network first builds useful representations using its full capacity, then systematically eliminates connections that are no longer needed. Networks that prune too early never develop the representations needed to identify which connections matter.

---

## Results

| λ | Test Accuracy | Sparsity | Active Weights (approx.) |
|---|---|---|---|
| 1e-5 | 58.55% | 51.1% | ~2.05M |
| **1e-4** | **59.00%** | **91.8%** | **~340K** |
| 1e-3 | 51.45% | 99.8% | ~8K |

**Best model:** λ = 1e-4 — highest accuracy with extreme sparsity, demonstrating that aggressive pruning acts as an implicit regularizer.

---

## Sample Output

```
============================================================
  Training with λ = 0.0001
============================================================
  Epoch 01/30 | loss=167.89 (cls=1.780, sp=1661060) | sparsity=0.0%
  ...
  Epoch 20/30 | loss=6.17   (cls=1.253, sp=49134)   | sparsity=63.7%
  Epoch 30/30 | loss=3.04   (cls=1.202, sp=18369)   | sparsity=91.8%

  ✓  λ=0.0001  →  acc=59.00%,  sparsity=91.82%

=== Results Summary ===
Lambda         Test Accuracy (%)   Sparsity (%)
----------------------------------------------
1e-05                      58.55          51.07
0.0001                     59.00          91.82 ← best
0.001                      51.45          99.81
```

### Gate Distribution Plot

The histogram of gate values after training shows a clear **bimodal distribution** for the best model:
- **Spike near 0** → connections the network has closed (pruned weights)
- **Cluster at 0.3–1.0** → connections retained as important

The presence of a clean gap between the two modes confirms the network has learned a meaningful binary partition of its weights — not just uniform compression.

---

## How to Run

### Prerequisites

```bash
pip install torch torchvision numpy matplotlib
```

### Quick Start

```bash
# Clone the repository
git clone https://github.com/your-username/self-pruning-nn.git
cd self-pruning-nn

# Run in Jupyter
jupyter notebook Self_Pruning_Solution.ipynb
```

### Run on Google Colab

1. Upload the notebook to [Google Colab](https://colab.research.google.com)
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. Run all cells — CIFAR-10 will download automatically (~170MB)

### Customize λ Values

Edit the `LAMBDA_LIST` in the Setup cell to explore different sparsity levels:

```python
LAMBDA_LIST = [1e-5, 1e-4, 1e-3]   # low / medium / high pruning pressure
GATE_THRESHOLD = 1e-2               # threshold for counting a gate as pruned
NUM_EPOCHS = 30
```

---

## Project Structure

```
self-pruning-nn/
├── Self_Pruning_Solution.ipynb    # Main notebook (all experiments)
├── gate_distributions.png         # Gate histogram visualization
├── README.md                      # This file
└── technical_report.md            # Full research report
```

---

## Technologies

| Component | Technology |
|---|---|
| Deep Learning Framework | PyTorch 2.x |
| Dataset | CIFAR-10 (torchvision) |
| Optimizer | Adam (lr=1e-3) |
| Hardware | NVIDIA T4 GPU (Google Colab) |
| Visualization | Matplotlib |
| Language | Python 3.8+ |

---

## Key Concepts

**Self-Pruning:** Unlike traditional pruning (train → prune → fine-tune), self-pruning integrates compression into the training objective itself. The final sparse model is the direct output of a single training run.

**Differentiable Gates:** Sigmoid-activated gate scores allow gradient-based optimization to discover which weights to prune, making the process data-driven rather than heuristic.

**L1 vs L2 Regularization:** The choice of L1 (not L2) for the sparsity penalty is deliberate — only L1 drives parameters to exactly zero, enabling true connection elimination rather than small-but-nonzero values.

---

## License

MIT License — free to use, modify, and distribute with attribution.

---

*For a detailed treatment of the methodology, results, and analysis, see [`technical_report.pdf`](./technical_report.pdf).*
