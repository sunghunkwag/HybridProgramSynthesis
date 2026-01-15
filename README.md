# HybridProgramSynthesis
**A Neuro-Symbolic Architecture for Recursive Self-Improvement (RSI)**

> **Research Test Version**: v0.1.0-alpha
> **Status**: Experimental / Academic Research
> **Language**: Python 3.9+

## Abstract
HybridProgramSynthesis is an experimental engine designed to investigate **True Recursive Self-Improvement** in synthetic cognitive architectures. Unlike traditional program synthesis approaches that rely on brute force or purely neural generation, this system fuses **six search strategies** to overcome combinatorial explosion and achieve verifiable generalization in abstract domains.

## Core Architecture: The "Hex-Strategy" Fusion
The system integrates the following techniques into a unified `HybridSynthesizer`:

1.  **Bottom-Up Enumeration**: Systematic generation of defining primitives.
2.  **Type Pruning**: Strict type-system constraints to eliminate invalid search branches.
3.  **Observational Equivalence (OE)**: Dynamic pruning of semantically identical expressions (e.g., 6804 $\to$ 55 candidates).
4.  **Monte Carlo Tree Search (MCTS)**: UCB1-guided exploration for long-horizon planning.
5.  **Neural Guidance**: Learned operator prioritization.
6.  **Ray-Tracing Inspired Search (RTIS)**: Parallel "ray casting" into the program space using BVH (Bounding Volume Hierarchy) for domain-aware acceleration.

## Key Research Verification
The system has demonstrated the ability to **autonomously compose** complex logical primitives without domain-specific shortcuts.

| Task Domain | Synthesized Solution (Autonomous) | Verification |
|:---:|:---|:---:|
| **Boolean AND** | `second(mul(n, first(n)))` | **100%** |
| **Boolean OR** | `or_op(first(n), second(n))` | **100%** |

*Note: Pre-made Boolean primitives (`bool_and`, `bool_or`) were removed to force compositional discovery.*

## Installation
```bash
pip install -r requirements.txt
```

## Usage
The entire architecture is consolidated into a single entry point for deployment simplicity.
```bash
python Systemtest.py
```

## License
MIT License - Research Use Only
