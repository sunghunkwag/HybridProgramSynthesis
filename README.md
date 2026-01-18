# Hybrid-Program-Synthesis-with-RSI


This repository contains an experimental implementation of a program synthesis engine capable of Recursive Self-Improvement (RSI) through neuro-genetic search and hierarchical library learning.

## Core Architecture

The system integrates four key components to achieve safe and genuine self-improvement:

1.  **Watchdog Process Isolation**: All synthesized code is executed in a detached process with strict resource limits and timeouts. This architectural decision enables the safe execution of arbitrary code, including infinite loops and complex recursion, without compromising host stability.
2.  **Neuro-Genetic Synthesizer**: A hybrid search engine combining neural guidance with genetic algorithms. Recent updates include full support for **Lambda expressions**, **Closures**, and **Higher-Order Functions**, allowing the discovery of generic algorithms (e.g., `map`, `reduce`, `fix`) beyond simple arithmetic.
3.  **Recursive Self-Improvement (RSI)**: The system maintains a persistent, evolving library of primitives (`rsi_primitive_registry.json`).
    *   **Meta-Learning**: Feature weights are adjusted via reinforcement learning based on synthesis success rates.
    *   **Semantic Abstraction**: Novel algorithms are abstracted into reusable primitives, verified for semantic uniqueness to prevent bloat.
4.  **Safe Interpreter**: A custom AST-based validator that enforces safety constraints during the search phase while permitting necessary language features like `lambda` and `recursion` for complex logic.

## Usage

To initiate the continuous self-improvement loop:

```bash
python Systemtest.py hrm-life
```

## Repository Structure

*   `Systemtest.py`: Primary entry point and orchestration engine.
*   `neuro_genetic_synthesizer.py`: Core synthesis logic implementing the genetic algorithm and safe interpreter.
*   `watchdog_executor.py`: Process isolation module for safe code execution.
*   `rsi_primitive_registry.json`: Persistent knowledge base of discovered primitives.

## License

MIT
