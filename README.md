# Hybrid-Program-Synthesis

A hybrid program synthesis engine with persistent self-modification.

## Overview

This system combines six search techniques for program synthesis:
- Bottom-Up Enumeration
- Type Pruning
- Observational Equivalence
- Monte Carlo Tree Search
- Neural Guidance
- BVH-based Domain Search

State modifications are persisted to disk (`rsi_modifier_state.json`) across restarts.

## Requirements

Python 3.9+

```bash
pip install -r requirements.txt
```

## Usage

```bash
python Systemtest.py hrm-life
```

## Files

- `Systemtest.py` - Main entry point
- `neuro_genetic_synthesizer.py` - Synthesis engine
- `rsi_modifier_state.json` - Persisted learning state (auto-generated)

## License

MIT
