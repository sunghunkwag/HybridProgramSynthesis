# Hybrid Program Synthesis with Safe RSI

A hybrid program synthesis engine implementing Recursive Self-Improvement (RSI) with process isolation and hierarchical abstraction.

## Core Architecture

The system is built on four safety pillars:

### 1. Process Isolation (WatchdogExecutor)
Executes all generated code in a separate process using `multiprocessing`.
- Enforces strict timeouts (default 2.0s) to prevent infinite loops.
- Captures stdout and return values without affecting the main process.
- No language-level restrictions (allows full `exec`), relying on process boundaries for safety.

### 2. Safe Interpreter
AST-based interpreter for DSL execution.
- Prevents usage of `eval` or `exec` within the synthesis loop.
- Only allows whitelisted atomic operations.

### 3. Hierarchical Library (DAG)
Manages learned concepts in a Directed Acyclic Graph.
- **Semantic Hashing**: Prevents duplicate functionality even if code differs.
- **Level Constraints**: Higher-level primitives can only utilize lower-level components.
- **Utility Scoring**: Automatically prunes unused or inefficient concepts.

### 4. RSI Transfer
Automated mechanism to transfer verified improvements to the core system.
- **Sandbox Verification**: Modifications are tested in the Watchdog sandbox first.
- **Atomic Updates**: Source code is updated only after dual verification passes.

## Usage

Run the main life loop:

```bash
python Systemtest.py hrm-life
```

## Files

- `Systemtest.py`: Main entry point containing the inlined Safe RSI engine.
- `watchdog_executor.py`: Standalone reference implementation of the process isolation module.
- `rsi_modifier_state.json`: Persisted state of the learned library and optimizer parameters.

## License

MIT
