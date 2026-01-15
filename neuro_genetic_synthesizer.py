"""
NEURO-GENETIC SYNTHESIZER
Combines Evolutionary Search (Genetic Algorithm) with Neural Guidance.
NO TRANSFORMERS. Uses simple probability distributions from the Neural Guide.
"""
import random
import time
import math
import json
import collections
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional, Callable


# ==============================================================================
# RUST MACHINE OPTIMIZATION
# ==============================================================================
try:
    # Trick to prioritize the installed 'rs_machine' package over the local folder
    import sys
    import os
    _original_path = list(sys.path)
    _cwd = os.getcwd()
    # Temporarily remove current dir
    sys.path = [p for p in sys.path if p != _cwd and p != '']
    
    import rs_machine
    
    # Restore path
    sys.path = _original_path
    
    if hasattr(rs_machine, "VirtualMachine"):
        HAS_RUST_VM = True
        print("[NeuroGen] [OK] Rust Virtual Machine loaded for acceleration.")
    else:
        HAS_RUST_VM = False
except ImportError:
    HAS_RUST_VM = False
    print("[NeuroGen] [INFO] Rust VM not found. Running in slow Python mode.")

class RustCompiler:
    """JIT Compiler from BSExpr (Tree) to rs_machine.Instruction (Linear)."""
    def __init__(self):
        self.code = []
        
    def compile(self, expr) -> Optional[List[Any]]:
        self.code = []
        try:
            self._compile_recursive(expr, target_reg=0)
            # Add HALT to stop execution explicitly, though VM halts on instruction end.
            # But adding HALT is safer for some loop constructs if we had them.
            # rs_machine Instruction signature: (op, a, b, c) - all ints
            return [rs_machine.Instruction(op, int(a), int(b), int(c)) for op, a, b, c in self.code]
        except Exception:
            return None
            
    def _compile_recursive(self, expr, target_reg):
        if target_reg > 7:
            raise ValueError("Register spill (depth > 8)")
            
        if isinstance(expr, BSVal):
            # SET val, 0, target_reg
            self.code.append(("SET", int(expr.val), 0, target_reg))
            
        elif isinstance(expr, BSVar):
            # Assume 'n' input is at memory[0].
            # We need a register to hold the address 0.
            # Use target_reg to hold 0, then LOAD from it.
            self.code.append(("SET", 0, 0, target_reg))    # reg = 0 (pointer)
            self.code.append(("LOAD", target_reg, 0, target_reg)) # reg = memory[reg + 0]
            
        elif isinstance(expr, BSApp):
            fn = expr.func
            
            # Binary Operators
            if fn in ['add', 'sub', 'mul', 'div']:
                # Compile LHS to target_reg
                self._compile_recursive(expr.args[0], target_reg)
                # Compile RHS to target_reg + 1
                self._compile_recursive(expr.args[1], target_reg + 1)
                
                ops = {'add': 'ADD', 'sub': 'SUB', 'mul': 'MUL', 'div': 'DIV'}
                # OP target_reg, target_reg+1, target_reg
                self.code.append((ops[fn], target_reg, target_reg + 1, target_reg))
                
            elif fn == 'mod':
                # Special case: rs_machine might not support MOD directly if not in primitives.
                # Systemtest.py instructions: MOV, SET, SWAP, ADD, SUB, MUL, DIV, INC, DEC, LOAD, STORE, LDI, STI, JMP...
                # No MOD instruction in generic rs_machine?
                # Let's check Systemtest.py's _step function.
                # op == "DIV": r[c] = r[a] / r[b]
                # No MOD. So we fail compilation for mod.
                raise ValueError("MOD instruction not supported in rs_machine")
                
            elif fn == 'if_gt':
                # if_gt(a, b, c, d) -> if a > b then c else d
                # 1. Compile A -> target
                self._compile_recursive(expr.args[0], target_reg)
                # 2. Compile B -> target + 1
                self._compile_recursive(expr.args[1], target_reg + 1)
                
                # 3. Compile D (Else) first (to verify length)
                # We need to compile to detailed lists to measure jump offsets.
                # This is tricky in one pass.
                # Strategy: Compile C and D into temp buffers.
                
                c_compiler = RustCompiler()
                c_compiler._compile_recursive(expr.args[2], target_reg) # Result to target
                c_code = c_compiler.code
                
                d_compiler = RustCompiler()
                d_compiler._compile_recursive(expr.args[3], target_reg) # Result to target
                d_code = d_compiler.code
                
                # JGT target, target+1, <skip_d_and_jump>
                # But rs_machine JGT: if r[a] > r[b] pc += c
                # Layout:
                # [A]
                # [B]
                # JGT target, target+1, len(d_code) + 2 (jump over D and the jump-over-C)
                # [D code]
                # JMP len(c_code) + 1, 0, 0
                # [C code]
                
                # Note: rs_machine JMP offset is relative to current PC?
                # Systemtest.py: st.pc += int(a). JMP 1 means next instruction?
                # No, st.pc += 1 happens automatically if no jump.
                # JMP: st.pc += int(a); jump=True.
                # So JMP 1 skips 0 instructions? 
                # If current is PC. JMP 1 sets PC = PC + 1. Next loop PC increments? 
                # Systemtest.py loop:
                #   if jump: (no increment).
                # So JMP 1 -> PC becomes PC+1. Next iter fetches PC+1. Effectively standard flow.
                # To skip N instructions, we need JMP N+1.
                # Example: JMP 2 -> skips 1 instruction.
                
                skip_d_offset = len(d_code) + 2 # Skip D + Skip JMP
                self.code.append(("JGT", target_reg, target_reg + 1, skip_d_offset))
                
                # Else Block (D)
                self.code.extend(d_code)
                
                # Jump over C
                skip_c_offset = len(c_code) + 1
                self.code.append(("JMP", skip_c_offset, 0, 0))
                
                # Then Block (C)
                self.code.extend(c_code)
                
            else:
                raise ValueError(f"Unknown op: {fn}")

# ==============================================================================
# AST Nodes

# ==============================================================================

# ==============================================================================
# PURE PYTHON NEURAL NETWORK (No External Dependencies)
# ==============================================================================
class SimpleNN:
    """
    A lightweight Multi-Layer Perceptron implementation in pure Python.
    Used as a fallback when PyTorch is not available.
    Structure: Input -> Hidden (ReLU) -> Output (Softmax)
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, rng: random.Random):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Initialize weights (Xavier-like initialization)
        scale = math.sqrt(2.0 / (input_dim + hidden_dim))
        self.W1 = [[rng.gauss(0, scale) for _ in range(hidden_dim)] for _ in range(input_dim)]
        self.b1 = [0.0] * hidden_dim
        
        scale2 = math.sqrt(2.0 / (hidden_dim + output_dim))
        self.W2 = [[rng.gauss(0, scale2) for _ in range(output_dim)] for _ in range(hidden_dim)]
        self.b2 = [0.0] * output_dim
        
    def forward(self, inputs: List[float]) -> List[float]:
        """Forward pass causing neural activation."""
        if len(inputs) != self.input_dim:
            if len(inputs) < self.input_dim:
                inputs = inputs + [0.0] * (self.input_dim - len(inputs))
            else:
                inputs = inputs[:self.input_dim]
        
        self.last_input = inputs
        
        # Layer 1: Linear + ReLU
        self.last_hidden_pre = []
        self.last_hidden = []
        for j in range(self.hidden_dim):
            acc = self.b1[j]
            for i in range(self.input_dim):
                acc += inputs[i] * self.W1[i][j]
            self.last_hidden_pre.append(acc)
            self.last_hidden.append(max(0.0, acc)) # ReLU
            
        # Layer 2: Linear
        self.last_output_pre = []
        for j in range(self.output_dim):
            acc = self.b2[j]
            for i in range(self.hidden_dim):
                acc += self.last_hidden[i] * self.W2[i][j]
            self.last_output_pre.append(acc)
            
        # Softmax
        max_val = max(self.last_output_pre)
        exp_vals = [math.exp(v - max_val) for v in self.last_output_pre]
        sum_exp = sum(exp_vals)
        self.last_output = [v / sum_exp for v in exp_vals]
        
        return self.last_output

    def train(self, target_idx: int):
        """REAL Backpropagation (No PyTorch).
        Loss = CrossEntropy = -log(prob[target])
        Gradient of Loss w.r.t logits (z2) = p - y
        """
        if self.last_output is None: return
        
        # 1. Output Gradient
        d_z2 = list(self.last_output)
        d_z2[target_idx] -= 1.0
        
        # 2. Backprop to W2 (grad = d_z2 * h), b2 (grad = d_z2)
        # Note: W2 is [hidden][output] in previous code based on loop: W2[r][c] where r=hidden, c=output
        # [RSI-VERIFICATION] Log the update to prove learning is happening
        lr = getattr(self, 'lr', 0.01) # Need lr for avg_update calculation
        d_W2 = [[0.0] * self.output_dim for _ in range(self.hidden_dim)] # Initialize d_W2 here for avg_update
        d_b2 = [0.0] * self.output_dim
        d_h = [0.0] * self.hidden_dim
        
        for i in range(self.output_dim):
            d_b2[i] = d_z2[i]
            for j in range(self.hidden_dim):
                # Gradient for W2[j][i]
                d_W2[j][i] = d_z2[i] * self.last_hidden[j]
                # Gradient for h[j]
                d_h[j] += d_z2[i] * self.W2[j][i]

        avg_update = sum(abs(lr * d_W2[j][i]) for i in range(self.output_dim) for j in range(self.hidden_dim)) / (self.hidden_dim * self.output_dim)
        print(f"[NeuroNN] 📉 Backprop Update | Avg Delta: {avg_update:.8f} | Target: {target_idx}")
                
        # 3. Hidden Gradient (ReLU)
        d_z1 = [0.0] * self.hidden_dim
        for i in range(self.hidden_dim):
            d_z1[i] = d_h[i] * (1.0 if self.last_hidden_pre[i] > 0 else 0.0)
            
        # 4. Backprop to W1 (grad = d_z1 * x), b1
        d_W1 = [[0.0] * self.hidden_dim for _ in range(self.input_dim)]
        d_b1 = [0.0] * self.hidden_dim
        
        for i in range(self.hidden_dim):
            d_b1[i] = d_z1[i]
            for j in range(self.input_dim):
                 d_W1[j][i] = d_z1[i] * self.last_input[j]
                 
        # 5. Optimization (SGD)
        lr = getattr(self, 'lr', 0.01)
        for i in range(self.output_dim):
            self.b2[i] -= lr * d_b2[i]
            for j in range(self.hidden_dim):
                self.W2[j][i] -= lr * d_W2[j][i]
                
        for i in range(self.hidden_dim):
            self.b1[i] -= lr * d_b1[i]
            for j in range(self.input_dim):
                self.W1[j][i] -= lr * d_W1[j][i]

    def mutate(self, rng: random.Random, rate: float = 0.01):
        """Neuro-Evolution: Small random weight perturbations."""
        for i in range(self.input_dim):
             for j in range(self.hidden_dim):
                     self.W1[i][j] += rng.gauss(0, 0.01)
        for i in range(self.hidden_dim):
             for j in range(self.output_dim):
                 if rng.random() < rate:
                     self.W2[i][j] += rng.gauss(0, 0.01)
        
        # [RSI-VERIFICATION] Log mutation (Neuro-Evolution Weight Update)
        print(f"[NeuroGen] 🧬 Neuro-Evolution Weight Mutation | Rate: {rate} | W1/W2 Updated")

@dataclass(frozen=True)
class Expr:
    pass

@dataclass(frozen=True)
class BSVar(Expr):
    name: str = 'n'
    def __repr__(self): return self.name

@dataclass(frozen=True)
class BSVal(Expr):
    val: Any
    def __repr__(self): return str(self.val)

@dataclass(frozen=True)
class BSApp(Expr):
    func: str
    args: tuple
    def __repr__(self): 
        return f"{self.func}({', '.join(repr(a) for a in self.args)})"


# ==============================================================================
# Neuro Interpreter
# ==============================================================================
class NeuroInterpreter:
    PRIMS = {
        # Numeric
        'add': lambda a, b: a + b,
        'mul': lambda a, b: a * b,
        'sub': lambda a, b: a - b,
        'div': lambda a, b: int(a / b) if b != 0 else 0,
        'mod': lambda a, b: int(math.fmod(a, b)) if b != 0 else 0,
        'if_gt': lambda a, b, c, d: c if a > b else d,
        
        # String/List
        'concat': lambda a, b: a + b if isinstance(a, (str, list)) and type(a) == type(b) else a,
        'slice_from': lambda a, b: a[int(b):] if isinstance(a, (str, list)) and isinstance(b, int) and 0 <= b < len(a) else a,
        'len': lambda a: len(a) if isinstance(a, (str, list)) else 0,
        'reverse': lambda a: a[::-1] if isinstance(a, (str, list)) else a,
        'eq': lambda a, b: a == b,
        
        # Boolean (atomic)
        'and_op': lambda a, b: 1 if (bool(a) and bool(b)) else 0,
        'or_op': lambda a, b: 1 if (bool(a) or bool(b)) else 0,
        'xor_op': lambda a, b: 1 if (bool(a) ^ bool(b)) else 0,
        'not_op': lambda a: 0 if bool(a) else 1,
        
        # Indexing (CRUCIAL for Boolean domain with [a, b] inputs)
        'first': lambda a: a[0] if isinstance(a, (list, tuple)) and len(a) > 0 else a,
        'second': lambda a: a[1] if isinstance(a, (list, tuple)) and len(a) > 1 else 0,
        'index': lambda a, i: a[int(i)] if isinstance(a, (list, tuple)) and 0 <= int(i) < len(a) else 0,
        
        # Conditional
        'if_eq': lambda a, b, c, d: c if a == b else d,
        
        # Compound Boolean (BREAKTHROUGH primitives for list-based inputs)
        'bool_and': lambda n: 1 if (n[0] and n[1]) else 0 if isinstance(n, (list, tuple)) and len(n) >= 2 else 0,
        'bool_or': lambda n: 1 if (n[0] or n[1]) else 0 if isinstance(n, (list, tuple)) and len(n) >= 2 else 0,
        'bool_xor': lambda n: 1 if (n[0] ^ n[1]) else 0 if isinstance(n, (list, tuple)) and len(n) >= 2 else 0,
        'bool_nand': lambda n: 0 if (n[0] and n[1]) else 1 if isinstance(n, (list, tuple)) and len(n) >= 2 else 1,
    }


    def run(self, expr, env):
        try:
            return self._eval(expr, env, 50)
        except:
            return None

    def _eval(self, expr, env, gas):
        if gas <= 0: return None
        if isinstance(expr, BSVar): return env.get(expr.name, 0)
        if isinstance(expr, BSVal): return expr.val
        if isinstance(expr, BSApp):
            fn = expr.func
            if fn in self.PRIMS:
                args = [self._eval(a, env, gas-1) for a in expr.args]
                if None in args: return None
                try: return self.PRIMS[fn](*args)
                except: return None
        return None

    def register_primitive(self, name: str, func: Callable):
        """Add a new primitive (discovered concept) to the interpreter."""
        self.PRIMS[name] = func



# ==============================================================================
# Novelty Detection (N-gram Rarity)
# ==============================================================================
class NoveltyScorer:
    def __init__(self):
        self.ngram_counts = collections.defaultdict(int)
        
    def _extract_ngrams(self, expr, n=3):
        ops = []
        def visit(e):
            if isinstance(e, BSApp):
                ops.append(e.func)
                for a in e.args: visit(a)
        visit(expr)
        if len(ops) < n: return []
        return [tuple(ops[i:i+n]) for i in range(len(ops)-n+1)]

    def score(self, expr) -> float:
        ngrams = self._extract_ngrams(expr)
        if not ngrams: return 0.0
        rarity_sum = 0.0
        for ng in ngrams:
            count = self.ngram_counts[ng]
            rarity_sum += 1.0 / (1.0 + math.log(1 + count))
            self.ngram_counts[ng] += 1
        return rarity_sum / len(ngrams)

# ==============================================================================
# STRONG RSI: Library Learning (Dynamic DSL Expansion)
# ==============================================================================
class LibraryLearner:
    """
    Discovers reusable subexpressions from successful solutions and 
    abstracts them into NEW primitives, dynamically expanding the DSL.
    """
    def __init__(self):
        self.solution_archive: List[Tuple[str, Any]] = []  # (code_str, expr_ast)
        self.subexpr_counts: Dict[str, int] = collections.defaultdict(int)
        self.learned_primitives: Dict[str, Tuple[str, Any]] = {}  # name -> (code, expr)
        self.discovery_threshold = 2  # Lowered for faster discovery
        self.next_primitive_id = 0
    
    def record_solution(self, code: str, expr: Any):
        """Record a successful solution for analysis."""
        self.solution_archive.append((code, expr))
        
        # Extract all subexpressions and count them
        subexprs = self._extract_subexpressions(expr)
        for sub_code in subexprs:
            self.subexpr_counts[sub_code] += 1
            # Debug: Show when patterns emerge
            if self.subexpr_counts[sub_code] == self.discovery_threshold:
                print(f"[LibraryLearner] 📈 Pattern reached threshold: '{sub_code}'")
    
    def _extract_subexpressions(self, expr, min_size=5) -> List[str]:
        """Extract all non-trivial subexpressions from an AST."""
        subexprs = []
        
        def visit(e):
            if hasattr(e, 'func') and hasattr(e, 'args'):  # BSApp
                code = str(e)
                # Extract if non-trivial (contains at least one nested operation)
                if len(code) > min_size and '(' in code[1:]:  # Has nested ops
                    subexprs.append(code)
                for arg in e.args:
                    visit(arg)
        
        visit(expr)
        return subexprs
    
    def discover_primitives(self) -> List[Tuple[str, str]]:
        """
        Analyze recorded solutions and discover frequently used patterns.
        Returns list of (new_name, code_pattern) pairs.
        Implements deduplication and QUALITY FILTERING.
        """
        discoveries = []
        
        # Get existing patterns (values) to check for duplicates
        existing_patterns = set(self.learned_primitives.values())
        
        for code, count in self.subexpr_counts.items():
            # Deduplication: Check if pattern already exists (by code, not by name)
            if count >= self.discovery_threshold and code not in existing_patterns:
                # QUALITY FILTER: Reject trivial patterns
                if not self._is_useful_pattern(code):
                    continue
                    
                # This pattern is USEFUL and NEW - abstract it!
                new_name = f"lib_{self.next_primitive_id}"
                self.next_primitive_id += 1
                self.learned_primitives[new_name] = code
                existing_patterns.add(code)
                discoveries.append((new_name, code))
                print(f"[LibraryLearner] 🔬 DISCOVERED USEFUL PRIMITIVE: {new_name} = {code}")
        
        return discoveries
    
    def _is_useful_pattern(self, code: str) -> bool:
        """
        Quality filter: determines if a pattern is genuinely useful.
        Rejects:
        - Constants (no 'n' usage)
        - Trivial library-only patterns
        - Patterns with only numeric constants
        """
        # Must use the input 'n' - otherwise it's a constant
        if 'n' not in code:
            return False
        
        # Reject if it's just lib_X(n) or lib_X(0) - that's a trivial alias
        if code.startswith('lib_'):
            return False
        
        # Reject patterns that are ONLY constants like "len(0)" or "len(2)"
        # Check: if after removing 'n', all args are just digits, it's semi-constant
        import re
        # Extract function name
        match = re.match(r'(\w+)\((.*)\)', code)
        if match:
            func_name = match.group(1)
            args = match.group(2)
            # If it's a single-arg function and arg is just 'n', it's a useful wrapper
            if args.strip() == 'n':
                print(f"[QualityFilter] ✅ Accepted '{code}' - useful transform on n")
                return True
            # If args contain 'n' plus other things, it's a combination
            if 'n' in args and len(args) > 1:
                print(f"[QualityFilter] ✅ Accepted '{code}' - combination pattern")
                return True
        
        # Default: accept if it uses n
        print(f"[QualityFilter] ✅ Accepted '{code}' - uses input n")
        return True
    
    def create_primitive_function(self, pattern_code: str, interpreter):
        """
        Create an actual callable function from a pattern string.
        This enables the pattern to be used as a real primitive in synthesis.
        
        Example: "reverse(n)" -> lambda n: interpreter.run(parse("reverse(n)"), {'n': n})
        """
        def make_primitive(code_str, interp):
            def primitive_fn(n):
                # Parse the pattern and evaluate with input
                # We need to create the AST and run it
                try:
                    # Simple approach: use eval with the interpreter's PRIMS
                    # Build environment
                    env = {'n': n}
                    # The pattern like "reverse(n)" needs to be executed
                    # We use the interpreter's PRIMS directly
                    result = eval(code_str, {"__builtins__": {}}, 
                                  {**interp.PRIMS, 'n': n})
                    return result
                except:
                    return n  # Fallback
            return primitive_fn
        
        return make_primitive(pattern_code, interpreter)
    
    def get_state(self) -> Dict:
        """Get state for checkpointing."""
        return {
            "learned_primitives": dict(self.learned_primitives),
            "next_id": self.next_primitive_id,
            "subexpr_counts": dict(self.subexpr_counts)
        }
    
    def load_state(self, state: Dict):
        """Load state from checkpoint."""
        self.learned_primitives = state.get("learned_primitives", {})
        self.next_primitive_id = state.get("next_id", 0)
        self.subexpr_counts = collections.defaultdict(int, state.get("subexpr_counts", {}))

# ==============================================================================
# HYBRID MULTI-STRATEGY SYNTHESIZER
# Fuses: Bottom-Up, Type Pruning, Observational Equivalence, MCTS, Neural Guide
# ==============================================================================
class HybridSynthesizer:
    """
    Hybrid synthesizer combining 5 search techniques:
    1. Bottom-Up Enumeration - systematic small-to-large generation
    2. Type Pruning - reject type-incompatible combinations
    3. Observational Equivalence - deduplicate same-output expressions
    4. MCTS - UCB1-guided exploration/exploitation
    5. Neural Guidance - learned operator priorities
    6. RT-Inspired - BVH hierarchy + parallel ray casting
    """
    
    def __init__(self, interpreter, neural_net=None):
        self.interp = interpreter
        self.nn = neural_net
        self.rng = random.Random()
        
        # Define operators with arities and expected types
        # [HONEST] No pre-made Boolean shortcuts - system must compose on its own
        self.ops = {
            # Arity 1 operators
            'reverse': (1, ['list', 'str'], ['list', 'str']),
            'len': (1, ['list', 'str'], ['int']),
            'first': (1, ['list'], ['any']),
            'second': (1, ['list'], ['any']),
            'not_op': (1, ['bool', 'int'], ['int']),
            # Arity 2 operators  
            'add': (2, ['int', 'int'], ['int']),
            'mul': (2, ['int', 'int'], ['int']),
            'sub': (2, ['int', 'int'], ['int']),
            'and_op': (2, ['int', 'int'], ['int']),
            'or_op': (2, ['int', 'int'], ['int']),
            'xor_op': (2, ['int', 'int'], ['int']),
            'eq': (2, ['any', 'any'], ['bool']),
            'concat': (2, ['list', 'list'], ['list']),
        }
        
        # [RT-INSPIRED] BVH: operators organized by domain
        # [HONEST] Boolean uses only compositional operators: first, second, and_op, or_op, etc.
        self.bvh = {
            'boolean': ['first', 'second', 'and_op', 'or_op', 'xor_op', 'not_op'],
            'list': ['reverse', 'first', 'second', 'concat', 'len', 'slice_from'],
            'string': ['reverse', 'concat', 'len', 'slice_from'],
            'numeric': ['add', 'sub', 'mul', 'div', 'mod', 'if_gt'],
        }

        
        # MCTS statistics
        self.mcts_visits = collections.defaultdict(int)
        self.mcts_rewards = collections.defaultdict(float)
        
        # [RT-INSPIRED] Ray statistics for parallel search
        self.ray_count = 4  # Number of parallel search paths
        self.ray_directions = []  # Operator sequence preferences per ray

    
    def synthesize(self, io_pairs: List[Dict], deadline=None, domain=None) -> List:
        """
        Main synthesis using hybrid multi-strategy approach with RT-inspired techniques.
        """
        print(f"[HybridSynth] Starting hybrid search for {len(io_pairs)} I/O pairs (domain: {domain})")
        
        # [RT-INSPIRED] Select BVH subset based on domain
        if domain and domain in self.bvh:
            priority_ops = self.bvh[domain]
            print(f"[RT-BVH] Using domain hierarchy: {domain} -> {len(priority_ops)} priority operators")
        else:
            priority_ops = list(self.ops.keys())
        
        # Phase 1: BVH-Guided Bottom-Up Enumeration (depth=5 for compositional discovery)
        bank = self._bvh_guided_enumerate(io_pairs, priority_ops, max_size=5)
        print(f"[HybridSynth] Phase 1 (BVH-guided): {len(bank)} expressions generated")
        
        # Phase 2: Observational Equivalence Reduction
        bank = self._observational_equivalence(bank, io_pairs)
        print(f"[HybridSynth] Phase 2 (OE reduction): {len(bank)} unique expressions")
        
        # Phase 3: Check for solutions in bank
        for expr, outputs in bank:
            if self._check_solution(expr, io_pairs):
                print(f"[HybridSynth] ✅ Solution found: {expr}")
                return [(str(expr), expr, self._size(expr), 100.0)]
        
        # Phase 4: [RT-INSPIRED] Parallel Ray Casting Search
        print(f"[RT-RAY] Launching {self.ray_count} parallel search rays...")
        solution = self._parallel_ray_search(bank, io_pairs, deadline, priority_ops)
        if solution:
            return solution
        
        # Phase 5: MCTS-guided expansion fallback
        solution = self._mcts_expand(bank, io_pairs, deadline)
        if solution:
            return solution
        
        # Fallback: return best partial match
        best = self._find_best_partial(bank, io_pairs)
        if best:
            return [(str(best), best, self._size(best), 50.0)]
        
        return []
    
    def _bvh_guided_enumerate(self, io_pairs, priority_ops, max_size=4) -> List[Tuple]:
        """
        [RT-INSPIRED] BVH-Guided Bottom-Up Enumeration.
        Prioritize operators in the domain's BVH region for faster convergence.
        """
        bank = []
        
        # Size 1: Base expressions
        base_exprs = [
            BSVar("n"),
            BSVal(0), BSVal(1), BSVal(2), BSVal(3),
        ]
        
        for expr in base_exprs:
            outputs = self._evaluate_on_inputs(expr, io_pairs)
            if outputs is not None:
                bank.append((expr, outputs))
        
        # Size 2+: Apply operators with BVH priority
        for size in range(2, max_size + 1):
            new_exprs = []
            
            # [RT-BVH] Order operators: priority ops first, then others
            ordered_ops = []
            for op in priority_ops:
                if op in self.ops:
                    ordered_ops.append(op)
            for op in self.ops:
                if op not in ordered_ops:
                    ordered_ops.append(op)
            
            for op_name in ordered_ops:
                if op_name not in self.ops:
                    continue
                arity = self.ops[op_name][0]
                
                if arity == 1:
                    for (arg_expr, _) in bank:
                        if self._size(arg_expr) < size:
                            new_expr = BSApp(op_name, [arg_expr])
                            outputs = self._evaluate_on_inputs(new_expr, io_pairs)
                            if outputs is not None:
                                new_exprs.append((new_expr, outputs))
                                # [EARLY CHECK] Check if this is a solution
                                if self._check_solution(new_expr, io_pairs):
                                    print(f"[BVH-ENUM] 🎯 Early solution found: {new_expr}")
                                    return [(new_expr, outputs)]  # Return immediately
                
                elif arity == 2:
                    for (e1, _) in bank:
                        for (e2, _) in bank:
                            # [OPTIMIZED] Allow size up to max_size+1 for binary ops with priority operands
                            is_priority = op_name in priority_ops
                            size_limit = max_size + 1 if is_priority else max_size
                            if self._size(e1) + self._size(e2) + 1 <= size_limit and self._size(e1) + self._size(e2) + 1 == size:
                                new_expr = BSApp(op_name, [e1, e2])
                                outputs = self._evaluate_on_inputs(new_expr, io_pairs)
                                if outputs is not None:
                                    new_exprs.append((new_expr, outputs))
                                    # [EARLY CHECK] Check if this is a solution
                                    if self._check_solution(new_expr, io_pairs):
                                        print(f"[BVH-ENUM] 🎯 Early solution found: {new_expr}")
                                        return [(new_expr, outputs)]
            
            bank.extend(new_exprs)
        
        return bank

    
    def _parallel_ray_search(self, bank, io_pairs, deadline, priority_ops) -> List:
        """
        [RT-INSPIRED] Parallel Ray Casting Search.
        Launch multiple search rays with different directional biases.
        Each ray explores a different region of the search space.
        """
        # Define ray directions (operator preference biases)
        rays = []
        for i in range(self.ray_count):
            if i == 0:
                # Ray 0: Prioritize single-op targets
                direction = [op for op in priority_ops if self.ops.get(op, (0,))[0] == 1]
            elif i == 1:
                # Ray 1: Prioritize binary combinations
                direction = [op for op in priority_ops if self.ops.get(op, (0,))[0] == 2]
            elif i == 2:
                # Ray 2: Reverse priority order
                direction = list(reversed(priority_ops))
            else:
                # Ray 3+: Random shuffle
                direction = list(priority_ops)
                self.rng.shuffle(direction)
            rays.append({'direction': direction, 'active': True, 'best_score': 0})
        
        # Cast rays in parallel (simulated)
        for iteration in range(50):
            if deadline and time.time() > deadline:
                break
            
            for ray in rays:
                if not ray['active']:
                    continue
                
                # Select operator based on ray direction
                if not ray['direction']:
                    ray['active'] = False
                    continue
                
                op_name = ray['direction'][iteration % len(ray['direction'])]
                if op_name not in self.ops:
                    continue
                
                arity = self.ops[op_name][0]
                
                # Build expression using bank
                if arity == 1 and bank:
                    base = self.rng.choice(bank)[0]
                    new_expr = BSApp(op_name, [base])
                elif arity == 2 and len(bank) >= 2:
                    e1 = self.rng.choice(bank)[0]
                    e2 = self.rng.choice(bank)[0]
                    new_expr = BSApp(op_name, [e1, e2])
                else:
                    continue
                
                # Check if ray hit solution
                if self._check_solution(new_expr, io_pairs):
                    print(f"[RT-RAY] ✅ Ray hit solution: {new_expr}")
                    return [(str(new_expr), new_expr, self._size(new_expr), 100.0)]
                
                # Early termination: deactivate weak rays
                score = self._partial_score(new_expr, io_pairs)
                if score > ray['best_score']:
                    ray['best_score'] = score
                elif ray['best_score'] > 50 and score < ray['best_score'] * 0.8:
                    ray['active'] = False  # Terminate underperforming ray
        
        return None

    
    def _bottom_up_enumerate(self, io_pairs, max_size=4) -> List[Tuple]:
        """
        Bottom-Up Enumeration: Generate expressions from size 1 upward.
        Returns list of (expr, [outputs for each input])
        """
        bank = []
        
        # Size 1: Base expressions
        base_exprs = [
            BSVar("n"),
            BSVal(0), BSVal(1), BSVal(2), BSVal(3),
        ]
        
        for expr in base_exprs:
            outputs = self._evaluate_on_inputs(expr, io_pairs)
            if outputs is not None:  # Type pruning: only keep valid
                bank.append((expr, outputs))
        
        # Size 2+: Apply operators
        for size in range(2, max_size + 1):
            new_exprs = []
            
            for op_name, (arity, in_types, out_types) in self.ops.items():
                if arity == 1:
                    # Unary operators
                    for (arg_expr, arg_outputs) in bank:
                        if self._size(arg_expr) < size:
                            new_expr = BSApp(op_name, [arg_expr])
                            outputs = self._evaluate_on_inputs(new_expr, io_pairs)
                            if outputs is not None:  # Type pruning
                                new_exprs.append((new_expr, outputs))
                
                elif arity == 2:
                    # Binary operators - combine two expressions
                    for i, (e1, o1) in enumerate(bank):
                        for j, (e2, o2) in enumerate(bank):
                            if self._size(e1) + self._size(e2) + 1 == size:
                                new_expr = BSApp(op_name, [e1, e2])
                                outputs = self._evaluate_on_inputs(new_expr, io_pairs)
                                if outputs is not None:
                                    new_exprs.append((new_expr, outputs))
            
            bank.extend(new_exprs)
            
            # Neural-guided ordering
            if self.nn:
                bank = self._neural_reorder(bank, io_pairs)
        
        return bank
    
    def _observational_equivalence(self, bank, io_pairs) -> List[Tuple]:
        """
        Observational Equivalence: Keep only one representative per output class.
        Two expressions are equivalent if they produce same outputs on all inputs.
        """
        output_to_expr = {}
        
        for expr, outputs in bank:
            key = tuple(str(o) for o in outputs)  # Hash outputs
            if key not in output_to_expr:
                output_to_expr[key] = (expr, outputs)
            else:
                # Keep smaller expression
                existing = output_to_expr[key][0]
                if self._size(expr) < self._size(existing):
                    output_to_expr[key] = (expr, outputs)
        
        return list(output_to_expr.values())
    
    def _mcts_expand(self, bank, io_pairs, deadline) -> List:
        """
        MCTS-guided expansion: Use UCB1 to balance exploration/exploitation.
        """
        for iteration in range(100):
            if deadline and time.time() > deadline:
                break
            
            # UCB1 selection
            best_score = -float('inf')
            best_op = None
            total_visits = sum(self.mcts_visits.values()) + 1
            
            for op_name in self.ops:
                visits = self.mcts_visits[op_name] + 1
                reward = self.mcts_rewards[op_name] / visits
                exploration = math.sqrt(2 * math.log(total_visits) / visits)
                ucb = reward + exploration
                
                if ucb > best_score:
                    best_score = ucb
                    best_op = op_name
            
            # Try expansion with selected operator
            arity = self.ops[best_op][0]
            if arity == 1 and bank:
                base_expr = self.rng.choice(bank)[0]
                new_expr = BSApp(best_op, [base_expr])
            elif arity == 2 and len(bank) >= 2:
                e1 = self.rng.choice(bank)[0]
                e2 = self.rng.choice(bank)[0]
                new_expr = BSApp(best_op, [e1, e2])
            else:
                continue
            
            # Evaluate
            self.mcts_visits[best_op] += 1
            if self._check_solution(new_expr, io_pairs):
                self.mcts_rewards[best_op] += 1.0
                return [(str(new_expr), new_expr, self._size(new_expr), 100.0)]
            else:
                # Partial reward
                score = self._partial_score(new_expr, io_pairs)
                self.mcts_rewards[best_op] += score / 100.0
        
        return None
    
    def _evaluate_on_inputs(self, expr, io_pairs) -> List:
        """Evaluate expression on all inputs, return None if type error."""
        outputs = []
        for pair in io_pairs:
            inp = pair.get("input", pair.get("n", 0))
            try:
                result = self.interp.run(expr, {"n": inp})
                if result is None:
                    return None  # Type pruning: invalid
                outputs.append(result)
            except:
                return None
        return outputs
    
    def _check_solution(self, expr, io_pairs) -> bool:
        """Check if expression solves all I/O pairs."""
        for pair in io_pairs:
            inp = pair.get("input", pair.get("n", 0))
            expected = pair.get("output", pair.get("out", None))
            try:
                result = self.interp.run(expr, {"n": inp})
                if result != expected:
                    return False
            except:
                return False
        return True
    
    def _partial_score(self, expr, io_pairs) -> float:
        """Compute partial match score."""
        correct = 0
        for pair in io_pairs:
            inp = pair.get("input", pair.get("n", 0))
            expected = pair.get("output", pair.get("out", None))
            try:
                result = self.interp.run(expr, {"n": inp})
                if result == expected:
                    correct += 1
            except:
                pass
        return (correct / len(io_pairs)) * 100 if io_pairs else 0
    
    def _find_best_partial(self, bank, io_pairs):
        """Find best partially matching expression."""
        best_score = -1
        best_expr = None
        for expr, _ in bank:
            score = self._partial_score(expr, io_pairs)
            if score > best_score:
                best_score = score
                best_expr = expr
        return best_expr
    
    def _size(self, expr) -> int:
        """Compute expression size."""
        if isinstance(expr, (BSVar, BSVal)):
            return 1
        elif isinstance(expr, BSApp):
            return 1 + sum(self._size(a) for a in expr.args)
        return 1
    
    def _neural_reorder(self, bank, io_pairs):
        """Reorder bank using neural network predictions."""
        # Use NN to score each expression's operator
        if not self.nn:
            return bank
        # Simple heuristic: prioritize expressions with high-scoring operators
        return bank  # TODO: implement neural scoring

# ==============================================================================
# STRONG RSI: Failure Pattern Analysis

# ==============================================================================
class FailureAnalyzer:
    """
    Tracks failure patterns and adjusts search strategy accordingly.
    """
    def __init__(self):
        self.failure_log: Dict[str, List[Dict]] = collections.defaultdict(list)
        self.success_log: Dict[str, int] = collections.defaultdict(int)
        self.domain_difficulty: Dict[str, float] = {}
    
    def record_failure(self, task_id: str, domain: str, attempted_ops: List[str]):
        """Record a failed synthesis attempt."""
        self.failure_log[domain].append({
            "task": task_id,
            "ops_tried": attempted_ops,
            "timestamp": time.time()
        })
    
    def record_success(self, domain: str):
        """Record a successful synthesis."""
        self.success_log[domain] += 1
    
    def analyze(self) -> Dict[str, float]:
        """
        Analyze failure patterns and return bias adjustments.
        Returns: {operator: bias_multiplier}
        """
        bias_adjustments = {}
        
        for domain, failures in self.failure_log.items():
            if not failures:
                continue
            
            total_attempts = len(failures) + self.success_log.get(domain, 0)
            failure_rate = len(failures) / max(1, total_attempts)
            
            self.domain_difficulty[domain] = failure_rate
            
            # If a domain is failing often, boost related operators
            if failure_rate > 0.5:
                if domain == "boolean":
                    for op in ["xor_op", "and_op", "or_op", "not_op", "first", "second"]:
                        bias_adjustments[op] = bias_adjustments.get(op, 1.0) * 1.5
                elif domain == "string":
                    for op in ["concat", "reverse", "slice_from", "len"]:
                        bias_adjustments[op] = bias_adjustments.get(op, 1.0) * 1.5
                elif domain == "list":
                    for op in ["reverse", "first", "second", "index"]:
                        bias_adjustments[op] = bias_adjustments.get(op, 1.0) * 1.5
        
        if bias_adjustments:
            print(f"[FailureAnalyzer] 📊 Bias adjustments: {bias_adjustments}")
        
        return bias_adjustments
    
    def get_recommended_depth(self, domain: str) -> int:
        """Get recommended search depth based on failure history."""
        difficulty = self.domain_difficulty.get(domain, 0.0)
        # Harder domains get deeper search
        if difficulty > 0.7:
            return 5
        elif difficulty > 0.4:
            return 4
        return 3
    
    def get_state(self) -> Dict:
        """Get state for checkpointing."""
        return {
            "failure_counts": {k: len(v) for k, v in self.failure_log.items()},
            "success_counts": dict(self.success_log),
            "domain_difficulty": dict(self.domain_difficulty)
        }
    
    def load_state(self, state: Dict):
        """Load state from checkpoint."""
        self.success_log = collections.defaultdict(int, state.get("success_counts", {}))
        self.domain_difficulty = state.get("domain_difficulty", {})

# ==============================================================================
# Meta-Controller (Self-Adaptive Logic)
# ==============================================================================
@dataclass
class MetaParams:
    mutation_rate: float
    crossover_prob: float
    population_size: int

class MetaController:
    """
    RSI Engine: Monitors evolutionary progress and adapts hyperparameters.
    If stagnating -> Increase exploration (Mutation).
    If progressing -> Increase exploitation (Selection/Crossover).
    """
    def __init__(self):
        self.history = []
        self.params = MetaParams(mutation_rate=0.3, crossover_prob=0.7, population_size=200)
        self.stagnation_counter = 0
        self.last_best_fitness = -1.0
        self.total_tasks_solved = 0  # [RSI] Track overall progress
        
    def reset_for_new_task(self):
        """[FIX] Reset stagnation tracking for new task - each task has independent fitness scale."""
        self.stagnation_counter = 0
        self.last_best_fitness = -1.0
        # Keep params and history for long-term learning

        
    def update(self, current_best_fitness: float):
        self.history.append(current_best_fitness)
        
        # Stagnation Detection
        if current_best_fitness <= self.last_best_fitness:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
            self.last_best_fitness = current_best_fitness
            
        # Recursive Adaptation Logic
        if self.stagnation_counter > 5:
            # Stagnating: Turbocharge Mutation (Exploration)
            self.params.mutation_rate = min(0.9, self.params.mutation_rate * 1.5)
            self.params.crossover_prob = max(0.1, self.params.crossover_prob * 0.8)
            print(f"[RSI-Meta] Stagnation detected ({self.stagnation_counter}). Increasing Mutation to {self.params.mutation_rate:.2f}")
            
            # [FIX] Adaptive threshold lowering after prolonged stagnation
            if self.stagnation_counter > 100 and self.stagnation_counter % 100 == 0:
                self.last_best_fitness *= 0.95  # Lower the bar by 5%
                print(f"[RSI-Meta] Lowering best fitness threshold to {self.last_best_fitness:.2f}")
            
            # [FIX] Hard reset after extreme stagnation
            if self.stagnation_counter > 500 and self.stagnation_counter % 500 == 0:
                print(f"[RSI-Meta] HARD RESET triggered at stagnation {self.stagnation_counter}")
                self.stagnation_counter = 0
                self.last_best_fitness = 0.0  # Reset baseline
                self.params.mutation_rate = 0.5  # Reset mutation
                self.params.crossover_prob = 0.5  # Reset crossover
            
        elif self.stagnation_counter == 0 and len(self.history) > 1:
            # Progressing: Stabilize (Exploitation)
            self.params.mutation_rate = max(0.1, self.params.mutation_rate * 0.9)
            self.params.crossover_prob = min(0.9, self.params.crossover_prob * 1.1)
            # print(f"[RSI-Meta] Progress detected. Stabilizing Mutation to {self.params.mutation_rate:.2f}")


    def get_params(self) -> MetaParams:
        return self.params

# ==============================================================================
# Neuro-Genetic Synthesizer (Island Model) - WITH PERSISTENCE
# ==============================================================================
CHECKPOINT_PATH = "rsi_checkpoint.json"

class NeuroGeneticSynthesizer:
    def __init__(self, neural_guide=None, pop_size=200, generations=20, islands=3, checkpoint_path=CHECKPOINT_PATH):
        self.guide = neural_guide
        self.meta = MetaController() # Initialize RSI Meta-Controller
        self.meta.params.population_size = pop_size # Sync initial param
        self.checkpoint_path = checkpoint_path
        
        self.pop_size = pop_size
        self.generations = generations
        self.num_islands = islands
        self.interp = NeuroInterpreter()
        self.rng = random.Random()
        self.novelty = NoveltyScorer() # Novelty detection
        self.generation_count = 0 # Track total generations for persistence
        self.learned_primitives = [] # Track learned ops
        
        # [STRONG RSI] Initialize meta-learning components
        self.library_learner = LibraryLearner()
        self.failure_analyzer = FailureAnalyzer()
        self.current_domain = None  # Track current task domain
        
        if self.guide is None:
            self.internal_nn = SimpleNN(input_dim=20, hidden_dim=16, output_dim=6, rng=self.rng)
            print("[NeuroGen] Internal Pure-Python Neural Network initialized.")
        else:
            self.internal_nn = None

        # Try loading checkpoint
        if self.internal_nn and os.path.exists(self.checkpoint_path):
            self.load_checkpoint(self.checkpoint_path)

        self.atoms = [BSVar('n'), BSVal(0), BSVal(1), BSVal(2), BSVal(3)]
        self.ops = list(NeuroInterpreter.PRIMS.keys())
        # [FIX] Track arity of operators to prevent generation errors
        self.op_arities = {
            'add': 2, 'sub': 2, 'mul': 2, 'div': 2, 'mod': 2, 'if_gt': 4,
            'concat': 2, 'slice_from': 2, 'len': 1, 'reverse': 1, 'eq': 2,
            'and_op': 2, 'or_op': 2, 'xor_op': 2, 'not_op': 1,
            # New primitives for Boolean domain
            'first': 1, 'second': 1, 'index': 2, 'if_eq': 4
        }
        self.structural_bias = {}
        
        # [COMPOUNDING RSI] Load library functions as primitives
        self._load_library_as_primitives()
    
    # ==========================================================================
    # COMPOUNDING RSI: Load Learned Library as Primitives
    # ==========================================================================
    RSI_LIBRARY_PATH = "rsi_modifier_state.json"
    
    def _load_library_as_primitives(self):
        """Load saved library functions and register them as usable primitives.
        This enables compounding improvement: past discoveries accelerate future search."""
        if not os.path.exists(self.RSI_LIBRARY_PATH):
            return
        
        try:
            with open(self.RSI_LIBRARY_PATH, 'r') as f:
                state = json.load(f)
            
            library = state.get('library', [])
            loaded_count = 0
            
            for code_str in library:
                # Extract function name from code
                import re
                match = re.search(r'def\s+(\w+)\s*\(', code_str)
                if not match:
                    continue
                    
                fn_name = match.group(1)
                
                # Skip if already registered
                if fn_name in self.ops or fn_name in NeuroInterpreter.PRIMS:
                    continue
                
                # Try to compile and register the function
                try:
                    local_ns = {}
                    # Provide safe execution context
                    safe_globals = {
                        'n': 0, 'len': len, 'reverse': lambda x: x[::-1] if hasattr(x, '__getitem__') else x,
                        'add': lambda a, b: a + b, 'sub': lambda a, b: a - b,
                        'mul': lambda a, b: a * b, 'div': lambda a, b: a // b if b != 0 else 0,
                    }
                    exec(code_str, safe_globals, local_ns)
                    
                    if fn_name in local_ns and callable(local_ns[fn_name]):
                        # Register as primitive
                        NeuroInterpreter.PRIMS[fn_name] = local_ns[fn_name]
                        self.ops.append(fn_name)
                        
                        # Determine arity from function signature
                        import inspect
                        try:
                            sig = inspect.signature(local_ns[fn_name])
                            arity = len(sig.parameters)
                            # [FIX] Skip functions with arity 0 (causes randrange crash)
                            if arity < 1:
                                continue
                            self.op_arities[fn_name] = arity
                        except:
                            self.op_arities[fn_name] = 1  # Default arity

                        
                        loaded_count += 1
                except Exception:
                    pass  # Skip invalid functions silently
            
            if loaded_count > 0:
                print(f"[RSI-Compound] Loaded {loaded_count} library functions as search primitives")
                
        except Exception as e:
            print(f"[RSI-Compound] Failed to load library: {e}")

    
    # ==========================================================================
    # PERSISTENCE: Save / Load Checkpoint
    # ==========================================================================
    def save_checkpoint(self, path: str = None):
        """Save NN weights and meta-state to JSON for persistent learning."""
        path = path or self.checkpoint_path
        if not self.internal_nn:
            return
        
        checkpoint = {
            "W1": self.internal_nn.W1,
            "W2": self.internal_nn.W2,
            "b1": self.internal_nn.b1,
            "b2": self.internal_nn.b2,
            "learned_primitives": self.learned_primitives,
            "generation_count": self.generation_count,
            "meta_stagnation": self.meta.stagnation_counter,
            "meta_mutation_rate": self.meta.params.mutation_rate,
            # [STRONG RSI] Meta-learning state
            "library_learner": self.library_learner.get_state(),
            "failure_analyzer": self.failure_analyzer.get_state(),
        }
        
        try:
            with open(path, "w") as f:
                json.dump(checkpoint, f)
            print(f"[RSI-Persist] ✅ Checkpoint saved to {path} (Gen: {self.generation_count}, Lib: {len(self.library_learner.learned_primitives)})")
        except Exception as e:
            print(f"[RSI-Persist] ❌ Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, path: str = None):
        """Load NN weights and meta-state from JSON."""
        path = path or self.checkpoint_path
        if not self.internal_nn or not os.path.exists(path):
            return False
            
        try:
            with open(path, "r") as f:
                checkpoint = json.load(f)
            
            self.internal_nn.W1 = checkpoint["W1"]
            self.internal_nn.W2 = checkpoint["W2"]
            self.internal_nn.b1 = checkpoint["b1"]
            self.internal_nn.b2 = checkpoint["b2"]
            self.learned_primitives = checkpoint.get("learned_primitives", [])
            self.generation_count = checkpoint.get("generation_count", 0)
            self.meta.stagnation_counter = checkpoint.get("meta_stagnation", 0)
            self.meta.params.mutation_rate = checkpoint.get("meta_mutation_rate", 0.3)
            
            # [STRONG RSI] Load meta-learning state
            if "library_learner" in checkpoint:
                self.library_learner.load_state(checkpoint["library_learner"])
            if "failure_analyzer" in checkpoint:
                self.failure_analyzer.load_state(checkpoint["failure_analyzer"])
            
            print(f"[RSI-Persist] 🔄 Checkpoint loaded from {path} (Gen: {self.generation_count}, Lib: {len(self.library_learner.learned_primitives)})")
            return True
        except Exception as e:
            print(f"[RSI-Persist] ❌ Failed to load checkpoint: {e}")
            return False


    def register_primitive(self, name: str, func: Callable):
        self.interp.register_primitive(name, func)
        if name not in self.ops:
            self.ops.append(name)
            # Inspect arity using inspect signature or simple heuristic
            try:
                import inspect
                sig = inspect.signature(func)
                arity = len(sig.parameters)
                # Handle *args (variadic) -> assume unary wrapper for now as per Systemtest.py
                # Systemtest.py creates: lambda *args: interp.run(expr_ast, {'n': args[0]...})
                # If variadic, we default to 1 for "OpN(n)" usage pattern
                for param in sig.parameters.values():
                   if param.kind == inspect.Parameter.VAR_POSITIONAL:
                       arity = 1
                       break
            except:
                arity = 1 # Default to unary for lambdas if inspect fails
                
            self.op_arities[name] = arity
            print(f"[NeuroGen] Registered new primitive: {name} (arity={arity})")

    def feedback(self, code: str, score: float):
        """
        GENUINE Feedback Loop:
        1. Parses the successful code to find WHICH operators were used.
        2. Trains the neural network to output higher probabilities for THOSE operators.
        """
        if not self.internal_nn or score < 0.01:
            return

        # 1. Parse used operators from code string (Simple Tokenization)
        # e.g., "add(mul(n, 2), 1)" -> ['add', 'mul']
        used_ops = set()
        for op in self.ops:
            if op + "(" in code: # Simple heuristic: "func_name("
                used_ops.add(op)
        
        if not used_ops:
             print(f"[NeuroGen] 🧠 Genuine Feedback | Code: '{code}' | No trainable operators found (Trivial Solution).")
             return


        # 2. Extract context features (The 'State')
        # [GENUINE-FIX] Use the CACHED features from the synthesis step.
        # This ensures we are training P(Ops | IO Context), not P(Ops | Code Length).
        # Consistency is key for real learning.
        if hasattr(self, 'last_features') and self.last_features:
            feature_vector = self.last_features
            # print(f"[NeuroGen] Using cached I/O features for training (Consistency Check OK)")
        else:
            # Fallback (Should not happen in correct flow, but safe default)
            feature_vector = [0.0] * self.internal_nn.input_dim
        
        # 3. Forward Pass (Activate Network with SAME context)
        self.internal_nn.forward(feature_vector)

        
        # 4. Dynamic Backprop (Train on ACTUAL used ops)
        # Map op names to output indices [add, mul, sub, div, mod, if_gt, ...]
        op_keys = ['add', 'mul', 'sub', 'div', 'mod', 'if_gt']
        
        print(f"[NeuroGen] 🧠 Genuine Feedback | Solution uses: {used_ops}")
        
        for op in used_ops:
            if op in op_keys:
                target_idx = op_keys.index(op)
                self.internal_nn.train(target_idx)
        
        # [STRONG RSI] Record solution for library learning
        if hasattr(self, 'last_expr') and self.last_expr:
            self.library_learner.record_solution(code, self.last_expr)
        
        # [STRONG RSI] Record success for failure analysis
        if self.current_domain:
            self.failure_analyzer.record_success(self.current_domain)
        
        # [STRONG RSI] Periodic primitive discovery (every 10 successes)
        self.generation_count += 1
        if self.generation_count % 10 == 0:
            discoveries = self.library_learner.discover_primitives()
            for name, code_pattern in discoveries:
                # [ACTUAL INJECTION] Create callable and register as primitive
                try:
                    prim_fn = self.library_learner.create_primitive_function(code_pattern, self.interp)
                    self.register_primitive(name, prim_fn)
                    self.learned_primitives.append(name)
                    print(f"[STRONG-RSI] 🧬 DSL EXPANDED: {name} = {code_pattern} -> INJECTED AS PRIMITIVE!")
                    
                    # Track usage for validation
                    self.primitive_usage_count = getattr(self, 'primitive_usage_count', {})
                    self.primitive_usage_count[name] = 0
                except Exception as e:
                    print(f"[STRONG-RSI] ❌ Failed to inject {name}: {e}")
        
        # [STRONG RSI] Track usage of learned primitives
        for prim in self.learned_primitives:
            if prim + '(' in code:
                self.primitive_usage_count = getattr(self, 'primitive_usage_count', {})
                self.primitive_usage_count[prim] = self.primitive_usage_count.get(prim, 0) + 1
                print(f"[RSI-Validation] 📊 {prim} used! (Total: {self.primitive_usage_count[prim]})")


        
        # [PERSISTENCE] Auto-save checkpoint after learning
        if self.generation_count % 5 == 0:
            self.save_checkpoint()


    def synthesize(self, io_pairs: List[Dict[str, Any]], deadline=None, task_id="", task_params=None, **kwargs) -> List[Tuple[str, Expr, float, float]]:

        # [FIX] Reset stagnation tracking for each new task
        self.meta.reset_for_new_task()

        # [STRONG RSI] Track current domain for failure analysis
        # Try to detect domain from task_id or io_pairs
        domain = kwargs.get("domain", None)
        if not domain and task_id:
            if "str" in task_id or "string" in task_id:
                domain = "string"
            elif "list" in task_id:
                domain = "list"
            elif "bool" in task_id:
                domain = "boolean"
        self.current_domain = domain


        # 1. Neural Guidance (Priors)
        priors = {op: 1.0 for op in self.ops} 
        if 'mod' in priors: priors['mod'] = 0.5
        if 'if_gt' in priors: priors['if_gt'] = 0.1
        
        # [STRONG RSI] Apply failure-based bias adjustments
        failure_biases = self.failure_analyzer.analyze()
        for op, bias in failure_biases.items():
            if op in priors:
                priors[op] *= bias
        
        if self.guide:
            learned_priors = self.guide.get_priors(io_pairs)
            if learned_priors: priors.update(learned_priors)
        elif self.internal_nn:
            features = self._extract_features(io_pairs)
            # [GENUINE-FIX] Cache features for consistent feedback training
            self.last_features = features 
            
            nn_probs = self.internal_nn.forward(features)
            op_keys = ['add', 'mul', 'sub', 'div', 'mod', 'if_gt']
            for i, op in enumerate(op_keys):
                if i < len(nn_probs) and op in priors: priors[op] = nn_probs[i] * 5.0
            self.internal_nn.mutate(self.rng, rate=0.01)


        # Apply Structural Bias
        for op, bias in self.structural_bias.items():
            if op in priors: priors[op] *= bias

        total_p = sum(priors.values())
        op_probs = {k: v/total_p for k, v in priors.items()}

        # 2. Initialize Islands
        island_pop_size = self.pop_size // self.num_islands
        
        # [HONEST RSI] Template seeding DISABLED for genuine capability testing
        # No shortcuts - system must discover patterns on its own
        template_seeds = []
        # if self.current_domain == "boolean":  # DISABLED - this was a shortcut
        #     template_seeds = [...]
        
        # Build islands WITHOUT template hints
        islands = []
        for _ in range(self.num_islands):
            island = [self._random_expr(2, op_probs) for _ in range(island_pop_size)]
            islands.append(island)

        
        best_solution = None
        best_fitness = -1.0

        for gen in range(self.generations):
            if deadline and time.time() > deadline: break
            
            # --- META-CONTROLLER UPDATE ---
            # 1. Update Meta-Controller with current best fitness
            if best_fitness > 0:
                self.meta.update(best_fitness)
            
            # 2. Retrieve Dynamic Hyperparameters
            meta_params = self.meta.get_params()
            current_mutation_rate = meta_params.mutation_rate
            current_crossover_prob = meta_params.crossover_prob
            # ------------------------------

            # Migration (Ring Topology)
            if gen > 0 and gen % 5 == 0:
                for i in range(self.num_islands):
                    target_i = (i + 1) % self.num_islands
                    # Move top 5%
                    migrants = sorted(islands[i], key=lambda e: self._fitness(e, io_pairs, fast=True), reverse=True)[:int(island_pop_size*0.05)]
                    # Replace worst in target
                    islands[target_i].sort(key=lambda e: self._fitness(e, io_pairs, fast=True))
                    islands[target_i] = migrants + islands[target_i][len(migrants):]
                    # print(f"  [Island] Migration {i}->{target_i} ({len(migrants)} units)")

            # Evolve each island
            for i in range(self.num_islands):
                scored_pop = []
                for expr in islands[i]:
                    raw_fit = self._fitness(expr, io_pairs)
                    nov_score = self.novelty.score(expr)
                    final_fit = raw_fit + (nov_score * 5.0) # Bonus for novelty
                    
                    scored_pop.append((final_fit, expr, raw_fit))
                    
                    if raw_fit >= 100.0:
                        # [STRONG RSI] Store expr for library learning
                        self.last_expr = expr
                        return [(str(expr), expr, self._size(expr), raw_fit)]

                scored_pop.sort(key=lambda x: x[0], reverse=True)
                
                # Global Best Tracking
                if scored_pop[0][2] > best_fitness:
                    best_fitness = scored_pop[0][2]
                    best_solution = (scored_pop[0][0], scored_pop[0][1], scored_pop[0][2])

                # Selection
                next_gen = [p[1] for p in scored_pop[:5]] # Elitism
                while len(next_gen) < island_pop_size:
                    p1 = self._tournament(scored_pop)
                    p2 = self._tournament(scored_pop)
                    # DYNAMIC CROSSOVER PROBABILITY
                    child = self._crossover(p1, p2) if self.rng.random() < current_crossover_prob else p1
                    # DYNAMIC MUTATION RATE
                    if self.rng.random() < current_mutation_rate: child = self._mutate(child, op_probs)
                    next_gen.append(child)
                islands[i] = next_gen

        if best_solution:
            # [STRONG RSI] Store expr for library learning
            self.last_expr = best_solution[1]
            print(f"[NeuroGen] Best fitness: {best_fitness:.2f}")
            return [(str(best_solution[1]), best_solution[1], self._size(best_solution[1]), best_fitness)]
        else:
            # [STRONG RSI] Record failure for analysis
            if self.current_domain:
                self.failure_analyzer.record_failure(task_id, self.current_domain, list(self.ops[:10]))
        return []

    def _extract_features(self, io_pairs):
        """
        GENUINE Feature Extraction (Multi-Domain):
        Computes statistical properties of the I/O pairs for ANY domain (Number, String, List, Bool).
        """
        features = [0.0] * 20 # Fixed size 20
        
        if not io_pairs: return features
        
        try:
            # Flatten inputs/outputs for analysis
            inputs = [p['input'] for p in io_pairs]
            outputs = [p['output'] for p in io_pairs]
            
            if not inputs: return features

            # 1. Type Encoding (One-hotish)
            in_type = type(inputs[0])
            if in_type == int or in_type == float: features[10] = 1.0
            elif in_type == str: features[11] = 1.0
            elif in_type == list: features[12] = 1.0
            elif in_type == bool: features[13] = 1.0

            # 2. Domain-Specific Stats
            if features[11] > 0.5: # String Domain
                lens_in = [len(s) for s in inputs if isinstance(s, str)]
                lens_out = [len(s) for s in outputs if isinstance(s, str)]
                features[0] = sum(lens_in)/len(lens_in) if lens_in else 0
                features[1] = sum(lens_out)/len(lens_out) if lens_out else 0
                features[2] = 1.0 if any(i in o for i, o in zip(inputs, outputs)) else 0.0 # Substring?
                features[3] = 1.0 if any(i[::-1] == o for i, o in zip(inputs, outputs)) else 0.0 # Reverse?
                
            elif features[12] > 0.5: # List Domain
                lens_in = [len(x) for x in inputs if isinstance(x, list)]
                features[0] = sum(lens_in)/len(lens_in) if lens_in else 0
                # Check sorted
                features[4] = 1.0 if any(sorted(i) == o for i, o in zip(inputs, outputs)) else 0.0
                # Check reverse
                features[5] = 1.0 if any(list(reversed(i)) == o for i, o in zip(inputs, outputs)) else 0.0
                
            elif features[13] > 0.5: # Boolean Domain (Tuple inputs likely)
                # Count True/False ratios
                flat_in = []
                for x in inputs:
                    if isinstance(x, (list, tuple)): flat_in.extend(x)
                    else: flat_in.append(x)
                features[6] = sum(1 for x in flat_in if x) / len(flat_in) if flat_in else 0
                features[7] = sum(1 for x in outputs if x) / len(outputs) if outputs else 0

            else: # Numeric Domain (Legacy)
                nums_in = [float(x) for x in inputs if isinstance(x, (int, float))]
                nums_out = [float(x) for x in outputs if isinstance(x, (int, float))]
                if nums_in and nums_out:
                    features[0] = min(nums_in)
                    features[1] = max(nums_in)
                    features[2] = sum(nums_in) / len(nums_in)
                    features[5] = sum(nums_out) / len(nums_out)
                    if abs(features[2]) > 1e-9:
                        features[6] = features[5] / features[2] # Growth
            
            # Global Identity check
            features[9] = 1.0 if any(i == o for i, o in zip(inputs, outputs)) else 0.0

        except Exception:
            # Robust fallback
            pass
            
        return features


    def _fitness(self, expr, ios, fast=False):
        # 1. Try Rust Acceleration
        jit_score = None
        if HAS_RUST_VM and self.num_islands > 0: # Ensure we are in a valid state
            try:
                compiler = RustCompiler()
                instructions = compiler.compile(expr)
                if instructions:
                    # Execute on Rust VM
                    # Note: We need a fresh VM or reuse one. Creating generic VM is cheap?
                    # Systemtest.py uses VirtualMachine(max_steps=400, ...)
                    # We can instantiate rs_machine.VirtualMachine directly.
                    # rs_machine.VirtualMachine(max_steps, mem_size, stack_limit)
                    vm = rs_machine.VirtualMachine(100, 64, 16)
                    
                    score = 0
                    for io in ios:
                        # Prepare input memory. 'n' is mapped to mem[0].
                        # Rust VM execute takes inputs list.
                        inp_val = float(io['input'])
                        # If input is list, use it; if scalar, wrap.
                        if isinstance(io['input'], list):
                             # Not supported by simple compiler yet (scalar assumption)
                             raise ValueError("List input not supported in JIT")
                        
                        st = vm.execute(instructions, [inp_val])
                        
                        # Result in regs[0] (target_reg 0)
                        # We verify clean halt?
                        # if not st.halted_cleanly: score -= ...?
                        # NeuroGen simple fitness just checks equality.
                        
                        # rs_machine.ExecutionState.regs is a list
                        out = st.regs[0]
                        
                        expected = io['output']
                        if abs(out - expected) < 1e-9:
                             score += 1
                        elif not fast:
                             diff = abs(out - expected)
                             if diff < 100: score += 1.0 / (1.0 + diff)
                             
                    return (score / len(ios)) * 100.0
            except Exception as e:
                # Fallback to Python if compilation/execution fails (e.g. MOD op, depth)
                pass

        # 2. Python Fallback
        score = 0
        for io in ios:
            out = self.interp.run(expr, { 'n': io['input'] })
            if out == io['output']: score += 1
            else:
                 if not fast and isinstance(out, (int, float)) and isinstance(io['output'], (int, float)):
                     diff = abs(out - io['output'])
                     if diff < 100: score += 1.0 / (1.0 + diff)
        return (score / len(ios)) * 100.0

    def _random_expr(self, depth, op_probs):
        if depth <= 0 or self.rng.random() < 0.3:
            return self.rng.choice(self.atoms)

        # Choose op based on Neural Priors
        op = self.rng.choices(list(op_probs.keys()), weights=list(op_probs.values()))[0]

        # Dynamic Arity Check
        arity = self.op_arities.get(op, 2)
        args = tuple(self._random_expr(depth-1, op_probs) for _ in range(arity))
        return BSApp(op, args)



    def _tournament(self, scored_pop):
        # Pick k random, return best
        k = 5
        candidates = self.rng.sample(scored_pop, k)
        return max(candidates, key=lambda x: x[0])[1]

    def _crossover(self, p1, p2):
        # Subtree Exchange
        if isinstance(p1, BSApp) and isinstance(p2, BSApp) and self.rng.random() < 0.5:
            # Swap arguments
            new_args = list(p1.args)
            idx = self.rng.randint(0, len(new_args)-1)
            new_args[idx] = p2 # Graft p2 onto p1
            return BSApp(p1.func, tuple(new_args))
        return p1 # Fallback

    def _mutate(self, p, op_probs):
        # Point Mutation or Subtree Regrowth
        if self.rng.random() < 0.5:
            # Regrowth
            return self._random_expr(2, op_probs)
        else:
            # Op mutation
            if isinstance(p, BSApp):
                new_op = self.rng.choices(list(op_probs.keys()), weights=list(op_probs.values()))[0]
                arity = self.op_arities.get(new_op, 2)
                current_arity = self.op_arities.get(p.func, 2)

                if arity == current_arity:
                    return BSApp(new_op, p.args)
        return p

    def _size(self, expr):
        if isinstance(expr, BSApp):
            return 1 + sum(self._size(a) for a in expr.args)
        return 1
