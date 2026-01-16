"""
NEURO-GENETIC SYNTHESIZER (Refactored for Safe & Genuine RSI)
Combines Evolutionary Search with Neural Guidance in a Sandboxed Environment.
"""
import ast
import random
import time
import math
import json
import collections
import hashlib
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Tuple, Optional, Callable, Set, Union
try:
    import numpy as np
except ImportError:
    np = None

# ==============================================================================
# CONFIGURATION
# ==============================================================================
MAX_GAS = 10000  # Maximum operations per execution (increased for complex algorithms)
MAX_RECURSION_DEPTH = 50  # Maximum recursion depth for safe bounded recursion
MAX_LIST_SIZE = 100  # Maximum list size for bounded iteration
REGISTRY_FILE = "rsi_primitive_registry.json"

# ==============================================================================
# I. SAFE INTERPRETER (Sandboxed AST Execution)
# ==============================================================================
class SafeInterpreter(ast.NodeVisitor):
    """
    Executes Python AST nodes in a strict sandbox.
    No exec() or eval(). No access to strict globals.
    """
    def __init__(self, primitives: Dict[str, Callable]):
        self.primitives = primitives
        self.gas = 0
        self.local_env = {}

    def run(self, node: Union[ast.AST, str], env: Dict[str, Any], gas_limit: int = MAX_GAS) -> Any:
        self.gas = gas_limit
        self.local_env = env
        
        if isinstance(node, str):
            try:
                node = ast.parse(node, mode='eval').body
            except SyntaxError as e:
                return {"__error__": "SyntaxError", "msg": str(e)}
                
        try:
            return self.visit(node)
        except Exception as e:
            # [DIAGNOSTIC] Do not swallow exceptions. Return them as data.
            return {"__error__": type(e).__name__, "msg": str(e)}

    def _check_gas(self):
        self.gas -= 1
        if self.gas <= 0:
            raise RecursionError("Gas limit exceeded")

    def visit_Constant(self, node):
        self._check_gas()
        return node.value

    # ... (other visit methods unchanged) ...

# ... (jump to PrimitiveValidator) ...

    def validate(self, name: str, ast_node: Any, validation_ios: List[Dict]) -> bool:
        """
        Runs the primitive against known I/O examples to ensure correctness.
        For a primitive 'concept_1(n)', we check if it correctly transforms input -> output.
        """
        if not validation_ios:
            return True # No tests provided, soft accept (or reject depending on strictness)
            
        print(f"[Validator] Running regression suite for {name} ({len(validation_ios)} tests)...")
        passes = 0
        for io in validation_ios:
            # Prepare env
            env = {'n': io['input']}
            try:
                res = self.interpreter.run(ast_node, env)
                if res == io['output']:
                    passes += 1
                elif passes == 0: # Log first failure for diagnostics
                     try:
                         # Safe type name extraction
                         type_in = type(io['input']).__name__
                         type_exp = type(io['output']).__name__
                         type_got = type(res).__name__
                         
                         # Check if result is an error dict
                         if isinstance(res, dict) and "__error__" in res:
                             got_str = f"EXCEPTION({res['__error__']}: {res['msg']})"
                         else:
                             got_str = str(res)
                             if len(got_str) > 100: got_str = got_str[:100] + "..."
                             
                         print(f"  [Validator-Diag] Fail: Input={io['input']} ({type_in}) | Expected={io['output']} ({type_exp}) | Got={got_str} ({type_got})")
                     except:
                         print(f"  [Validator-Diag] Fail: Input={io['input']} | Expected={io['output']} | Got={res}")
            except Exception as e:
                if passes == 0:
                     print(f"  [Validator-Diag] Exception: {str(e)[:100]}")
                pass
        
        score = passes / len(validation_ios)
        if score == 1.0:
            print(f"[Validator] {name} passed regression suite (100%).")
            return True
        else:
            print(f"[Validator] {name} failed regression suite (Score: {score:.2f}). Rejected.")
            return False

class LibraryManager:
    def __init__(self, registry_path=REGISTRY_FILE):
        self.registry_path = registry_path
        self.primitives: Dict[str, PrimitiveNode] = {}
        self.hasher = SemanticHasher()
        
        # Load Base Primitives (Level 0)
        self._register_base_primitives()
        
        # Load Persisted Primitives
        self.load_registry()

    def _register_base_primitives(self):
        # =======================================================================
        # BASE PRIMITIVES - Level 0
        # Extended for real algorithm discovery (sorting, searching, recursion)
        # =======================================================================
        base_funcs = {
            # --- Arithmetic ---
            'add': (lambda a, b: a + b, 0),
            'sub': (lambda a, b: a - b, 0),
            'mul': (lambda a, b: a * b, 0),
            'div': (lambda a, b: int(a / b) if b != 0 else 0, 0),
            'mod': (lambda a, b: a % b if b != 0 else 0, 0),
            'neg': (lambda a: -a, 0),
            'abs_val': (lambda a: abs(a) if isinstance(a, (int, float)) else a, 0),
            'min_val': (lambda a, b: min(a, b), 0),
            'max_val': (lambda a, b: max(a, b), 0),
            
            # --- Comparison Operators ---
            'eq': (lambda a, b: a == b, 0),
            'neq': (lambda a, b: a != b, 0),
            'lt': (lambda a, b: a < b, 0),
            'gt': (lambda a, b: a > b, 0),
            'lte': (lambda a, b: a <= b, 0),
            'gte': (lambda a, b: a >= b, 0),
            
            # --- Boolean Logic ---
            'not_op': (lambda a: not a, 0),
            'and_op': (lambda a, b: a and b, 0),
            'or_op': (lambda a, b: a or b, 0),
            
            # --- Conditional ---
            'if_then_else': (lambda cond, then_val, else_val: then_val if cond else else_val, 0),
            'if_gt': (lambda a, b, c, d: c if a > b else d, 0),
            'if_lt': (lambda a, b, c, d: c if a < b else d, 0),
            
            # --- List/String Access ---
            'len': (lambda x: len(x) if isinstance(x, (str, list, tuple)) else 0, 0),
            'first': (lambda x: x[0] if isinstance(x, (list, tuple, str)) and len(x) > 0 else None, 0),
            'last': (lambda x: x[-1] if isinstance(x, (list, tuple, str)) and len(x) > 0 else None, 0),
            'get': (lambda lst, i: lst[int(i)] if isinstance(lst, (list, str)) and 0 <= int(i) < len(lst) else None, 0),
            'tail': (lambda x: x[1:] if isinstance(x, (list, str)) and len(x) > 0 else ([] if isinstance(x, list) else ''), 0),
            'init': (lambda x: x[:-1] if isinstance(x, (list, str)) and len(x) > 0 else ([] if isinstance(x, list) else ''), 0),
            'reverse': (lambda x: x[::-1] if isinstance(x, (str, list)) else x, 0),
            'is_empty': (lambda x: len(x) == 0 if isinstance(x, (list, str, tuple)) else True, 0),
            
            # --- List Construction ---
            'cons': (lambda x, lst: [x] + lst if isinstance(lst, list) else [x], 0),
            'snoc': (lambda lst, x: lst + [x] if isinstance(lst, list) else [x], 0),
            'concat': (lambda a, b: a + b if isinstance(a, (list, str)) and isinstance(b, (list, str)) else a, 0),
            'take': (lambda n, lst: lst[:min(int(n), len(lst))] if isinstance(lst, (list, str)) else lst, 0),
            'drop': (lambda n, lst: lst[min(int(n), len(lst)):] if isinstance(lst, (list, str)) else lst, 0),
            'slice_list': (lambda lst, s, e: lst[int(s):int(e)] if isinstance(lst, (list, str)) else lst, 0),
            'singleton': (lambda x: [x], 0),
            'range_list': (lambda n: list(range(min(max(0, int(n)), 100))), 0),  # Bounded to 100
            
            # --- List Transformation (Bounded Iteration - Max 100) ---
            'map_double': (lambda lst: [x * 2 for x in lst[:100]] if isinstance(lst, list) else lst, 0),
            'map_square': (lambda lst: [x * x for x in lst[:100]] if isinstance(lst, list) else lst, 0),
            'map_negate': (lambda lst: [-x for x in lst[:100]] if isinstance(lst, list) else lst, 0),
            'filter_positive': (lambda lst: [x for x in lst[:100] if x > 0] if isinstance(lst, list) else lst, 0),
            'filter_negative': (lambda lst: [x for x in lst[:100] if x < 0] if isinstance(lst, list) else lst, 0),
            'filter_even': (lambda lst: [x for x in lst[:100] if x % 2 == 0] if isinstance(lst, list) else lst, 0),
            'filter_odd': (lambda lst: [x for x in lst[:100] if x % 2 == 1] if isinstance(lst, list) else lst, 0),
            
            # --- Aggregation (Bounded - Max 100) ---
            'sum_list': (lambda lst: sum(lst[:100]) if isinstance(lst, list) else 0, 0),
            'prod_list': (lambda lst: self._safe_product(lst[:100]) if isinstance(lst, list) else 1, 0),
            'min_list': (lambda lst: min(lst[:100]) if isinstance(lst, list) and len(lst) > 0 else None, 0),
            'max_list': (lambda lst: max(lst[:100]) if isinstance(lst, list) and len(lst) > 0 else None, 0),
            'count_list': (lambda lst: len(lst) if isinstance(lst, list) else 0, 0),
            
            # --- Sorting (Built-in, bounded) ---
            'sort_asc': (lambda lst: sorted(lst[:100]) if isinstance(lst, list) else lst, 0),
            'sort_desc': (lambda lst: sorted(lst[:100], reverse=True) if isinstance(lst, list) else lst, 0),
            
            # --- Search ---
            'elem_in': (lambda x, lst: x in lst if isinstance(lst, (list, str)) else False, 0),
            'index_of': (lambda x, lst: lst.index(x) if isinstance(lst, list) and x in lst else -1, 0),
            
            # --- Insertion Sort Building Blocks ---
            'insert_sorted': (lambda x, lst: self._insert_sorted(x, lst), 0),
            
            # --- Merge Sort Building Blocks ---
            'split_half': (lambda lst: (lst[:len(lst)//2], lst[len(lst)//2:]) if isinstance(lst, list) else ([], []), 0),
            'merge_sorted': (lambda a, b: self._merge_sorted(a, b), 0),
            
            # --- Higher-Order Functions (Genuine Functional Primitives) ---
            'map_fn': (lambda fn_name, lst: self._safe_map(fn_name, lst), 0),
            'filter_fn': (lambda fn_name, lst: self._safe_filter(fn_name, lst), 0),
            'fold_fn': (lambda fn_name, init, lst: self._safe_fold(fn_name, init, lst), 0),
            'recurse_fn': (lambda fn_name, arg: self._safe_recurse(fn_name, arg), 0),
        }
        
        self.runtime_primitives = {}  # Callable map for Interpreter
        for name, (func, level) in base_funcs.items():
            self.runtime_primitives[name] = func
            self.primitives[name] = PrimitiveNode(name=name, code="<native>", level=0, usage_count=100)

    def _safe_product(self, lst: list) -> int:
        """Compute product of list elements with overflow protection."""
        if not lst:
            return 1
        result = 1
        for x in lst[:100]:  # Bounded iteration
            result *= x
            if abs(result) > 10**15:  # Overflow protection
                return result
        return result

    def _insert_sorted(self, x, lst: list) -> list:
        """Insert element x into a sorted list maintaining order."""
        if not isinstance(lst, list):
            return [x]
        if len(lst) > 100:  # Safety bound
            lst = lst[:100]
        result = []
        inserted = False
        for elem in lst:
            if not inserted and x <= elem:
                result.append(x)
                inserted = True
            result.append(elem)
        if not inserted:
            result.append(x)
        return result

    def _merge_sorted(self, a: list, b: list) -> list:
        """Merge two sorted lists into one sorted list."""
        if not isinstance(a, list):
            a = []
        if not isinstance(b, list):
            b = []
        # Safety bounds
        a = a[:50]
        b = b[:50]
        result = []
        i, j = 0, 0
        while i < len(a) and j < len(b):
            if a[i] <= b[j]:
                result.append(a[i])
                i += 1
            else:
                result.append(b[j])
                j += 1
        result.extend(a[i:])
        result.extend(b[j:])
        return result

    def _safe_recurse(self, func_name: str, arg, depth: int = 0, max_depth: int = 50):
        """
        Bounded recursion with depth limit (Y-combinator pattern).
        Applies a function recursively with a callback for self-reference.
        """
        if depth >= max_depth:
            return arg  # Base case fallback
        if func_name not in self.runtime_primitives:
            return arg
        fn = self.runtime_primitives[func_name]
        try:
            # The function should accept (arg, recurse_callback)
            result = fn(arg, lambda x: self._safe_recurse(func_name, x, depth + 1, max_depth))
            return result
        except TypeError:
            # If function doesn't accept callback, just apply once
            return fn(arg)

    def _safe_map(self, func_name: str, lst: list) -> list:
        """Apply a primitive to each element, max 100 iterations."""
        if not isinstance(lst, list) or len(lst) > 100:
            return lst[:100] if isinstance(lst, list) else []
        if func_name not in self.runtime_primitives:
            return lst
        fn = self.runtime_primitives[func_name]
        try:
            return [fn(x) for x in lst[:100]]
        except:
            return lst

    def _safe_filter(self, func_name: str, lst: list) -> list:
        """Filter list by predicate primitive, max 100 iterations."""
        if not isinstance(lst, list) or len(lst) > 100:
            return lst[:100] if isinstance(lst, list) else []
        if func_name not in self.runtime_primitives:
            return lst
        fn = self.runtime_primitives[func_name]
        try:
            return [x for x in lst[:100] if fn(x)]
        except:
            return lst

    def _safe_fold(self, func_name: str, init, lst: list):
        """Left fold with max 100 iterations."""
        if not isinstance(lst, list) or len(lst) > 100:
            return init
        if func_name not in self.runtime_primitives:
            return init
        fn = self.runtime_primitives[func_name]
        acc = init
        try:
            for x in lst[:100]:
                acc = fn(acc, x)
            return acc
        except:
            return init

    def register_new_primitive(self, name: str, code: str, interpreter: SafeInterpreter, validation_ios: List[Dict] = None) -> bool:
        """
        Attempts to register a new primitive.
        CHECKS:
        1. Semantic Uniqueness
        2. Valid DAG (no circular deps, level check)
        3. Regression Validation (HotSwap Pattern)
        """
        # [RSI-Fix] Semantic De-Bloating (No Fake Complexity)
        if self._is_bloated(code):
            print(f"[Library] Rejecting {name}: Detected syntactic bloat (e.g. reverse(reverse(...)))")
            return False
            
        # 1. Parse & Hash
        s_hash = self.hasher.hash_code(code)
        
        # Check Uniqueness
        for p in self.primitives.values():
            if p.semantic_hash == s_hash and p.name != name:
                print(f"[Library] Rejecting {name}: Semantically identical to {p.name}")
                return False
        
        # 2. Analyze Dependencies for DAG Level
        try:
            tree = ast.parse(code)
            deps = []
            max_dep_level = -1
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    fname = node.func.id
                    if fname not in self.primitives:
                        print(f"[Library] Rejecting {name}: Unknown dependency {fname}")
                        return False
                    deps.append(fname)
                    max_dep_level = max(max_dep_level, self.primitives[fname].level)
            
            new_level = max_dep_level + 1
            
            # 3. Regression Suite Validation
            validator = PrimitiveValidator(interpreter)
            if validation_ios:
                # We need to compile the AST to run it inside validation
                # But actually SemanticHasher used ast.parse above, so 'tree' is valid AST
                if not validator.validate(name, tree.body[0].value, validation_ios):
                    return False

            # Create Node
            node = PrimitiveNode(
                name=name,
                code=code,
                ast_node=tree,
                level=new_level,
                usage_count=1,
                weight=5.0, # High initial weight for novelty
                dependencies=deps,
                semantic_hash=s_hash
            )
            
            self.primitives[name] = node
            self._compile_primitive_runtime(node, interpreter)
            self.save_registry()
            print(f"[Library] Registered Level {new_level} primitive: {name}")
            return True
            
        except Exception as e:
            print(f"[Library] Error registering {name}: {e}")
            return False

    def _compile_primitive_runtime(self, node: PrimitiveNode, interpreter: SafeInterpreter):
        """Creates a callable lambda for the AST and adds to runtime."""
        if node.code == "<native>": return
        
        def executed_func(*args):
             # Map args to variables 'n', 'a', etc. based on signature?
             # For simplicity, we assume generic 'arg0', 'arg1' or positional 'args' if possible.
             # BUT, our synthesizer output usually uses variables like 'n'. 
             # We need to know the argument names.
             # Simple Assumption: Functions are lambda-like or single expression on 'n' or 'x'.
             # Let's map args positionally to detected variables in Order of appearance? No, unsafe.
             # BETTER: Enforce primitives are defined as "lambda n: ..." strings? 
             # OR: Just support `n` as the first arg.
             env = {}
             if len(args) > 0: env['n'] = args[0]
             if len(args) > 1: env['m'] = args[1]
             
             return interpreter.run(node.ast_node, env)
        
        self.runtime_primitives[node.name] = executed_func

    def save_registry(self):
        data = {}
        for name, node in self.primitives.items():
            if node.code == "<native>": continue
            data[name] = {
                "code": node.code,
                "level": node.level,
                "usage": node.usage_count,
                "weight": node.weight,
                "hash": node.semantic_hash
            }
        
        with open(self.registry_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load_registry(self):
        if not os.path.exists(self.registry_path):
            return
            
        with open(self.registry_path, 'r') as f:
            data = json.load(f)
            
        for name, info in data.items():
            try:
                tree = ast.parse(info['code'])
                node = PrimitiveNode(
                    name=name,
                    code=info['code'],
                    ast_node=tree,
                    level=info['level'],
                    usage_count=info['usage'],
                    weight=info['weight'],
                    semantic_hash=info['hash']
                )
                self.primitives[name] = node
            except:
                print(f"[Library] Failed to load {name} - syntax error")

    def get_weighted_ops(self) -> List[str]:
        """Returns a list of operator names tailored for random.choices (weighted)."""
        ops = list(self.primitives.keys())
        weights = [self.primitives[op].weight for op in ops]
        return ops, weights
        
    def _is_bloated(self, code: str) -> bool:
        """
        Detects common tautologies/bloat that inflate complexity without value.
        True if code can be trivially simplified.
        """
        original = code
        simpler = code
        
        # Iterative reduction
        prev = ""
        while simpler != prev:
            prev = simpler
            simpler = simpler.replace("reverse(reverse(", "")
            simpler = simpler.replace("not_op(not_op(", "")
            simpler = simpler.replace("neg(neg(", "")
            simpler = simpler.replace("add(n, 0)", "n")
            simpler = simpler.replace("mul(n, 1)", "n")
            simpler = simpler.replace("sub(n, 0)", "n")
            
            if simpler.endswith("))"):
                 simpler = simpler.replace("))", ")") 
        
        if len(simpler) < len(original) * 0.8:
             return True
        return False

    def feedback(self, used_ops: List[str], success: bool):
        """Update weights based on success/failure."""
        for op in used_ops:
            if op in self.primitives:
                if success:
                    self.primitives[op].weight = min(100.0, self.primitives[op].weight * 1.1)
                    self.primitives[op].usage_count += 1
                else:
                    self.primitives[op].weight = max(0.1, self.primitives[op].weight * 0.95)


# ==============================================================================
# IV. NEURO GENETIC SYNTHESIZER (Main Engine)
# ==============================================================================
class NeuroGeneticSynthesizer:
    _UNARY_OPS = {
        'reverse', 'first', 'last', 'len', 'not_op', 'abs_val', 'neg', 'is_empty',
        'sum_list', 'prod_list', 'min_list', 'max_list', 'count_list', 'range_list'
    }
    _TERNARY_OPS = {'if_then_else'}
    _QUATERNARY_OPS = {'if_gt', 'if_lt'}

    def __init__(self, neural_guide=None, pop_size=200, generations=20, islands=3, checkpoint_path=None, **kwargs):
        """Backwards-compatible init that ignores legacy params but keeps new safe architecture."""
        self.library = LibraryManager()
        self.interpreter = SafeInterpreter(self.library.runtime_primitives)
        
        # Re-register loaded runtime primitives to interpreter
        for name, node in self.library.primitives.items():
            self.library._compile_primitive_runtime(node, self.interpreter)

    def register_primitive(self, name: str, func: Callable):
        """Dynamic runtime registration of new primitives."""
        if self.interpreter and hasattr(self.interpreter, 'primitives'):
             self.interpreter.primitives[name] = func

    def feedback(self, used_ops: List[str], success: bool):
        """Delegate feedback to the library manager."""
        self.library.feedback(used_ops, success)

    def synthesize(self, io_pairs: List[Dict], timeout: Optional[float] = 2.0, **kwargs) -> List[Tuple[str, Any, int, int]]:
        """
        Attempts to find a program that satisfies the IO pairs.
        Returns: list of (code_str, ast_node, complexity, score)
        """
        if timeout is None:
            timeout = 2.0
        start_time = time.time()
        best_programs = []
        
        ops, weights = self.library.get_weighted_ops()
        
        # [TYPE-CONSTRAINT] Prune search space based on input/output types (Refined Instruction 3)
        if io_pairs and 'input' in io_pairs[0] and 'output' in io_pairs[0]:
            inp = io_pairs[0]['input']
            outp = io_pairs[0]['output']
            banned_ops = set()
            
            # Case 1: Int -> ? (Input is scalar)
            if isinstance(inp, int):
                # Integer input cannot support List operations -> Ban them
                banned_ops.update({
                    'reverse', 'split_half', 'first', 'last', 'tail', 'init', 
                    'len', 'sum_list', 'prod_list', 'min_list', 'max_list', 
                    'count_list', 'is_empty', 'snoc', 'cons', 'append',
                    'map_fn', 'filter_fn', 'fold', 'sort' 
                })

            # Case 2: List -> Int (Aggregation task)
            elif isinstance(inp, list) and isinstance(outp, int):
                # We need List->Int ops (len, sum, max), but MUST BAN List->List ops
                # because they return the wrong type (List instead of Int).
                banned_ops.update({
                    'reverse', 'split_half', 'init', 'tail', 'cons', 'snoc', 'append',
                    'map_fn', 'filter_fn', 'sort', 'range_list'
                })
                # Note: 'fold' returns T, so it might be valid if it returns Int. Keep safe for now?
                # Actually fold is the ultimate reducer. Let's KEEP fold, first, last (elements could be int).
                # Banning explicitly list-returning ops.
            
            if banned_ops:
                filtered_ops = []
                filtered_weights = []
                for op, w in zip(ops, weights):
                    if op not in banned_ops:
                        filtered_ops.append(op)
                        filtered_weights.append(w)
                
                ops = filtered_ops
                weights = filtered_weights
                # Ensure we didn't filter everything
                if not ops:
                    print("[Synthesizer] WARN: All ops banned by type constraint! Reverting.")
                    ops, weights = self.library.get_weighted_ops()
                    
        population = self._generate_initial_population(20, ops, weights)
        
        generations = 0
        while time.time() - start_time < timeout:
            generations += 1
            # Evaluation
            scored_pop = []
            for code in population:
                score = self._evaluate(code, io_pairs)
                if score == 1.0:
                    # Found Solution
                    print(f"[Synthesizer] 🏆 Solution found in Gen {generations}: {code}")
                    self._record_success(code, io_pairs)
                    best_programs.append((code, None, len(code), 1.0))
                    # Early exit on perfect solution? Or keep searching?
                    # Let's return immediate for speed
                    return best_programs
                scored_pop.append((code, score))
            
            # Selection & Breeding
            scored_pop.sort(key=lambda x: x[1], reverse=True)
            top_50 = scored_pop[:10]
            
            next_pop = [p[0] for p in top_50]
            while len(next_pop) < 20:
                parent1 = random.choice(top_50)[0]
                parent2 = random.choice(top_50)[0]
                child = self._crossover(parent1, parent2)
                child = self._mutate(child, ops, weights)
                next_pop.append(child)
            
            population = next_pop
            
        return best_programs

    def _generate_initial_population(self, size, ops, weights):
        pop = []
        for _ in range(size):
            # Generate random small expression trees
            depth = random.randint(1, 3)
            pop.append(self._random_expr(depth, ops, weights))
        return pop

    def _random_expr(self, depth, ops, weights):
        if depth <= 0 or random.random() < 0.3:
            return "n" # Base terminal
        
        op = random.choices(ops, weights=weights, k=1)[0]
        # Basic Arity Check (Heuristic)
        # In a real system we'd inspect the signature.
        # Here we hardcode arity for Level 0, assume 1 for learned?
        # Safe fallback: try to guess arity or standard 2.
        
        # For simplicity in this demo, we assume most are binary or unary.
        # We construct a call string.
        arity = 2
        if op in self._UNARY_OPS:
            arity = 1
        elif op in self._TERNARY_OPS:
            arity = 3
        elif op in self._QUATERNARY_OPS:
            arity = 4
        
        args = [self._random_expr(depth-1, ops, weights) for _ in range(arity)]
        return f"{op}({', '.join(args)})"

    def _evaluate(self, code: str, io_pairs: List[Dict]) -> float:
        total_score = 0
        for io in io_pairs:
            inp = io['input']
            expected = io['output']
            
            # Setup Env
            env = {'n': inp}
            
            res = self.interpreter.run(code, env)

            if self._compare_outputs(res, expected):
                total_score += 1
            # Partial credit for being close? (Optional)
            
        return total_score / len(io_pairs) if io_pairs else 0

    def _compare_outputs(self, result: Any, expected: Any) -> bool:
        if isinstance(result, bool) or isinstance(expected, bool):
            return result == expected
        if isinstance(result, (int, float)) and isinstance(expected, (int, float)):
            return abs(result - expected) <= 1e-6
        if np is not None:
            try:
                res_arr = np.asarray(result)
                exp_arr = np.asarray(expected)
                if res_arr.dtype.kind in "if" and exp_arr.dtype.kind in "if":
                    return np.allclose(res_arr, exp_arr, atol=1e-6, rtol=1e-6)
            except Exception:
                pass
        return result == expected

    def _crossover(self, p1: str, p2: str) -> str:
        """
        Structure-Aware Crossover using AST.
        Swaps a random subtree from p1 with a random subtree from p2.
        """
        try:
            # Parse both to AST
            tree1 = ast.parse(p1, mode='eval')
            tree2 = ast.parse(p2, mode='eval')
            
            # Helper to collect swappable nodes (Calls, BinOps, Names, Constants)
            def collect_nodes(tree):
                nodes = []
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Call, ast.BinOp, ast.Name, ast.Constant)):
                        nodes.append(node)
                return nodes
                
            nodes1 = collect_nodes(tree1)
            nodes2 = collect_nodes(tree2)
            
            if not nodes1 or not nodes2:
                return p1
                
            # Pick valid targets
            target1 = random.choice(nodes1)
            replacement = random.choice(nodes2)
            
            # Swap logic (Transformer)
            class SubtreeSwapper(ast.NodeTransformer):
                def __init__(self, target, replacement):
                    self.target = target
                    self.replacement = replacement
                    self.swapped = False
                    
                def visit(self, node):
                    if not self.swapped and node is self.target:
                        self.swapped = True
                        return self.replacement
                    return self.generic_visit(node)
            
            # Apply swap to a copy of tree1? 
            # AST is mutable, but we need to preserve p1 forfallback?
            # Easiest is to just modify tree1 in place. 
            # NodeTransformer modifies the tree.
            
            swapper = SubtreeSwapper(target1, replacement)
            new_tree = swapper.visit(tree1)
            
            if hasattr(ast, 'unparse'):
                return ast.unparse(new_tree)
            else:
                # Fallback for older python (unlikely given requirements, but safe)
                # If no unparse, we can't easily go back to string.
                # Just return p1 as fail-safe.
                return p1
                
        except Exception:
            # Fallback on any parse/transform error
            return p1

    def _mutate(self, code: str, ops, weights) -> str:
        if random.random() < 0.5:
            # Change Operator
            new_op = random.choices(ops, weights=weights, k=1)[0]
            if '(' in code:
                args = code.split('(', 1)[1]
                return f"{new_op}({args}"
            else:
                return f"{new_op}({code})"
        else:
            # Wrap in new operator
            new_op = random.choices(ops, weights=weights, k=1)[0]
            return f"{new_op}({code})"

    def _record_success(self, code: str, io_pairs: List[Dict]):
        # Extract primitives used
        used = []
        for name in self.library.primitives:
            if name in code:
                used.append(name)
        self.library.feedback(used, success=True)
        
        # Try to compress/learn? (RSI Step)
        # If code is long, maybe register it as a new primitive?
        if len(code) > 20 and code.count('(') > 1:
             new_name = f"concept_{len(self.library.primitives)}"
             # [RSI] Validation: Pass I/O pairs to verify consistency before registering
             self.library.register_new_primitive(new_name, code, self.interpreter, validation_ios=io_pairs)

# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    print("[RSI Core] Safe Architecture Loaded.")
    
    # Self-Test
    synth = NeuroGeneticSynthesizer()
    
    # Test 1: Base Primitive
    print("Test 1: Run 'add(n, 5)'")
    res = synth.interpreter.run("add(n, 5)", {'n': 10})
    print(f"Result: {res} (Expected 15)")
    
    # Test 2: Synthesis
    print("\nTest 2: Synthesize 'n * 2'")
    io = [{'input': 1, 'output': 2}, {'input': 5, 'output': 10}, {'input': 3, 'output': 6}]
    synth.synthesize(io)
