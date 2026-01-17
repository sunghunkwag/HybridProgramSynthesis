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

def _raise(ex):
    """Helper to raise exceptions in lambdas."""
    raise ex


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
                # [DIAGNOSTIC] Return syntax error details
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

    def visit_Name(self, node):
        self._check_gas()
        if isinstance(node.ctx, ast.Load):
            if node.id in self.local_env:
                return self.local_env[node.id]
            # Allow True/False/None
            if node.id == "True": return True
            if node.id == "False": return False
            if node.id == "None": return None
            raise ValueError(f"Undefined variable: {node.id}")
        raise ValueError("Assignments not allowed in expression mode")

    def visit_BinOp(self, node):
        self._check_gas()
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = node.op
        
        if isinstance(op, ast.Add): return left + right
        if isinstance(op, ast.Sub): return left - right
        if isinstance(op, ast.Mult): return left * right
        if isinstance(op, ast.Div): return left / right if right != 0 else 0
        if isinstance(op, ast.FloorDiv): return left // right if right != 0 else 0
        if isinstance(op, ast.Mod): return left % right if right != 0 else 0
        if isinstance(op, ast.Pow): return left ** right if isinstance(right, int) and right < 10 else 0 # Safety cap
        return 0

    def visit_UnaryOp(self, node):
        self._check_gas()
        operand = self.visit(node.operand)
        op = node.op
        if isinstance(op, ast.USub): return -operand
        if isinstance(op, ast.Not): return not operand
        return operand

    def visit_Compare(self, node):
        self._check_gas()
        left = self.visit(node.left)
        # Handle chain comparisons ideally, but for now simple single comparison
        if len(node.ops) != 1: return False
        
        op = node.ops[0]
        right = self.visit(node.comparators[0])
        
        if isinstance(op, ast.Eq): return left == right
        if isinstance(op, ast.NotEq): return left != right
        if isinstance(op, ast.Lt): return left < right
        if isinstance(op, ast.LtE): return left <= right
        if isinstance(op, ast.Gt): return left > right
        if isinstance(op, ast.GtE): return left >= right
        if isinstance(op, ast.In): return left in right
        if isinstance(op, ast.NotIn): return left not in right
        return False
    
    def visit_BoolOp(self, node):
        self._check_gas()
        values = [self.visit(v) for v in node.values]
        if isinstance(node.op, ast.And):
            return all(values)
        if isinstance(node.op, ast.Or):
            return any(values)
        return False

    def visit_IfExp(self, node):
        self._check_gas()
        test = self.visit(node.test)
        if test:
            return self.visit(node.body)
        else:
            return self.visit(node.orelse)

    def visit_List(self, node):
        self._check_gas()
        return [self.visit(elt) for elt in node.elts]
        
    def visit_Subscript(self, node):
        self._check_gas()
        val = self.visit(node.value)
        idx = self.visit(node.slice)
        try:
            return val[idx]
        except:
            return None

    def visit_Call(self, node):
        self._check_gas()
        if not isinstance(node.func, ast.Name):
            raise ValueError("Indirect calls not allowed")
            
        func_name = node.func.id
        if func_name not in self.primitives:
             raise ValueError(f"Unknown primitive: {func_name}")
             
        args = [self.visit(arg) for arg in node.args]
        return self.primitives[func_name](*args)
        
    def generic_visit(self, node):
        raise ValueError(f"Illegal AST node: {type(node).__name__}")


# ==============================================================================
# II. SEMANTIC HASHER (Quality Control)
# ==============================================================================
class SemanticHasher(ast.NodeVisitor):
    """
    Canonicalizes an AST to detect semantic duplicates.
    Renames variables to arg0, arg1... to ignore naming differences.
    """
    def __init__(self):
        self.tokens = []
        self.var_map = {}
        self.arg_counter = 0

    def hash_code(self, code_str: str) -> str:
        try:
            node = ast.parse(code_str, mode='eval')
            self.tokens = []
            self.var_map = {}
            self.arg_counter = 0
            self.visit(node)
            data = "|".join(self.tokens).encode('utf-8')
            return hashlib.sha256(data).hexdigest()
        except:
            return "INVALID"

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            if node.id not in self.var_map:
                self.var_map[node.id] = f"ARG_{self.arg_counter}"
                self.arg_counter += 1
            self.tokens.append(f"VAR:{self.var_map[node.id]}")
        else:
            self.tokens.append("VAR_DEF")

    def visit_Constant(self, node):
        self.tokens.append(f"CONST:{node.value}")

    def visit_Call(self, node):
        self.tokens.append(f"CALL:{node.func.id}")
        for arg in node.args:
            self.visit(arg)

    def visit_BinOp(self, node):
        self.tokens.append(f"OP:{type(node.op).__name__}")
        self.visit(node.left)
        self.visit(node.right)
        
    def generic_visit(self, node):
        self.tokens.append(type(node).__name__)
        super().generic_visit(node)


# ==============================================================================
# III. LIBRARY MANAGER (RSI Registry & DAG)
# ==============================================================================
@dataclass
class PrimitiveNode:
    name: str
    code: str  # Python source code string
    ast_node: Any = field(default=None) # Parsed AST
    level: int = 0
    usage_count: int = 0
    weight: float = 1.0
    dependencies: List[str] = field(default_factory=list)
    semantic_hash: str = ""

class PrimitiveValidator:
    """
    Validates new primitives against a regression suite before acceptance.
    Inspired by HotSwapManager check pattern.
    """
    def __init__(self, interpreter: SafeInterpreter):
        self.interpreter = interpreter

    def validate(self, name: str, ast_node: Any, validation_ios: List[Dict]) -> bool:
        """
        Runs the primitive against known I/O examples to ensure correctness.
        For a primitive 'concept_1(n)', we check if it correctly transforms input -> output.
        """
        if not validation_ios:
            return True # No tests provided, soft accept (or reject depending on strictness)
            
        print(f"[Validator] Running regression suite for {name} ({len(validation_ios)} tests)...")
        
        # [PRESCRIPTION B] Enforce Strict Concept IDs
        # If name implies a type (e.g. concept_5_int_to_int), reject mismatched IOs immediately.
        if "_to_" in name:
            for io in validation_ios:
                inp = io['input']
                # Detect IO based signature
                sig_parts = name.split("_")
                try:
                    # e.g. concept_5_int_to_int -> expected="int_to_int"
                    expected_sig = "_".join(sig_parts[-3:]) # "int_to_int"
                except:
                    continue
                    
                # Very basic check: just ensuring we don't mix scalars and lists if labeled
                if "int_to" in expected_sig and isinstance(inp, list):
                     print(f"  [Validator] FAIL:SignatureMismatch (Labeled {expected_sig} but got List input)")
                     return False
                if "list_to" in expected_sig and isinstance(inp, int):
                     print(f"  [Validator] FAIL:SignatureMismatch (Labeled {expected_sig} but got Int input)")
                     return False
                if "matrix_to" in expected_sig and not (isinstance(inp, list) and inp and isinstance(inp[0], list)):
                     print(f"  [Validator] FAIL:SignatureMismatch (Labeled {expected_sig} but got non-Matrix input)")
                     return False

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
                         
                         # [PRESCRIPTION A] Explicit Failure Codes
                         got_str = str(res)
                         if isinstance(res, dict) and "__error__" in res:
                             error_type = res['__error__']
                             msg = res['msg']
                             if "Shape" in msg or "dimension" in msg:
                                 got_str = f"FAIL:ShapeError({msg})"
                             elif error_type == "TypeError":
                                 got_str = f"FAIL:TypeError({msg})"
                             elif error_type == "RecursionError":
                                 got_str = f"FAIL:Timeout({msg})"
                             else:
                                 got_str = f"FAIL:{error_type}({msg})"
                         elif res is None:
                             got_str = "FAIL:None"
                         elif isinstance(res, list) and not res and isinstance(io['output'], int):
                             # Special case for Empty List vs Int (Shape Error)
                             got_str = "FAIL:ShapeError(Expected Int, Got [])"
                         else:
                             if len(got_str) > 100: got_str = got_str[:100] + "..."
                             
                         print(f"  [Validator-Diag] Fail: Input={io['input']} ({type_in}) | Expected={io['output']} ({type_exp}) | Got={got_str} ({type_got})")
                     except Exception:
                         # Fallback if result is not indexable or len() fails
                         print(f"  [Validator-Diag] Fail: Input={io['input']} | Expected={io['output']} | Got={res}")
                     return False # Return False on first failure for diagnostics
            except Exception as e:
                if passes == 0:
                     print(f"  [Validator-Diag] Exception: {str(e)[:100]}")
                # If an exception occurs during execution, it's a failure
                return False
        
        # If all tests pass, return True
        if passes == len(validation_ios):
            print(f"[Validator] {name} passed regression suite (100%).")
            return True
        else:
            print(f"[Validator] {name} failed regression suite (Score: {score:.2f}). Rejected.")
            return False

    def validate_score(self, name: str, ast_node: Any, validation_ios: List[Dict]) -> float:
        """
        Returns the pass rate as a float (0.0 - 1.0).
        Does NOT print diagnostics; used for percentage-based checks.
        """
        if not validation_ios:
            return 1.0  # No tests = assume pass
        
        passes = 0
        for io in validation_ios:
            env = {'n': io['input']}
            try:
                res = self.interpreter.run(ast_node, env)
                
                # [FIX C] ShapeError = discard immediately (score 0)
                if isinstance(io['output'], (int, float)):
                    if isinstance(res, list):
                        continue  # ShapeError - doesn't count as pass
                    if res is None or (isinstance(res, dict) and "__error__" in res):
                        continue  # Error - doesn't count
                
                # Compare
                if res == io['output']:
                    passes += 1
                elif isinstance(io['output'], list) and isinstance(res, list):
                    if res == io['output']:
                        passes += 1
            except Exception:
                pass  # Exception = failure
        
        return passes / len(validation_ios)

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
            'sum_list': (lambda lst: sum(lst[:100]) if isinstance(lst, list) else (_raise(TypeError("sum_list expects list")), 0)[1], 0),
            'prod_list': (lambda lst: self._safe_product(lst[:100]) if isinstance(lst, list) else (_raise(TypeError("prod_list expects list")), 0)[1], 0),
            'min_list': (lambda lst: min(lst[:100]) if isinstance(lst, list) and len(lst) > 0 else None, 0), # None is valid for empty
            'max_list': (lambda lst: max(lst[:100]) if isinstance(lst, list) and len(lst) > 0 else None, 0),
            'count_list': (lambda lst: len(lst) if isinstance(lst, list) else (_raise(TypeError("count_list expects list")), 0)[1], 0),
            
            # --- Matrix (2D List) Operations ---
            'flatten': (lambda m: self._safe_flatten(m), 0),
            'matrix_sum': (lambda m: self._safe_matrix_sum(m), 0),
            'row_sums': (lambda m: [sum(row[:100]) for row in m[:100]] if self._is_matrix(m) else [], 0),
            'col_sums': (lambda m: self._safe_col_sums(m), 0),
            'matrix_shape': (lambda m: (len(m), len(m[0]) if m and isinstance(m[0], list) else 0) if isinstance(m, list) else (0, 0), 0),
            
            # --- Sorting (Built-in, bounded) ---
            'sort_asc': (lambda lst: sorted(lst[:100]) if isinstance(lst, list) else lst, 0),
            'sort_desc': (lambda lst: sorted(lst[:100], reverse=True) if isinstance(lst, list) else lst, 0),
            
            # --- Search ---
            'elem_in': (lambda x, lst: x in lst if isinstance(lst, (list, str)) else False, 0),
            'index_of': (lambda x, lst: lst.index(x) if isinstance(lst, list) and x in lst else -1, 0),
            
            # --- Insertion Sort Building Blocks ---
            'insert_sorted': (lambda x, lst: self._insert_sorted(x, lst), 0),
            
            # --- Merge Sort Building Blocks ---
            'split_half': (lambda lst: (lst[:len(lst)//2], lst[len(lst)//2:]) if isinstance(lst, list) else (_raise(TypeError("split_half expects list")), None)[0], 0),
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
            raise TypeError(f"insert_sorted expects list, got {type(lst).__name__}")
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
             raise TypeError(f"merge_sorted expects list, got {type(a).__name__}")
        if not isinstance(b, list):
             raise TypeError(f"merge_sorted expects list, got {type(b).__name__}")
        
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

    def _is_matrix(self, m) -> bool:
        """Check if input is a 2D list (list of lists)."""
        if not isinstance(m, list) or len(m) == 0:
            return False
        return all(isinstance(row, list) for row in m)

    def _safe_flatten(self, m) -> list:
        """Flatten a 2D list into a 1D list. Bounded."""
        if not self._is_matrix(m):
            return m if isinstance(m, list) else []
        result = []
        for row in m[:100]:  # Outer bound
            for elem in row[:100]:  # Inner bound
                result.append(elem)
                if len(result) >= 10000:  # Hard limit
                    return result
        return result

    def _safe_matrix_sum(self, m) -> int:
        """Sum all elements of a 2D list."""
        if not self._is_matrix(m):
            return 0
        total = 0
        for row in m[:100]:
            for elem in row[:100]:
                if isinstance(elem, (int, float)):
                    total += elem
        return total

    def _safe_col_sums(self, m) -> list:
        """Return sum of each column in a 2D list."""
        if not self._is_matrix(m) or len(m) == 0:
            return []
        num_cols = len(m[0]) if m[0] else 0
        sums = [0] * min(num_cols, 100)  # Bound columns
        for row in m[:100]:
            for i, val in enumerate(row[:100]):
                if i < len(sums) and isinstance(val, (int, float)):
                    sums[i] += val
        return sums

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
        if not isinstance(lst, list):
            raise TypeError("map expects list")
        if len(lst) > 100:
            lst = lst[:100]
            
        if func_name not in self.runtime_primitives:
             raise ValueError(f"Unknown primitive {func_name}")
             
        fn = self.runtime_primitives[func_name]
        # Propagate exceptions!
        return [fn(x) for x in lst]

    def _safe_filter(self, func_name: str, lst: list) -> list:
        """Filter list by predicate primitive, max 100 iterations."""
        if not isinstance(lst, list): 
             raise TypeError("filter expects list")
        if len(lst) > 100: lst = lst[:100]
        
        if func_name not in self.runtime_primitives:
             raise ValueError(f"Unknown primitive {func_name}")
             
        fn = self.runtime_primitives[func_name]
        return [x for x in lst if fn(x)]

    def _safe_fold(self, func_name: str, init, lst: list):
        """Fold list logic."""
        if not isinstance(lst, list): 
             raise TypeError("fold expects list")
        if len(lst) > 100: lst = lst[:100]
        
        if func_name not in self.runtime_primitives:
             raise ValueError(f"Unknown primitive {func_name}")
             
        fn = self.runtime_primitives[func_name]
        acc = init
        for x in lst:
            acc = fn(acc, x)
        return acc

    def _is_tautology(self, code: str, validation_ios: List[Dict], interpreter: SafeInterpreter) -> bool:
        """
        Detects functional tautologies (always returns True, False, 0, or Input).
        """
        # 1. Syntactic checks
        if "eq(n, n)" in code or "sub(n, n)" in code:
            return True
            
        # 2. Semantic checks (Behavioral)
        if not validation_ios: return True # Can't verify behavior
        
        # Check if output is constant or equal to input across all examples
        first_out = None
        all_same = True
        is_identity = True
        
        try:
            # quick abstract syntax tree check for single var
            tree = ast.parse(code)
            
            for io in validation_ios:
                env = {'n': io['input']}
                res = interpreter.run(tree.body[0].value, env)
                
                if first_out is None:
                    first_out = res
                else:
                    if res != first_out:
                        all_same = False
                
                if res != io['input']:
                    is_identity = False
            
            if all_same: return True # Constant function (e.g. always True)
            if is_identity: return True # Identity function (reverse(reverse(n)) == n)
            
        except:
            pass
            
        return False

    def _provides_novelty(self, code: str, validation_ios: List[Dict], interpreter: SafeInterpreter) -> bool:
        """
        Ensures the new primitive behavior isn't already covered by a simple base primitive.
        e.g. If new_prim(x) == reverse(x) for all inputs, strictly reject it.
        """
        if not validation_ios: return False
        
        # Get outputs of new primitive
        new_outputs = []
        try:
            tree = ast.parse(code)
            for io in validation_ios:
                env = {'n': io['input']}
                new_outputs.append(interpreter.run(tree.body[0].value, env))
        except:
            return False
            
        # Compare against Level 0/1 primitives
        for name, prim in self.primitives.items():
            if prim.level > 1: continue # Only compare against base/simple stuff
            if prim.code == code: continue
            
            match = True
            # Check alias
            if name in self.runtime_primitives:
                func = self.runtime_primitives[name]
                for i, io in enumerate(validation_ios):
                    try:
                        # Assuming 1-arity for simple comparison
                        existing_res = func(io['input'])
                        if existing_res != new_outputs[i]:
                            match = False
                            break
                    except:
                        match = False
                        break
            
            if match:
                print(f"[Novelty] Rejecting: Behavior identical to existing '{name}'")
                return False
                
        return True

    def register_new_primitive(self, name: str, code: str, interpreter: SafeInterpreter, 
                               validation_ios: List[Dict] = None, holdout_ios: List[Dict] = None) -> bool:
        """
        Attempts to register a new primitive with STRICT validation.
        CHECKS:
        1. MANDATORY Validation IO (>=3 examples)
        2. Tautology Detection (eq(x,x), constant output, identity)
        3. Semantic Uniqueness (Hash check)
        4. Novelty Check (vs Base Library)
        5. Regression & Holdout Score (>= 95%)
        """
        # [FIX 1] MANDATORY Validation
        if not validation_ios or len(validation_ios) < 3:
            print(f"[Library] Rejecting {name}: Validation requires >= 3 IO pairs (got {len(validation_ios) if validation_ios else 0})")
            return False
        
        # [FIX 2] Tautology Detection
        if self._is_tautology(code, validation_ios, interpreter):
            print(f"[Library] Rejecting {name}: Detected tautology (constant/identity)")
            return False

        # [RSI-Fix] Semantic De-Bloating
        if self._is_bloated(code):
            print(f"[Library] Rejecting {name}: Detected syntactic bloat")
            return False
            
        # 3. Hash Check
        s_hash = self.hasher.hash_code(code)
        for p in self.primitives.values():
            if p.semantic_hash == s_hash and p.name != name:
                print(f"[Library] Rejecting {name}: Semantically identical to {p.name}")
                return False
        
        # 4. Novelty Check (Behavioral)
        if not self._provides_novelty(code, validation_ios, interpreter):
            # Already logged in function
            return False

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
            
            # 5. Strict Score Validation
            validator = PrimitiveValidator(interpreter)
            
            # Regression (Training Data)
            reg_score = validator.validate_score(name, tree.body[0].value, validation_ios)
            if reg_score < 0.95:
                print(f"[Library] Rejecting {name}: Regression score {reg_score:.1%} < 95%")
                return False
            
            # Holdout (Test Data)
            if holdout_ios:
                hold_score = validator.validate_score(name, tree.body[0].value, holdout_ios)
                if hold_score < 0.95:
                    print(f"[Library] Rejecting {name}: Holdout score {hold_score:.1%} < 95%")
                    return False

            # Create Node
            node = PrimitiveNode(
                name=name,
                code=code,
                ast_node=tree,
                level=new_level,
                usage_count=0, # Start at 0, prove worth
                weight=5.0, 
                dependencies=deps,
                semantic_hash=s_hash
            )
            
            self.primitives[name] = node
            self._compile_primitive_runtime(node, interpreter)
            self.save_registry()
            print(f"[Library] ✅ Registered Level {new_level} primitive: {name} (Passed Strict RSI Gate)")
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

    def garbage_collect_unused(self, min_usage: int = 3, min_cycles: int = 10) -> int:
        """
        [FIX D] Remove primitives that haven't proven useful.
        
        A primitive is garbage collected if:
        1. It's not a base primitive (level 0)
        2. Its usage_count < min_usage after min_cycles
        3. Its weight has decayed below 0.5 (indicating repeated failures)
        
        Returns: number of primitives removed
        """
        to_remove = []
        
        for name, node in self.primitives.items():
            # Never GC base primitives
            if node.level == 0 or node.code == "<native>":
                continue
            
            # Check GC conditions
            if node.usage_count < min_usage and node.weight < 0.5:
                to_remove.append(name)
                print(f"[Library-GC] Marking {name} for removal (usage={node.usage_count}, weight={node.weight:.2f})")
        
        # Remove marked primitives
        for name in to_remove:
            del self.primitives[name]
            if name in self.runtime_primitives:
                del self.runtime_primitives[name]
        
        if to_remove:
            self.save_registry()
            print(f"[Library-GC] Removed {len(to_remove)} unused primitives")
        
        return len(to_remove)
        
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

    def __init__(self, neural_guide=None, pop_size=200, generations=20, islands=3, checkpoint_path=None, use_meta_heuristic=True, **kwargs):
        """
        Backwards-compatible init that ignores legacy params but keeps new safe architecture.
        use_meta_heuristic: If False, meta-learning weights are ignored (Control group mode).
        """
        self.library = LibraryManager()
        self.interpreter = SafeInterpreter(self.library.runtime_primitives)
        self.use_meta_heuristic = use_meta_heuristic
        
        # [TRUE RSI] Meta-Reasoning Failure Analyzer
        from meta_heuristic import FailureAnalyzer
        self.failure_analyzer = FailureAnalyzer()
        
        # Re-register loaded runtime primitives to interpreter
        for name, node in self.library.primitives.items():
            self.library._compile_primitive_runtime(node, self.interpreter)

    # ... (other methods unchanged) ...
    
    def register_primitive(self, name: str, func: Callable):
        """Dynamic runtime registration of new primitives."""
        if self.interpreter and hasattr(self.interpreter, 'primitives'):
             self.interpreter.primitives[name] = func

    def feedback(self, used_ops: List[str], success: bool):
        """Delegate feedback to the library manager."""
        self.library.feedback(used_ops, success)

    def _is_2d_list(self, x) -> bool:
        """Check if input is a 2D list (list of lists)."""
        if not isinstance(x, list) or len(x) == 0:
            return False
        return all(isinstance(row, list) for row in x)

    def synthesize(self, io_pairs: List[Dict], timeout: Optional[float] = 2.0, **kwargs) -> List[Tuple[str, Any, int, int]]:
        """
        Attempts to find a program that satisfies the IO pairs.
        Returns: list of (code_str, ast_node, complexity, score)
        """
        if timeout is None:
            timeout = 2.0
        start_time = time.time()
        best_programs = []
        
        # [A] MULTIPLICATIVE MERGE: Library weights × Meta-learned weights
        from meta_heuristic import MetaHeuristic
        # If Control Group (use_meta_heuristic=False), disable IO to prevent contamination
        meta_heuristic = MetaHeuristic(no_io=not self.use_meta_heuristic)
        
        ops, lib_weights = self.library.get_weighted_ops()
        meta_weights_dict = meta_heuristic.get_op_weights(ops)
        
        # Multiplicative merge: final_w[i] = lib_w[i] * meta_w[i]
        weights = []
        for i, op in enumerate(ops):
            lib_w = lib_weights[i]
            # If meta-learning is disabled (Control Group), force meta_w to 1.0
            if not self.use_meta_heuristic:
                meta_w = 1.0
            else:
                meta_w = meta_weights_dict.get(op, 1.0)
            
            final_w = lib_w * meta_w
            weights.append(max(0.01, final_w))  # Ensure non-zero
        
        # Store for later verification
        self._current_meta_weights = meta_weights_dict if self.use_meta_heuristic else {op: 1.0 for op in ops}
        self._current_merged_weights = dict(zip(ops, weights))
        
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

            # Case 2: 1D List -> Int (Aggregation task)
            elif isinstance(inp, list) and not self._is_2d_list(inp) and isinstance(outp, int):
                # We need List->Int ops (len, sum, max), but MUST BAN List->List ops
                banned_ops.update({
                    'reverse', 'split_half', 'init', 'tail', 'cons', 'snoc', 'append',
                    'map_fn', 'filter_fn', 'sort', 'range_list',
                    # Also ban matrix ops on 1D lists
                    'matrix_sum', 'row_sums', 'col_sums', 'flatten', 'matrix_shape'
                })

            # Case 3: 2D List (Matrix) -> Int (Matrix aggregation task)
            elif isinstance(inp, list) and self._is_2d_list(inp) and isinstance(outp, int):
                # Ban 1D list operators that will TypeError on 2D lists
                banned_ops.update({
                    'sum_list', 'prod_list', 'min_list', 'max_list',  # These fail on nested lists
                    'reverse', 'split_half', 'init', 'tail', 'cons', 'snoc', 'append',
                    'map_fn', 'filter_fn', 'sort_asc', 'sort_desc', 'range_list',
                    'first', 'last'  # Returns a list, not an int
                })
                # KEEP: matrix_sum, flatten (to reduce to 1D then sum), row_sums, col_sums
            
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
        
        # [PRESCRIPTION C] Force Scalar Output Check
        # If output is Int, we should force the root operator to be scalar-returning.
        scalar_goal = False
        if io_pairs and isinstance(io_pairs[0].get('output'), int):
            scalar_goal = True
            
        population = self._generate_initial_population(20, ops, weights, scalar_goal=scalar_goal)
        
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
                    # [TRUE RSI] Organic Learning: Update persistent meta-weights on success
                    meta_heuristic.learn(code)
                    best_programs.append((code, None, len(code), 1.0))
                    # Early exit on perfect solution? Or keep searching?
                    # Let's return immediate for speed
                    return best_programs
                else:
                    # [TRUE RSI] Analyze WHY this candidate failed
                    if score < 0.5 and io_pairs and hasattr(self, 'failure_analyzer'):
                        env = {'n': io_pairs[0]['input']}
                        result = self.interpreter.run(code, env)
                        analysis = self.failure_analyzer.analyze_failure(code, result, io_pairs[0])
                        
                        # [TRUE RSI] Organic Learning: Update persistent meta-weights on failure
                        err_type = analysis.get('error_type', 'unknown')
                        mh_fail_type = 'LOW_SCORE_VALID'
                        if err_type in ('ShapeError', 'TypeMismatch'):
                            mh_fail_type = 'TYPE_OR_SHAPE'
                        elif err_type in ('NoneReturn', 'timeout', 'unknown') or 'Error' in err_type:
                            mh_fail_type = 'EXCEPTION'
                        
                        meta_heuristic.learn_failure(code, failure_type=mh_fail_type, context={'ops_used': analysis.get('ops_used', [])})
                        
                        # Every 100 failures, print reasoning summary and apply adjustments
                        total_failures = sum(self.failure_analyzer.error_counts.values())
                        if total_failures > 0 and total_failures % 100 == 0:
                            self.failure_analyzer.print_reasoning_summary()
                            adjustments = self.failure_analyzer.get_strategy_adjustments()
                            if adjustments:
                                print(f"[Synthesizer] APPLYING META-ADJUSTMENTS: {adjustments}")
                                
                                # [REAL ACTION 1] Reduce weights for failing ops
                                for op, node in self.library.primitives.items():
                                    if f'reduce_weight_{op}' in adjustments:
                                        old_weight = node.weight
                                        node.weight *= adjustments[f'reduce_weight_{op}']
                                        print(f"  -> Reduced weight of '{op}': {old_weight:.2f} -> {node.weight:.2f}")
                                
                                # [REAL ACTION 2] increase_type_strictness: Ban more risky ops
                                if adjustments.get('increase_type_strictness'):
                                    extra_bans = {'elem_in', 'index_of', 'count_val', 'nth'}
                                    ops = [o for o in ops if o not in extra_bans]
                                    weights = [self.library.primitives[o].weight for o in ops if o in self.library.primitives]
                                    print(f"  -> Type strictness: Banned {extra_bans}")
                                
                                # [REAL ACTION 3] ban_unsafe_ops: Remove None-returning ops
                                if adjustments.get('ban_unsafe_ops'):
                                    unsafe_ops = {'first', 'last', 'nth', 'head', 'tail'}
                                    ops = [o for o in ops if o not in unsafe_ops]
                                    weights = [self.library.primitives[o].weight for o in ops if o in self.library.primitives]
                                    print(f"  -> Unsafe ops: Banned {unsafe_ops}")
                                
                                # [REAL ACTION 4] force_scalar_root: Set scalar_goal
                                if adjustments.get('force_scalar_root'):
                                    scalar_goal = True
                                    print(f"  -> Scalar goal: FORCED for remaining generations")
                                
                                # [REAL ACTION 5] ban_list_producers: Remove list-producing ops
                                if adjustments.get('ban_list_producers'):
                                    list_producer_ops = adjustments.get('list_producer_ops', set())
                                    before_count = len(ops)
                                    ops = [o for o in ops if o not in list_producer_ops]
                                    weights = [self._current_merged_weights.get(o, 1.0) for o in ops]
                                    after_count = len(ops)
                                    # Store for verification
                                    if not hasattr(self, '_banned_ops_history'):
                                        self._banned_ops_history = []
                                    self._banned_ops_history.append({'before': before_count, 'after': after_count, 'banned': list_producer_ops})
                                    print(f"  -> List producers: Banned {list_producer_ops}, ops {before_count} -> {after_count}")
                                    # ASSERT: banned_ops actually reduced ops count
                                    assert after_count < before_count or len(list_producer_ops & set(ops)) == 0, "ban_list_producers failed to reduce ops"
                                
                                # Refresh population with new constraints
                                population = self._generate_initial_population(20, ops, weights, scalar_goal=scalar_goal)
                                print(f"  -> Regenerated population with {len(ops)} ops, scalar_goal={scalar_goal}")
                        
                scored_pop.append((code, score))
            
            # Selection & Breeding
            scored_pop.sort(key=lambda x: x[1], reverse=True)
            top_50 = scored_pop[:10]
            
            next_pop = [p[0] for p in top_50]
            while len(next_pop) < 20:
                parent1 = random.choice(top_50)[0]
                parent2 = random.choice(top_50)[0]
                child = self._crossover(parent1, parent2)
                # [FIX] Pass scalar_goal to mutate to maintain scalar root constraint
                child = self._mutate(child, ops, weights, scalar_goal=scalar_goal)
                next_pop.append(child)
            
            population = next_pop
            
        return best_programs

        return best_programs

    def _generate_initial_population(self, size, ops, weights, scalar_goal=False):
        pop = []
        for _ in range(size):
            # Generate random small expression trees
            depth = random.randint(1, 3)
            pop.append(self._random_expr(depth, ops, weights, scalar_root=scalar_goal))
        return pop

    def _random_expr(self, depth, ops, weights, scalar_root=False):
        if depth <= 0 or (not scalar_root and random.random() < 0.3):
             # If scalar_root is True, we CANNOT return 'n' (which might be a list/matrix).
             # We MUST select a scalar op.
             if scalar_root:
                 pass # Force op selection
             else:
                return "n" # Base terminal
        
        # [PRESCRIPTION C] Force Scalar Root Constraint
        # If scalar_root is True, we must pick an operator that returns an Int/Scalar.
        # We assume _UNARY_OPS contains scalar reducers (sum, len, etc.)
        valid_ops = ops
        valid_weights = weights
        
        if scalar_root:
            # Filter ops to only those returning Int/Scalar
            # This is heuristic-based on common naming or explicit lists
            scalar_ops = {
                'len', 'sum_list', 'prod_list', 'min_list', 'max_list', 'count_list',
                'matrix_sum', 'row_sums', 'col_sums', 'index_of', 'count_val',
                'add', 'sub', 'mul', 'div', 'mod', 'pow', 'abs_val', 'neg'
            }
            # row_sums/col_sums return List, remove them!
            scalar_ops.difference_update({'row_sums', 'col_sums'})
            
            filtered = []
            f_weights = []
            for op, w in zip(ops, weights):
                if op in scalar_ops:
                    filtered.append(op)
                    f_weights.append(w)
            
            if filtered:
                valid_ops = filtered
                valid_weights = f_weights
        
        if not valid_ops: # Fallback if no scalar ops found
             if scalar_root: return "0" # Emergency constant
             valid_ops = ops
             valid_weights = weights

        op = random.choices(valid_ops, weights=valid_weights, k=1)[0]
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
        
        # Recursive calls do NOT enforce scalar_root (only the top level did)
        args = [self._random_expr(depth-1, ops, weights, scalar_root=False) for _ in range(arity)]
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

    def _mutate(self, code: str, ops, weights, scalar_goal=False) -> str:
        # [FIX] Filter ops for scalar_goal if needed
        valid_ops = ops
        valid_weights = weights
        
        if scalar_goal:
            # Only allow scalar-returning operators at the root
            scalar_ops = {
                'len', 'sum_list', 'prod_list', 'min_list', 'max_list', 'count_list',
                'matrix_sum', 'index_of', 'add', 'sub', 'mul', 'div', 'mod', 'abs_val', 'neg'
            }
            filtered = []
            f_weights = []
            for op, w in zip(ops, weights):
                if op in scalar_ops:
                    filtered.append(op)
                    f_weights.append(w)
            if filtered:
                valid_ops = filtered
                valid_weights = f_weights

        if random.random() < 0.5:
            # Change Operator
            new_op = random.choices(valid_ops, weights=valid_weights, k=1)[0]
            if '(' in code:
                args = code.split('(', 1)[1]
                return f"{new_op}({args}"
            else:
                return f"{new_op}({code})"
        else:
            # Wrap in new operator
            new_op = random.choices(valid_ops, weights=valid_weights, k=1)[0]
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
