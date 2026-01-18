
import json
import os
import tempfile
from typing import Dict, Any, List, Tuple
from collections import defaultdict
import re


class FailureAnalyzer:
    """
    TRUE RSI Meta-Reasoning Component.
    
    This is NOT simple weight adjustment. This ACTUALLY REASONS about:
    1. WHY did this program fail?
    2. What PATTERN of failures am I seeing?
    3. What HYPOTHESIS can I form about the problem?
    4. How should I CHANGE my search strategy?
    """
    
    def __init__(self):
        # Track failure patterns
        self.error_counts = defaultdict(int)  # error_type -> count
        self.op_failure_map = defaultdict(lambda: defaultdict(int))  # op -> error_type -> count
        self.type_mismatch_patterns = []  # [(input_type, output_type, expected_type), ...]
        self.hypothesis_log = []  # List of generated hypotheses
        
        # Derived strategy adjustments
        self.banned_op_combinations = set()  # (op1, op2) pairs that always fail
        self.type_constraints = {}  # input_type -> allowed_ops
        
    def analyze_failure(self, program: Any, error_info: Dict, io_pair: Dict) -> Dict[str, Any]:
        """
        Deeply analyze a failure to understand WHY it happened.
        Returns: analysis dict with error_type, cause, hypothesis, action
        """
        analysis = {
            'error_type': 'unknown',
            'cause': None,
            'hypothesis': None,
            'recommended_action': None
        }
        
        # 1. Classify error type
        if isinstance(error_info, dict) and '__error__' in error_info:
            error_type = error_info['__error__']
            msg = error_info.get('msg', '')
        elif error_info is None:
            error_type = 'NoneReturn'
            msg = 'Program returned None'
        elif isinstance(error_info, list) and isinstance(io_pair.get('output'), int):
            error_type = 'ShapeError'
            msg = f'Expected Int, got List of length {len(error_info)}'
        else:
            error_type = 'ValueMismatch'
            msg = f'Expected {io_pair.get("output")}, got {error_info}'
        
        analysis['error_type'] = error_type
        self.error_counts[error_type] += 1
        
        # 2. Extract operators used
        ops_used = self._extract_ops(program)
        for op in ops_used:
            self.op_failure_map[op][error_type] += 1
        
        # 3. Track type mismatch pattern
        inp = io_pair.get('input')
        exp = io_pair.get('output')
        inp_type = self._classify_type(inp)
        exp_type = self._classify_type(exp)
        got_type = self._classify_type(error_info)
        
        self.type_mismatch_patterns.append((inp_type, got_type, exp_type))
        
        # 4. GENERATE HYPOTHESIS (This is the real meta-reasoning)
        hypothesis = self._generate_hypothesis(error_type, ops_used, inp_type, exp_type, got_type)
        analysis['hypothesis'] = hypothesis
        
        # 5. RECOMMEND ACTION
        action = self._recommend_action(hypothesis, ops_used)
        analysis['recommended_action'] = action
        
        # Log for transparency
        if len(self.hypothesis_log) < 100:  # Limit memory
            self.hypothesis_log.append(hypothesis)
        
        return analysis
    
    def _extract_ops(self, program: Any) -> List[str]:
        """Extract operator names from program AST."""
        ops = []
        def traverse(node):
            node_type = type(node).__name__
            if node_type == 'BSBinOp':
                ops.append(f'op_{node.op}')
                traverse(node.left)
                traverse(node.right)
            elif node_type == 'BSRecCall':
                ops.append('recursion')
                traverse(node.arg)
            elif hasattr(node, 'func'):
                ops.append(str(node.func))
        try:
            traverse(program)
        except:
            pass
        return ops
    
    def _classify_type(self, value: Any) -> str:
        """Classify value type for pattern matching."""
        if value is None:
            return 'none'
        if isinstance(value, bool):
            return 'bool'
        if isinstance(value, int):
            return 'int'
        if isinstance(value, float):
            return 'float'
        if isinstance(value, list):
            if value and isinstance(value[0], list):
                return 'matrix'
            return 'list'
        if isinstance(value, dict):
            return 'error' if '__error__' in value else 'dict'
        return 'unknown'
    
    def _generate_hypothesis(self, error_type: str, ops: List[str], 
                              inp_type: str, exp_type: str, got_type: str) -> str:
        """
        GENUINE META-REASONING: Generate a hypothesis about WHY the failure occurred.
        """
        # ShapeError patterns
        if error_type == 'ShapeError':
            if got_type == 'list' and exp_type == 'int':
                if 'flatten' in str(ops) or 'reverse' in str(ops):
                    return f"HYPOTHESIS: Using list-returning op ({ops}) when Int expected. Need aggregator (sum/len/max)."
                return f"HYPOTHESIS: Type mismatch - program produces {got_type} but task expects {exp_type}. Missing scalar reduction."
        
        # TypeError patterns  
        if error_type == 'TypeError':
            if inp_type == 'matrix':
                return f"HYPOTHESIS: Operators incompatible with matrix input. Need matrix-aware ops (matrix_sum, flatten first)."
            if inp_type == 'list':
                return f"HYPOTHESIS: Operators may be treating list elements incorrectly. Check element-wise vs aggregate ops."
        
        # NoneReturn patterns
        if error_type == 'NoneReturn':
            return f"HYPOTHESIS: Operator returned None - likely out-of-bounds access or invalid operation. Check input bounds."
        
        # Recursion patterns
        if 'recursion' in ops and error_type in ['RecursionError', 'Timeout']:
            return f"HYPOTHESIS: Recursion not terminating. Base case may be missing or wrong."
        
        # Default
        return f"HYPOTHESIS: Error {error_type} with ops {ops}. Input={inp_type}, Expected={exp_type}, Got={got_type}."
    
    def _recommend_action(self, hypothesis: str, ops: List[str]) -> str:
        """
        Recommend a concrete action to fix the strategy.
        """
        if 'Missing scalar reduction' in hypothesis or 'Need aggregator' in hypothesis:
            return "ACTION: Force scalar_goal=True for Int outputs. Ban list-returning ops at root."
        
        if 'matrix-aware ops' in hypothesis:
            return "ACTION: Add matrix operators (matrix_sum, flatten) to search space for matrix inputs."
        
        if 'Recursion not terminating' in hypothesis:
            return "ACTION: Reduce max recursion depth. Increase base case weight."
        
        if 'out-of-bounds' in hypothesis:
            return "ACTION: Add bounds-checking wrappers. Prefer safe-access primitives."
        
        return "ACTION: Log pattern for further analysis. No immediate strategy change."
    
    def get_strategy_adjustments(self) -> Dict[str, Any]:
        """
        Based on accumulated failure patterns, return recommended strategy changes.
        This is called periodically to update the search strategy.
        """
        adjustments = {}
        
        total_errors = sum(self.error_counts.values())
        if total_errors == 0:
            return adjustments
        
        # If ShapeError dominates (even 30%), force scalar constraints
        shape_ratio = self.error_counts['ShapeError'] / total_errors
        if shape_ratio > 0.3:  # Lowered from 0.5
            adjustments['force_scalar_root'] = True
            adjustments['ban_list_ops_at_root'] = True
            print(f"[META] ShapeError ratio {shape_ratio:.1%} > 30%, forcing scalar root")
        
        # If ValueError dominates, we have type mismatches
        value_ratio = self.error_counts.get('ValueError', 0) / total_errors
        if value_ratio > 0.3:
            adjustments['increase_type_strictness'] = True
            print(f"[META] ValueError ratio {value_ratio:.1%} > 30%, increasing type strictness")
        
        # If NoneReturn dominates, operators are returning None
        none_ratio = self.error_counts.get('NoneReturn', 0) / total_errors
        if none_ratio > 0.3:
            adjustments['ban_unsafe_ops'] = True
            print(f"[META] NoneReturn ratio {none_ratio:.1%} > 30%, banning unsafe ops")
        
        # Find operators that consistently fail (LOOSENED: 5 failures, 50% dominance)
        for op, error_map in self.op_failure_map.items():
            total_op_failures = sum(error_map.values())
            if total_op_failures > 5:  # Lowered from 10
                dominant_error = max(error_map, key=error_map.get)
                if error_map[dominant_error] / total_op_failures > 0.5:  # Lowered from 0.8
                    adjustments[f'reduce_weight_{op}'] = 0.5  # Reduce by 50%, not 90%
                    print(f"[META] Operator '{op}' fails with {dominant_error} {error_map[dominant_error]}/{total_op_failures} times, reducing weight")
        
        # [C] Analyze type_mismatch_patterns → ban_list_producers
        if len(self.type_mismatch_patterns) > 10:
            # Count list→int patterns (input=list, expected=int, got=list)
            list_to_int_fails = sum(1 for inp, got, exp in self.type_mismatch_patterns 
                                     if inp == 'list' and exp == 'int' and got == 'list')
            pattern_ratio = list_to_int_fails / len(self.type_mismatch_patterns)
            
            if pattern_ratio > 0.3:  # 30% threshold
                adjustments['ban_list_producers'] = True
                adjustments['list_producer_ops'] = {'reverse', 'split_half', 'init', 'tail', 'sort', 'filter_fn'}
                print(f"[META] list→int pattern {pattern_ratio:.1%} > 30%, banning list producers: {adjustments['list_producer_ops']}")
        
        return adjustments
    
    def print_reasoning_summary(self):
        """Print a summary of meta-reasoning for debugging."""
        print("\n=== META-REASONING SUMMARY ===")
        print(f"Total errors analyzed: {sum(self.error_counts.values())}")
        print(f"Error distribution: {dict(self.error_counts)}")
        if self.hypothesis_log:
            print(f"Recent hypotheses:")
            for h in self.hypothesis_log[-3:]:
                print(f"  - {h}")
        adjustments = self.get_strategy_adjustments()
        if adjustments:
            print(f"Recommended adjustments: {adjustments}")
        print("==============================\n")


class MetaHeuristic:
    """
    Real RSI Component: Meta-Heuristic Search Engine.
    
    Implements Condition (A): "Improving its own learning/search/evaluation algorithms".
    This class maintains a set of feature weights that evolve over time based on
    successful synthesis ("reinforcement learning on search strategy").
    
    It allows the system to 'learn to search' better, finding solutions faster
    as it gains experience, without any hardcoded cheats.
    """
    WEIGHTS_FILE = "rsi_meta_weights.json"
    DEFAULT_WEIGHTS = {
        'recursion': 1.0,
        'op_plus': 1.0,
        'op_minus': 1.0,
        'op_mult': 1.0,
        'depth_penalty': 0.1,  # penalize deep trees slightly
        'learning_rate': 0.1
    }
    
    def __init__(self, load_path="rsi_meta_weights.json", no_io=False):
        self.WEIGHTS_FILE = load_path
        self.no_io = no_io
        
        if self.no_io:
            self.weights = dict(self.DEFAULT_WEIGHTS)
        else:
            self.weights = self._load_weights()
            if not self.weights:
                self.weights = dict(self.DEFAULT_WEIGHTS)
    
    def _load_weights(self) -> Dict[str, float]:
        if self.no_io: return {}
        if os.path.exists(self.WEIGHTS_FILE):
            try:
                with open(self.WEIGHTS_FILE, 'r') as f:
                    data = json.load(f)
                return self._sanitize_weights(data)
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                return {}
        return {}
    
    def _save_weights(self):
        directory = os.path.dirname(self.WEIGHTS_FILE) or "."
        fd, temp_path = tempfile.mkstemp(prefix="meta_weights_", dir=directory)
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(self.weights, f, indent=2)
            os.replace(temp_path, self.WEIGHTS_FILE)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _sanitize_weights(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Ensure loaded weights are numeric and sane."""
        if not isinstance(data, dict):
            return {}
        sanitized = dict(self.DEFAULT_WEIGHTS)
        for key, value in data.items():
            # [RSI] Allow organic expansion - do not filter by DEFAULT_WEIGHTS keys
            if isinstance(value, (int, float)):
                sanitized[key] = float(value)
        return sanitized
            
    def get_score(self, program: Any) -> float:
        """
        Calculate heuristic score for a program.
        Higher score = more promising.
        """
        score = 0.0
        
        # Feature extraction
        features = self._extract_features(program)
        
        # Linear combination
        score += features.get('recursion', 0) * self.weights.get('recursion', 1.0)
        score += features.get('op_plus', 0) * self.weights.get('op_plus', 1.0)
        score += features.get('op_minus', 0) * self.weights.get('op_minus', 1.0)
        score += features.get('op_mult', 0) * self.weights.get('op_mult', 1.0)
        
        # Penalize size/depth if learned
        score -= features.get('size', 0) * self.weights.get('depth_penalty', 0.1)
        
        return score
        
    def _extract_features(self, program: Any) -> Dict[str, int]:
        features = defaultdict(int)
        
        if isinstance(program, (list, tuple)):
            features['size'] = len(program)
            for item in program:
                if isinstance(item, str):
                    features[item] += 1
            return dict(features)
            
        # [RSI] Handle String Programs (NeuroGeneticSynthesizer v2 AST-String)
        if isinstance(program, str):
            # Extract all identifiers
            tokens = re.findall(r'[a-zA-Z_]\w*', program)
            features['size'] = len(tokens)
            for token in tokens:
                features[token] += 1
            return dict(features)
            
        features = {'size': 0, 'recursion': 0, 'op_plus': 0, 'op_minus': 0, 'op_mult': 0}
        
        def traverse(node):
            features['size'] += 1
            node_type = type(node).__name__
            
            if node_type == 'BSRecCall':
                features['recursion'] += 1
                traverse(node.arg)
            elif node_type == 'BSBinOp':
                if node.op == '+': features['op_plus'] += 1
                elif node.op == '-': features['op_minus'] += 1
                elif node.op == '*': features['op_mult'] += 1
                traverse(node.left)
                traverse(node.right)
            # BSVar and BSVal don't add specific features here yet
            
        traverse(program)
        return features

    def get_op_weights(self, op_names: List[str]) -> Dict[str, float]:
        """
        Return meta-learned weights for specific operators.
        Used for multiplicative merge with library weights.
        
        Returns dict mapping op_name -> weight (default 1.0 if unknown).
        """
        result = {}
        for op in op_names:
            # Check if we have learned weight for this op
            if op in self.weights:
                result[op] = self.weights[op]
            else:
                # Default weight 1.0 (neutral)
                result[op] = 1.0
        return result

    def learn(self, successful_program: Any):
        """
        Update weights based on success.
        Reinforcement Learning: Valid solution = Positive Reward.
        """
        features = self._extract_features(successful_program)
        lr = self.weights.get('learning_rate', 0.1)
        
        # Increase weights for used features
        # Increase weights for used features
        for feat, count in features.items():
            if count > 0:
                if feat not in self.weights:
                    self.weights[feat] = 1.0
                # Gradient ascent-ish
                self.weights[feat] += lr * count
        
        # Normalize to prevent explosion (optional, but good practice)
        # For simple version, we just save.
        self._save_weights()
        print(f"[RSI-Meta] 🧠 Updated Search Heuristics: {self.weights}")

    def learn_failure(self, failed_program: Any, failure_type: str = "LOW_SCORE_VALID", context: Dict = None):
        """
        [TRUE RSI] Update weights based on FAILURE with taxonomy.
        
        failure_type must be one of:
        - TYPE_OR_SHAPE: Type mismatch or shape error
        - EXCEPTION: Runtime exception (ValueError, TypeError, etc.)
        - LOW_SCORE_VALID: Valid execution but wrong result
        """
        # Track failure by type
        if not hasattr(self, 'failure_counts'):
            self.failure_counts = {'TYPE_OR_SHAPE': 0, 'EXCEPTION': 0, 'LOW_SCORE_VALID': 0}
        self.failure_counts[failure_type] = self.failure_counts.get(failure_type, 0) + 1
        
        features = self._extract_features(failed_program)
        base_lr = self.weights.get('learning_rate', 0.1)
        
        # Different penalty multipliers per failure type
        penalty_mult = {
            'TYPE_OR_SHAPE': 0.3,    # Severe: reduce weight significantly
            'EXCEPTION': 0.2,         # Very severe: reduce more
            'LOW_SCORE_VALID': 0.05   # Mild: small reduction
        }
        lr = base_lr * penalty_mult.get(failure_type, 0.1)
        
        # Decrease weights for features in failed programs
        # Decrease weights for features in failed programs
        for feat, count in features.items():
            if count > 0 and feat != 'learning_rate':
                if feat not in self.weights:
                    self.weights[feat] = 1.0
                self.weights[feat] = max(0.01, self.weights[feat] - lr * count)
        
        # Increase depth_penalty if failed program was deep
        if features.get('size', 0) > 5:
            self.weights['depth_penalty'] = min(1.0, self.weights.get('depth_penalty', 0.1) + lr)
        
        # Track which ops failed for later banning
        if context and 'ops_used' in context:
            if not hasattr(self, 'failed_ops'):
                self.failed_ops = {}
            for op in context['ops_used']:
                if op not in self.failed_ops:
                    self.failed_ops[op] = {'TYPE_OR_SHAPE': 0, 'EXCEPTION': 0, 'LOW_SCORE_VALID': 0}
                self.failed_ops[op][failure_type] += 1
        
        # Save updated weights to disk (Critical for Treatment group learning)
        self._save_weights()




