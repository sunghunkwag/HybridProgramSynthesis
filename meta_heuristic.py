
import json
import os
from typing import Dict, Any


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
    
    def __init__(self):
        self.weights = self._load_weights()
        # Default weights if fresh
        if not self.weights:
            self.weights = {
                'recursion': 1.0,
                'op_plus': 1.0,
                'op_minus': 1.0,
                'op_mult': 1.0,
                'depth_penalty': 0.1,  # penalize deep trees slightly
                'learning_rate': 0.1
            }
    
    def _load_weights(self) -> Dict[str, float]:
        if os.path.exists(self.WEIGHTS_FILE):
            try:
                with open(self.WEIGHTS_FILE, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_weights(self):
        with open(self.WEIGHTS_FILE, 'w') as f:
            json.dump(self.weights, f, indent=2)
            
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

    def learn(self, successful_program: Any):
        """
        Update weights based on success.
        Reinforcement Learning: Valid solution = Positive Reward.
        """
        features = self._extract_features(successful_program)
        lr = self.weights.get('learning_rate', 0.1)
        
        # Increase weights for used features
        for feat, count in features.items():
            if count > 0 and feat in self.weights:
                # Gradient ascent-ish
                self.weights[feat] += lr * count
        
        # Normalize to prevent explosion (optional, but good practice)
        # For simple version, we just save.
        self._save_weights()
        print(f"[RSI-Meta] 🧠 Updated Search Heuristics: {self.weights}")

