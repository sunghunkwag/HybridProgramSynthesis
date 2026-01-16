"""
LIBRARY MANAGER - Hierarchical DAG with Quality Gate
Pillars 2, 3, 4, 5 of Safe & Genuine RSI

Manages learned primitives with:
- Semantic deduplication (Pillar 2)
- DAG structure with level constraints (Pillar 3)
- Weighted probabilistic sampling (Pillar 4)
- JSON registry persistence (Pillar 5)
"""

import json
import os
import random
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple

from safe_interpreter import DSLExpr, DSLVar, DSLVal, DSLApp, SemanticHasher, UtilityScorer


@dataclass
class Primitive:
    """
    A learned primitive in the library.
    
    Attributes:
        name: Unique identifier
        level: DAG level (0 = atomic, 1+ = composite)
        expr: DSL expression tree
        semantic_hash: For deduplication
        dependencies: Set of primitives this one calls
        usage_weight: For probabilistic sampling (higher = more likely)
        usage_count: Times used in successful solutions
        created_at: Timestamp
    """
    name: str
    level: int
    expr: DSLExpr
    semantic_hash: str
    dependencies: Set[str] = field(default_factory=set)
    usage_weight: float = 1.0
    usage_count: int = 0
    compression_ratio: float = 1.0
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        """Serialize to JSON-compatible dict."""
        return {
            'name': self.name,
            'level': self.level,
            'expr': self._serialize_expr(self.expr),
            'semantic_hash': self.semantic_hash,
            'dependencies': list(self.dependencies),
            'usage_weight': self.usage_weight,
            'usage_count': self.usage_count,
            'compression_ratio': self.compression_ratio,
            'created_at': self.created_at,
        }
    
    @staticmethod
    def from_dict(d: Dict) -> 'Primitive':
        """Deserialize from dict."""
        return Primitive(
            name=d['name'],
            level=d['level'],
            expr=Primitive._deserialize_expr(d['expr']),
            semantic_hash=d['semantic_hash'],
            dependencies=set(d.get('dependencies', [])),
            usage_weight=d.get('usage_weight', 1.0),
            usage_count=d.get('usage_count', 0),
            compression_ratio=d.get('compression_ratio', 1.0),
            created_at=d.get('created_at', time.time()),
        )
    
    @staticmethod
    def _serialize_expr(expr: DSLExpr) -> Dict:
        """Serialize DSLExpr to dict."""
        if isinstance(expr, DSLVal):
            return {'type': 'val', 'value': expr.value}
        elif isinstance(expr, DSLVar):
            return {'type': 'var', 'name': expr.name}
        elif isinstance(expr, DSLApp):
            return {
                'type': 'app',
                'func': expr.func,
                'args': [Primitive._serialize_expr(a) for a in expr.args]
            }
        return {'type': 'unknown'}
    
    @staticmethod
    def _deserialize_expr(d: Dict) -> DSLExpr:
        """Deserialize dict to DSLExpr."""
        if d['type'] == 'val':
            return DSLVal(d['value'])
        elif d['type'] == 'var':
            return DSLVar(d['name'])
        elif d['type'] == 'app':
            return DSLApp(
                d['func'],
                [Primitive._deserialize_expr(a) for a in d['args']]
            )
        return DSLVal(None)


class LibraryManager:
    """
    Manages the primitive library with quality control and DAG structure.
    
    Key features:
    - Semantic deduplication via hash
    - Level-based DAG (no circular dependencies)
    - Weighted sampling for evolution
    - JSON persistence (no source injection)
    """
    
    REGISTRY_PATH = "rsi_library_registry.json"
    WEIGHT_DECAY = 0.95  # Decay factor per cycle for unused primitives
    WEIGHT_BOOST = 1.5   # Boost factor when used successfully
    
    def __init__(self, registry_path: str = None):
        self.registry_path = registry_path or self.REGISTRY_PATH
        self.primitives: Dict[str, Primitive] = {}
        self._hash_index: Dict[str, str] = {}  # semantic_hash -> name
        self._level_index: Dict[int, Set[str]] = {}  # level -> set of names
        
        # Load existing registry
        self.load()
    
    # =========================================================================
    # Pillar 2: Quality Gate
    # =========================================================================
    
    def add_primitive(
        self, 
        name: str, 
        expr: DSLExpr, 
        dependencies: Set[str] = None,
        compression_ratio: float = 1.0,
    ) -> Tuple[bool, str]:
        """
        Add a new primitive with quality checks.
        
        Returns:
            (success, message)
        """
        dependencies = dependencies or set()
        
        # 1. Compute semantic hash
        semantic_hash = SemanticHasher.hash(expr)
        
        # 2. Check uniqueness
        if semantic_hash in self._hash_index:
            existing = self._hash_index[semantic_hash]
            return False, f"Duplicate of {existing}"
        
        # 3. Compute level from dependencies
        level = self._compute_level(dependencies)
        if level < 0:
            return False, "Invalid dependencies (circular or unknown)"
        
        # 4. Validate name
        if name in self.primitives:
            return False, f"Name already exists: {name}"
        
        # 5. Create and store primitive (initially with 0 usage)
        prim = Primitive(
            name=name,
            level=level,
            expr=expr,
            semantic_hash=semantic_hash,
            dependencies=dependencies,
            usage_weight=1.0,
            usage_count=0,
            compression_ratio=compression_ratio,
        )
        
        self.primitives[name] = prim
        self._hash_index[semantic_hash] = name
        
        if level not in self._level_index:
            self._level_index[level] = set()
        self._level_index[level].add(name)
        
        return True, f"Added {name} at level {level}"
    
    def mark_candidate(
        self, 
        name: str, 
        expr: DSLExpr, 
        original_size: int
    ) -> Tuple[bool, str]:
        """
        Add a candidate primitive (not yet permanent).
        
        Candidates are tracked but not persisted until they meet
        the usage threshold.
        """
        prim_size = UtilityScorer.compute_size(expr)
        compression = UtilityScorer.compression_ratio(original_size, prim_size)
        
        if compression < UtilityScorer.MIN_COMPRESSION_RATIO:
            return False, f"Compression too low: {compression:.2f}"
        
        # Find dependencies
        deps = self._extract_dependencies(expr)
        
        return self.add_primitive(
            name,
            expr,
            deps,
            compression_ratio=compression,
        )
    
    def record_usage(self, name: str, success: bool) -> None:
        """Record that a primitive was used (successfully or not)."""
        if name not in self.primitives:
            return
        
        prim = self.primitives[name]
        
        if success:
            prim.usage_count += 1
            prim.usage_weight *= self.WEIGHT_BOOST
        else:
            prim.usage_weight *= self.WEIGHT_DECAY
        
        # Clamp weight
        prim.usage_weight = max(0.1, min(10.0, prim.usage_weight))
    
    def should_persist(self, name: str) -> bool:
        """Check if primitive meets persistence threshold."""
        if name not in self.primitives:
            return False
        
        prim = self.primitives[name]
        return UtilityScorer.should_keep(
            compression_ratio=prim.compression_ratio,
            usage_count=prim.usage_count,
            is_unique=True,  # Already passed on add
        )
    
    def prune_unused(self) -> int:
        """Remove primitives that haven't met usage threshold after decay."""
        to_remove = []
        
        for name, prim in self.primitives.items():
            # Keep if level 0 (atomic) or meets threshold
            if prim.level == 0:
                continue
            if prim.usage_count < UtilityScorer.MIN_USAGE_COUNT:
                if prim.usage_weight < 0.3:  # Decayed too much
                    to_remove.append(name)
        
        for name in to_remove:
            self._remove_primitive(name)
        
        return len(to_remove)
    
    def decay_all_weights(self) -> None:
        """Apply decay to all primitives (call once per cycle)."""
        for prim in self.primitives.values():
            if prim.level > 0:  # Don't decay atomics
                prim.usage_weight *= self.WEIGHT_DECAY
                prim.usage_weight = max(0.1, prim.usage_weight)
    
    # =========================================================================
    # Pillar 3: DAG Structure
    # =========================================================================
    
    def _compute_level(self, dependencies: Set[str]) -> int:
        """
        Compute the level for a new primitive based on its dependencies.
        
        Returns -1 if dependencies are invalid (unknown or circular).
        """
        if not dependencies:
            return 1  # No deps = level 1 (simple composition of atomics)
        
        max_dep_level = 0
        for dep in dependencies:
            if dep not in self.primitives:
                # Check if built-in atomic
                from safe_interpreter import SafeInterpreter
                if dep in SafeInterpreter.ALLOWED_OPS:
                    max_dep_level = max(max_dep_level, 0)
                    continue  # Atomics are level 0
                return -1  # Unknown dependency
            
            dep_level = self.primitives[dep].level
            max_dep_level = max(max_dep_level, dep_level)
        
        return max_dep_level + 1
    
    def get_primitives_at_level(self, level: int) -> List[Primitive]:
        """Get all primitives at a specific level."""
        if level not in self._level_index:
            return []
        return [self.primitives[n] for n in self._level_index[level]]
    
    def get_available_for_level(self, max_level: int) -> List[str]:
        """Get all primitives available for use at a given level (lower levels only)."""
        result = []
        for lvl in range(max_level):
            if lvl in self._level_index:
                result.extend(self._level_index[lvl])
        return result
    
    def _extract_dependencies(self, expr: DSLExpr) -> Set[str]:
        """Extract all function calls from an expression."""
        deps = set()
        self._collect_deps(expr, deps)
        return deps
    
    def _collect_deps(self, expr: DSLExpr, deps: Set[str]) -> None:
        """Recursively collect dependencies."""
        if isinstance(expr, DSLApp):
            deps.add(expr.func)
            for arg in expr.args:
                self._collect_deps(arg, deps)
    
    # =========================================================================
    # Pillar 4: Weighted Sampling
    # =========================================================================
    
    def weighted_sample(self, n: int = 1, max_level: int = None) -> List[str]:
        """
        Sample primitives with probability proportional to usage_weight.
        
        Args:
            n: Number of primitives to sample
            max_level: Only sample from levels < max_level (for DAG constraint)
            
        Returns:
            List of primitive names
        """
        candidates = []
        weights = []
        
        for name, prim in self.primitives.items():
            if max_level is not None and prim.level >= max_level:
                continue
            candidates.append(name)
            weights.append(prim.usage_weight)
        
        if not candidates:
            return []
        
        # Normalize weights
        total = sum(weights)
        if total <= 0:
            return random.sample(candidates, min(n, len(candidates)))
        
        probs = [w / total for w in weights]
        
        # Sample with replacement
        result = []
        for _ in range(n):
            r = random.random()
            cumsum = 0
            for i, p in enumerate(probs):
                cumsum += p
                if r <= cumsum:
                    result.append(candidates[i])
                    break
        
        return result
    
    def get_weighted_ops(self) -> Dict[str, float]:
        """Get all ops with their weights for the synthesizer."""
        result = {}
        
        # Add atomics with weight 1.0
        from safe_interpreter import SafeInterpreter
        for op in SafeInterpreter.ALLOWED_OPS:
            result[op] = 1.0
        
        # Add learned primitives with their weights
        for name, prim in self.primitives.items():
            result[name] = prim.usage_weight
        
        return result
    
    # =========================================================================
    # Pillar 5: Safe Persistence
    # =========================================================================
    
    def save(self) -> bool:
        """Save library to JSON registry (no source injection)."""
        try:
            # Only save primitives that meet threshold
            to_save = {
                name: prim.to_dict()
                for name, prim in self.primitives.items()
                if self.should_persist(name) or prim.level == 0
            }
            
            registry = {
                'version': '2.0',
                'timestamp': time.time(),
                'primitives': to_save,
            }
            
            with open(self.registry_path, 'w') as f:
                json.dump(registry, f, indent=2)
            
            print(f"[LibraryManager] Saved {len(to_save)} primitives to {self.registry_path}")
            return True
            
        except Exception as e:
            print(f"[LibraryManager] Save failed: {e}")
            return False
    
    def load(self) -> bool:
        """Load library from JSON registry."""
        if not os.path.exists(self.registry_path):
            return False
        
        try:
            with open(self.registry_path, 'r') as f:
                registry = json.load(f)
            
            version = registry.get('version', '1.0')
            primitives_data = registry.get('primitives', {})
            
            for name, data in primitives_data.items():
                prim = Primitive.from_dict(data)
                self.primitives[name] = prim
                self._hash_index[prim.semantic_hash] = name
                
                if prim.level not in self._level_index:
                    self._level_index[prim.level] = set()
                self._level_index[prim.level].add(name)
            
            print(f"[LibraryManager] Loaded {len(self.primitives)} primitives (v{version})")
            return True
            
        except Exception as e:
            print(f"[LibraryManager] Load failed: {e}")
            return False
    
    def _remove_primitive(self, name: str) -> None:
        """Remove a primitive from all indices."""
        if name not in self.primitives:
            return
        
        prim = self.primitives[name]
        
        # Remove from indices
        if prim.semantic_hash in self._hash_index:
            del self._hash_index[prim.semantic_hash]
        
        if prim.level in self._level_index:
            self._level_index[prim.level].discard(name)
        
        del self.primitives[name]
    
    # =========================================================================
    # Stats
    # =========================================================================
    
    def stats(self) -> Dict[str, Any]:
        """Get library statistics."""
        level_counts = {}
        for lvl, names in self._level_index.items():
            level_counts[lvl] = len(names)
        
        return {
            'total': len(self.primitives),
            'by_level': level_counts,
            'avg_weight': (
                sum(p.usage_weight for p in self.primitives.values()) / len(self.primitives)
                if self.primitives else 0
            ),
            'avg_usage': (
                sum(p.usage_count for p in self.primitives.values()) / len(self.primitives)
                if self.primitives else 0
            ),
        }


if __name__ == "__main__":
    # Quick test
    lib = LibraryManager("test_registry.json")
    
    # Add a simple primitive: double(x) = add(x, x)
    expr = DSLApp('add', [DSLVar('var_0'), DSLVar('var_0')])
    success, msg = lib.add_primitive('double', expr, {'add'})
    print(f"Add double: {success} - {msg}")
    
    # Simulate usage
    for _ in range(5):
        lib.record_usage('double', success=True)
    
    print(f"Should persist: {lib.should_persist('double')}")
    print(f"Stats: {lib.stats()}")
    
    # Save and reload
    lib.save()
    
    lib2 = LibraryManager("test_registry.json")
    print(f"Loaded: {lib2.stats()}")
    
    # Cleanup
    os.remove("test_registry.json")
