"""
WATCHDOG EXECUTOR - Process-Isolated Code Execution for Safe RSI
Uses multiprocessing for TRUE isolation, not language restriction.

Key Features:
- Spawns separate process for code execution
- Timeout mechanism to kill infinite loops
- Full exec() freedom inside isolated process
- Captures stdout and return values safely
"""

import multiprocessing
import sys
import io
import traceback
from typing import Dict, Any, Optional


class WatchdogExecutor:
    """
    Executes untrusted code in an isolated subprocess with timeout protection.
    
    This is the "Koala Watchdog" pattern - the main process is protected
    from crashes, infinite loops, and malicious code because execution
    happens in a completely separate process.
    
    Note: This provides PROCESS isolation, not LANGUAGE restriction.
    The child process has full exec() capabilities.
    """
    
    DEFAULT_TIMEOUT = 2.0
    
    def __init__(self, timeout: float = DEFAULT_TIMEOUT):
        """
        Initialize the WatchdogExecutor.
        
        Args:
            timeout: Maximum seconds to wait before killing the process
        """
        self.timeout = timeout
    
    @staticmethod
    def _target_runner(code: str, return_dict: dict) -> None:
        """
        Internal function that runs inside the child process.
        
        This is where the actual exec() happens, completely isolated
        from the main process. Crashes here don't affect the parent.
        
        Args:
            code: Python code string to execute
            return_dict: Shared dict for returning results to parent
        """
        # Capture stdout to see what the code prints
        captured_stdout = io.StringIO()
        original_stdout = sys.stdout
        
        try:
            sys.stdout = captured_stdout
            
            # Create execution scope
            local_scope = {}
            global_scope = {
                '__builtins__': __builtins__,
                '__name__': '__watchdog_child__',
            }
            
            # UNRESTRICTED EXECUTION - process isolation provides safety
            exec(code, global_scope, local_scope)
            
            # Check for common entry points
            result = None
            if 'solve' in local_scope:
                result = local_scope['solve']()
            elif 'main' in local_scope:
                result = local_scope['main']()
            elif 'result' in local_scope:
                result = local_scope['result']
            
            return_dict['success'] = True
            return_dict['result'] = result
            return_dict['output'] = captured_stdout.getvalue()
            return_dict['local_scope'] = {
                k: v for k, v in local_scope.items() 
                if not k.startswith('_') and _is_serializable(v)
            }
            
        except Exception:
            return_dict['success'] = False
            return_dict['error'] = traceback.format_exc()
            return_dict['output'] = captured_stdout.getvalue()
            
        finally:
            sys.stdout = original_stdout
    
    def run_safe(self, code: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Execute code in an isolated subprocess with timeout protection.
        
        Args:
            code: Python code string to execute
            timeout: Override default timeout (seconds)
            
        Returns:
            Dict with keys:
                - success: bool - whether execution completed without error
                - result: Any - return value from solve()/main()/result
                - output: str - captured stdout
                - error: str - traceback if failed
                - killed: bool - True if process was killed due to timeout
        """
        timeout = timeout or self.timeout
        
        # Create shared state with Manager
        with multiprocessing.Manager() as manager:
            return_dict = manager.dict()
            return_dict['success'] = False
            return_dict['error'] = 'Unknown fatal error (process may have crashed)'
            return_dict['output'] = ''
            return_dict['result'] = None
            
            # Spawn isolated process
            process = multiprocessing.Process(
                target=self._target_runner,
                args=(code, return_dict)
            )
            process.start()
            
            # Wait for completion with timeout
            process.join(timeout)
            
            # Check if process is still running (infinite loop or hang)
            if process.is_alive():
                # KILL IT - no mercy for infinite loops
                process.terminate()
                process.join(0.5)  # Give it a moment to terminate gracefully
                
                if process.is_alive():
                    # Still alive? Force kill.
                    process.kill()
                    process.join()
                
                return {
                    'success': False,
                    'error': '🐨 Koala Watchdog: Process killed due to timeout (Infinite Loop detected)',
                    'output': '(Process terminated)',
                    'result': None,
                    'killed': True,
                }
            
            # Process completed - return captured state
            result = dict(return_dict)
            result['killed'] = False
            return result
    
    def validate_code(self, code: str, test_inputs: list = None) -> Dict[str, Any]:
        """
        Validate code by running it with optional test inputs.
        
        Args:
            code: Python code to validate
            test_inputs: Optional list of test cases
            
        Returns:
            Validation result dict
        """
        # First, check if code at least compiles
        try:
            compile(code, '<watchdog>', 'exec')
        except SyntaxError as e:
            return {
                'success': False,
                'error': f'Syntax Error: {e}',
                'valid': False,
            }
        
        # Run in sandbox
        result = self.run_safe(code)
        result['valid'] = result['success']
        return result


def _is_serializable(obj: Any) -> bool:
    """Check if an object can be safely passed through multiprocessing."""
    try:
        import pickle
        pickle.dumps(obj)
        return True
    except:
        return False


# Singleton instance for convenience
_default_executor = None

def get_watchdog() -> WatchdogExecutor:
    """Get the default WatchdogExecutor instance."""
    global _default_executor
    if _default_executor is None:
        _default_executor = WatchdogExecutor()
    return _default_executor


def run_safe(code: str, timeout: float = 2.0) -> Dict[str, Any]:
    """Convenience function to run code safely."""
    return get_watchdog().run_safe(code, timeout)


# =============================================================================
# INTEGRATION EXAMPLE
# =============================================================================
if __name__ == '__main__':
    # This block is REQUIRED for multiprocessing on Windows/macOS
    
    print("=" * 60)
    print("WATCHDOG EXECUTOR TEST SUITE")
    print("=" * 60)
    
    executor = WatchdogExecutor(timeout=2.0)
    
    # Test 1: Normal execution
    print("\n[Test 1] Normal code execution:")
    code1 = """
def solve():
    return 2 + 2
print("Calculating...")
"""
    result1 = executor.run_safe(code1)
    print(f"  Success: {result1['success']}")
    print(f"  Result: {result1.get('result')}")
    print(f"  Output: {result1.get('output', '').strip()}")
    
    # Test 2: Infinite loop (should be killed)
    print("\n[Test 2] Infinite loop (should timeout):")
    code2 = """
while True:
    pass
"""
    result2 = executor.run_safe(code2)
    print(f"  Success: {result2['success']}")
    print(f"  Killed: {result2.get('killed')}")
    print(f"  Error: {result2.get('error', '')[:60]}...")
    
    # Test 3: Runtime error
    print("\n[Test 3] Runtime error:")
    code3 = """
def solve():
    return 1 / 0
"""
    result3 = executor.run_safe(code3)
    print(f"  Success: {result3['success']}")
    print(f"  Error contains ZeroDivision: {'ZeroDivision' in result3.get('error', '')}")
    
    # Test 4: Import (should work - no restrictions)
    print("\n[Test 4] Import (unrestricted):")
    code4 = """
import math
result = math.sqrt(16)
"""
    result4 = executor.run_safe(code4)
    print(f"  Success: {result4['success']}")
    print(f"  Result: {result4.get('local_scope', {}).get('result')}")
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
