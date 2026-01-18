import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from Systemtest import InventionEvaluator, InventionTask


def test_invention_evaluator_runs_with_watchdog():
    evaluator = InventionEvaluator()
    task = InventionTask(kind="unit", input=1, expected=2)
    code = "def solve(task):\n    return task.expected\n"
    success, info = evaluator._run_in_subprocess(code, task, timeout=1.0)
    assert success, info


def test_invention_evaluator_times_out_with_watchdog():
    evaluator = InventionEvaluator()
    task = InventionTask(kind="unit", input=1, expected=2)
    code = "def solve(task):\n    while True:\n        pass\n"
    start = time.time()
    success, info = evaluator._run_in_subprocess(code, task, timeout=0.2)
    elapsed = time.time() - start
    assert not success
    assert "Watchdog timeout" in info
    assert elapsed < 2.0
