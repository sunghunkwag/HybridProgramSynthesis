import os
import re


def test_systemtest_has_no_raw_exec_eval():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Systemtest.py")
    with open(path, "r", encoding="utf-8") as handle:
        data = handle.read()
    assert re.search(r"\bexec\(", data) is None
    assert re.search(r"\beval\(", data) is None
