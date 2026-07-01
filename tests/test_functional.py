from __future__ import annotations

import subprocess
import sys


class TestFunctional:
    def test_functional_pipeline(self):
        result = subprocess.run(
            [sys.executable, "functional_test.py"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr[:500], file=sys.stderr)
        assert result.returncode == 0, (
            f"functional_test.py failed (exit {result.returncode}): "
            f"{result.stderr[-500:]}"
        )
