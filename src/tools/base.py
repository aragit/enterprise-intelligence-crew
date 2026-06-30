from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    success: bool
    data: Any = None
    error: str | None = None
    duration_ms: float = 0.0


def retry(max_attempts: int = 3, delay: float = 1.0):
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_result: ToolResult | None = None
            for attempt in range(1, max_attempts + 1):
                start = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    result = ToolResult(success=False, error=str(e))
                elapsed = (time.perf_counter() - start) * 1000
                if isinstance(result, ToolResult):
                    result.duration_ms = elapsed
                if result.success:
                    return result
                last_result = result
                if attempt < max_attempts:
                    time.sleep(delay)
            return last_result or ToolResult(success=False, error="max retries exceeded")
        return wrapper
    return decorator
