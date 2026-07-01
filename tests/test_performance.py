from __future__ import annotations

import time


class TestPerformance:

    def test_pipeline_latency_under_threshold(self, mock_intelligence_crew):
        crew = mock_intelligence_crew
        start = time.perf_counter()
        crew.run_pipeline("test query")
        elapsed = time.perf_counter() - start
        assert elapsed < 30.0, f"Pipeline too slow: {elapsed:.2f}s (threshold: 30s)"

    def test_build_agents_overhead(self, mock_intelligence_crew):
        crew = mock_intelligence_crew
        times = []
        for _ in range(5):
            start = time.perf_counter()
            crew.build_agents()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        avg = sum(times) / len(times)
        assert avg < 1.0, f"Agent build too slow: {avg:.3f}s"
