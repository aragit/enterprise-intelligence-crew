from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor


class TestConcurrency:

    def test_concurrent_pipelines_no_race_condition(self, mock_intelligence_crew):
        crew = mock_intelligence_crew
        results = []

        def run_pipeline(query):
            result = crew.run_pipeline(query)
            results.append(result)

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(run_pipeline, f"query_{i}") for i in range(3)]
            for f in futures:
                f.result()

        assert len(results) == 3

    def test_file_lock_no_deadlock(self, tmp_path):
        from src.api.routes import _save_task_store
        errors = []

        def writer(idx):
            path = tmp_path / f"tasks_{idx}.json"
            try:
                _save_task_store({f"task_{idx}": {"status": "done"}}, store_path=path)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0, f"Errors during concurrent writes: {errors}"
