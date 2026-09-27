"""Reproduce the linear-surrogate inference timings reported in the manuscript."""

from pathlib import Path
import statistics
import time

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATHS = {
    "Elastic Net (selected)": ROOT / "models" / "ElasticNet_Optimized.joblib",
    "PLS benchmark": ROOT / "models" / "PLS_Regression.joblib",
}
N_FEATURES = 72
N_CALLS_PER_BATCH = 1000
N_BATCHES = 5


def main() -> None:
    x = np.zeros((1, N_FEATURES), dtype=float)
    for name, path in MODEL_PATHS.items():
        model = joblib.load(path)
        model.predict(x)  # warm-up
        timings_ms = []
        for _ in range(N_BATCHES):
            start = time.perf_counter()
            for _ in range(N_CALLS_PER_BATCH):
                model.predict(x)
            timings_ms.append((time.perf_counter() - start) * 1_000 / N_CALLS_PER_BATCH)
        print(f"{name}: batch means (ms/prediction): {timings_ms}")
        print(f"{name}: median (ms/prediction): {statistics.median(timings_ms):.6f}")


if __name__ == "__main__":
    main()
