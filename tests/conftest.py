import json
from pathlib import Path

import pytest
import torch

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# FP32 references must be true FP32. TF32 would add ~1e-3 relative error to the
# reference itself and hide kernel bugs.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def pytest_collection_modifyitems(config, items):
    if not torch.cuda.is_available():
        skip = pytest.mark.skip(reason="CUDA GPU required")
        for item in items:
            item.add_marker(skip)


class ErrorLog:
    """Collects per-case error stats and writes them to results/ at session end."""

    def __init__(self):
        self.records: dict[str, list[dict]] = {}

    def add(self, suite: str, **record):
        self.records.setdefault(suite, []).append(record)

    def dump(self):
        RESULTS_DIR.mkdir(exist_ok=True)
        for suite, recs in self.records.items():
            worst: dict = {}
            for r in recs:
                key = r["dtype"]
                w = worst.setdefault(key, {"max_abs_err": 0.0, "max_rel_err": 0.0, "cases": 0})
                w["max_abs_err"] = max(w["max_abs_err"], r["max_abs_err"])
                w["max_rel_err"] = max(w["max_rel_err"], r["max_rel_err"])
                w["cases"] += 1
            out = {
                "suite": suite,
                "total_cases": len(recs),
                "passed_cases": sum(r["passed"] for r in recs),
                "worst_by_dtype": worst,
                "rel_err_definition": "max|out - ref| / max|ref| over the whole output tensor",
                "cases": recs,
            }
            (RESULTS_DIR / f"correctness_{suite}.json").write_text(
                json.dumps(out, indent=2), encoding="utf-8"
            )


@pytest.fixture(scope="session")
def error_log():
    log = ErrorLog()
    yield log
    log.dump()
