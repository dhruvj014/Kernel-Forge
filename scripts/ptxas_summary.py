"""Parses `-Xptxas -v` output from build logs into results/ptxas.json.

Records registers, shared memory, spills and constant memory per kernel so
each kernel version's resource usage is tracked alongside its benchmarks.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FUNC_RE = re.compile(r"Compiling entry function '(\w+)' for '(sm_\d+)'")
SPILL_RE = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")
USED_RE = re.compile(r"Used (\d+) registers(?:, used (\d+) barriers)?(.*)")
SMEM_RE = re.compile(r"(\d+) bytes smem")
CMEM_RE = re.compile(r"(\d+) bytes cmem\[0\]")


def demangle(names: list[str]) -> dict[str, str]:
    try:
        out = subprocess.run(["cu++filt"], input="\n".join(names), capture_output=True, text=True, check=True)
        return dict(zip(names, out.stdout.splitlines(), strict=False))
    except (OSError, subprocess.SubprocessError):
        return {}


def parse(text: str) -> dict:
    kernels: dict = {}
    current = None
    for line in text.splitlines():
        if m := FUNC_RE.search(line):
            current = m.group(1)
            kernels[current] = {"arch": m.group(2)}
        elif current and (m := SPILL_RE.search(line)):
            kernels[current].update(stack_bytes=int(m.group(1)), spill_store_bytes=int(m.group(2)),
                                    spill_load_bytes=int(m.group(3)))
        elif current and (m := USED_RE.search(line)):
            info = kernels[current]
            info["registers"] = int(m.group(1))
            if m.group(2):
                info["barriers"] = int(m.group(2))
            rest = m.group(3)
            if s := SMEM_RE.search(rest):
                info["smem_static_bytes"] = int(s.group(1))
            if c := CMEM_RE.search(rest):
                info["cmem0_bytes"] = int(c.group(1))
            current = None
    return kernels


def main() -> None:
    out = ROOT / "results" / "ptxas.json"
    # Merge with earlier results: an incremental build only recompiles (and
    # only prints ptxas info for) the files that changed.
    kernels: dict = json.loads(out.read_text(encoding="utf-8"))["kernels"] if out.exists() else {}
    for log in sys.argv[1:]:
        p = Path(log)
        if p.exists():
            kernels.update(parse(p.read_text(encoding="utf-8", errors="replace")))
    names = demangle(list(kernels))
    for mangled, info in kernels.items():
        info["demangled"] = names.get(mangled, info.get("demangled", mangled))
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"kernels": kernels}, indent=2), encoding="utf-8")
    for info in kernels.values():
        print(f"  {info['demangled'][:70]:70s} regs={info.get('registers')} smem={info.get('smem_static_bytes', 0)} "
              f"spills={info.get('spill_store_bytes', 0)}")
    print(f"wrote {out} ({len(kernels)} kernels)")


if __name__ == "__main__":
    main()
