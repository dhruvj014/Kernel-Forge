"""Downloads the Phase 5 models and dataset into the project-local Hugging Face cache.

Everything goes to <repo>/.cache/hf (gitignored). Only the files needed to run
the models in PyTorch are fetched: safetensors weights, config, tokenizer and
generation config, not duplicate .bin / GGUF / ONNX exports.

  .venv\\Scripts\\python.exe scripts\\download_assets.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

MODELS = ["TinyLlama/TinyLlama-1.1B-Chat-v1.0", "Qwen/Qwen2.5-1.5B"]
ALLOW = ["*.safetensors", "*.safetensors.index.json", "config.json", "generation_config.json",
         "tokenizer*", "special_tokens_map.json", "vocab.json", "merges.txt"]


def main() -> None:
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    for repo in MODELS:
        path = snapshot_download(repo, allow_patterns=ALLOW)
        size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
        print(f"{repo}: {size / 2**30:.2f} GiB at {path}", flush=True)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    print(f"wikitext-2-raw-v1 test: {len(ds)} rows", flush=True)
    print(f"HF_HOME={os.environ['HF_HOME']}")


if __name__ == "__main__":
    sys.exit(main())
