#!/usr/bin/env python3
"""Record the software and model versions behind the outputs.

Writes OUT/environment.json. The Hugging Face models are identified by the
commit of the snapshot in the local cache, which is the exact revision used.
"""
from __future__ import annotations

import json
import platform
import subprocess
from importlib import metadata
from pathlib import Path

from paths import OUT, SEED

PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "spacy", "torch", "transformers",
            "sentence-transformers", "matplotlib"]
HF_MODELS = ["cardiffnlp/twitter-xlm-roberta-base-sentiment",
             "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"]


def hf_revision(model: str) -> str | None:
    root = Path.home() / ".cache" / "huggingface" / "hub" / ("models--" + model.replace("/", "--"))
    ref = root / "refs" / "main"
    return ref.read_text().strip() if ref.exists() else None


def main() -> None:
    import spacy
    env = dict(python=platform.python_version(), platform=platform.platform(), seed=SEED,
               packages={p: metadata.version(p) for p in PACKAGES},
               spacy_models={m: spacy.load(m).meta["version"]
                             for m in ("en_core_web_sm", "de_core_news_sm")},
               hf_models={m: hf_revision(m) for m in HF_MODELS})
    try:
        env["tesseract"] = subprocess.run(["tesseract", "--version"], capture_output=True,
                                          text=True).stdout.splitlines()[0]
    except (OSError, IndexError):
        env["tesseract"] = None
    (OUT / "environment.json").write_text(json.dumps(env, indent=1), encoding="utf8")
    print(json.dumps(env, indent=1))


if __name__ == "__main__":
    main()
