"""Download and cache all datasets and models used by the experiments.

Usage:
    .venv/bin/python experiments/prepare_data.py

Caches (idempotent, total well under 500 MB):
  data/ag_news_train.csv            AG News train split (~29 MB)
  data/sklearn_20ng/                20 Newsgroups via scikit-learn (~14 MB)
  data/hf/                          HF models:
      cross-encoder/nli-deberta-v3-xsmall   (~275 MB)
      sentence-transformers/all-MiniLM-L6-v2 (~90 MB)
"""

import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import DATA_DIR, EMBED_MODEL, NLI_MODEL, ROOT  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    print("== 20 Newsgroups ==")
    from semreuse.corpus import load_20newsgroups

    c = load_20newsgroups(str(DATA_DIR))
    print(f"   {len(c)} documents, {len(c.leaves)} leaves, "
          f"{len(c.groups)} groups")

    print("== AG News ==")
    from semreuse.corpus import load_agnews

    c = load_agnews(str(DATA_DIR), max_rows=40000)
    print(f"   {len(c)} documents (capped), {len(c.leaves)} classes")

    print("== NLI cross-encoder ==")
    from sentence_transformers import CrossEncoder

    CrossEncoder(NLI_MODEL)
    print(f"   {NLI_MODEL} cached")

    print("== Sentence embedder (baseline) ==")
    from sentence_transformers import SentenceTransformer

    SentenceTransformer(EMBED_MODEL)
    print(f"   {EMBED_MODEL} cached")

    du = subprocess.run(["du", "-sh", str(DATA_DIR)], capture_output=True,
                        text=True)
    print(f"== data/ footprint: {du.stdout.strip()} ==")


if __name__ == "__main__":
    main()
