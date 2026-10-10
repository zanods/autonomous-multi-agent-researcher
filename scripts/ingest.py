from __future__ import annotations

import argparse
import os
import sys

# Allow running as `python scripts/ingest.py` from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.vectorstore import build_vectorstore, default_persist_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest .txt/.md files from a folder into ChromaDB"
    )
    parser.add_argument("--src", default="./data", help="Folder with .txt/.md files")
    parser.add_argument("--dir", default=None, help="Chroma persist dir (default: ./chroma_db)")
    args = parser.parse_args()

    persist_dir = args.dir or default_persist_dir()
    texts, metas = [], []
    for fname in sorted(os.listdir(args.src)):
        if not fname.endswith((".txt", ".md")):
            continue
        with open(os.path.join(args.src, fname), encoding="utf-8") as f:
            content = f.read()
        for i, chunk in enumerate(c.strip() for c in content.split("\n\n") if c.strip()):
            texts.append(chunk)
            metas.append({"source": f"{fname}#chunk{i}"})

    if not texts:
        parser.error(f"no .txt/.md files found in {args.src}")

    build_vectorstore(texts, metas, persist_dir=persist_dir)
    print(f"ingested {len(texts)} chunks -> {persist_dir}")


if __name__ == "__main__":
    main()
