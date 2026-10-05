from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from scripts.collection.searchr1_corpus_index import build

ROOT = Path(__file__).resolve().parents[2]
SERVER_PATH = ROOT / "workspace" / "external" / "search-r1-repo" / "search_r1" / "search" / "retrieval_server.py"


def load_server_module():
    spec = importlib.util.spec_from_file_location("search_r1_retrieval_server", SERVER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", required=True)
    parser.add_argument("--corpus_path", required=True)
    parser.add_argument("--work_dir", required=True, help="where the extracted jsonl + offset index are cached")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = load_server_module()
    corpus = build(Path(args.corpus_path), Path(args.work_dir))
    server.load_corpus = lambda corpus_path: corpus

    config = server.Config(
        retrieval_method="bm25",
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retrieval_topk=args.topk,
    )
    server.retriever = server.get_retriever(config)
    server.config = config

    import uvicorn

    uvicorn.run(server.app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
