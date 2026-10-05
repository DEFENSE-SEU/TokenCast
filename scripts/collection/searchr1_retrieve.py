from __future__ import annotations

import argparse
import json
import sys
import urllib.request


def retrieve(query: str, topk: int, endpoint: str) -> list[dict]:


    payload = json.dumps({"queries": [query], "topk": topk, "return_scores": True}).encode("utf-8")
    request = urllib.request.Request(endpoint, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        body = json.loads(response.read().decode("utf-8"))
    return [scored["document"] for scored in body["result"][0]]


def format_information(documents: list[dict]) -> str:
    if not documents:
        return "<information>\nno results found\n</information>"
    blocks = []
    for index, document in enumerate(documents, start=1):
        text = document.get("contents") or document.get("text") or ""
        title = document.get("title")
        header = f"Doc {index}" + (f" ({title})" if title else "")
        blocks.append(f"{header}\n{text}".strip())
    return "<information>\n" + "\n\n".join(blocks) + "\n</information>"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    endpoint = f"http://{args.host}:{args.port}/retrieve"
    try:
        documents = retrieve(args.query, args.topk, endpoint)
    except Exception as error:
        print(f"search request failed: {error}", file=sys.stderr)
        sys.exit(1)
    print(format_information(documents))


if __name__ == "__main__":
    main()
