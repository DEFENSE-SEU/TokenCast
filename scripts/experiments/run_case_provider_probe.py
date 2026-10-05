from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from token_prediction.collection.dsh.env_file import parse_environment_file
from token_prediction.collection.dsh.provider_environment import resolve_hosted_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    models = ["claude-4.5-haiku", "deepseek/deepseek-v4-flash"]
    policy = dict(models=models, requests_per_model=1, max_output_tokens=128,
                  automatic_retries=0, max_user_message_bytes=128,
                  monetary_charge_cny=None, purpose="availability_only")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    if not args.execute:
        print(json.dumps(policy))
        return
    values = resolve_hosted_environment({}, file_environment=parse_environment_file(
        args.env_file.read_text(encoding="utf-8"))).environment
    key = values[values["DSH_API_KEY_ENV"]]
    base = values["DSH_BASE_URL"].rstrip("/")
    reports = []
    for model in models:
        body = {"model": model, "messages": [{"role": "user", "content":
                'Reply with exactly this JSON: {"ok":true}'}], "max_tokens": 128,
                "temperature": 0, "stream": False}
        if "deepseek" in model:
            body["thinking"] = {"type": "disabled"}
        request = Request(base + "/chat/completions", data=json.dumps(body).encode(),
                          headers={"Authorization": "Bearer " + key,
                                   "Content-Type": "application/json"})
        started = time.perf_counter()
        report = {"requested_model": model, "max_output_tokens": 128}
        try:
            with urlopen(request, timeout=60) as response:
                value = json.loads(response.read())
            choice = value.get("choices", [{}])[0]
            report.update(returned_model=value.get("model"), usage=value.get("usage"),
                          finish_reason=choice.get("finish_reason"),
                          content=choice.get("message", {}).get("content"), status="received")
        except HTTPError as exc:
            report.update(status="http_error", http_status=exc.code,
                          detail=exc.read(2000).decode("utf-8", errors="replace"))
        except (URLError, TimeoutError, ValueError) as exc:
            report.update(status="request_error", detail=str(exc))
        report["wall_seconds"] = time.perf_counter() - started
        report["monetary_charge_cny"] = None

        report = json.loads(json.dumps(report).replace(key, "[REDACTED]"))
        reports.append(report)
        (args.output / "results.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
