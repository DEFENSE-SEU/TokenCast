from __future__ import annotations

import json
from pathlib import Path
import time
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class BoundedGateway:
    def __init__(self, *, base_url: str, api_key: str, ledger_path: Path,
                 models: tuple[str, ...], max_calls: int | None = 8,
                 max_accounted_tokens: int | None = 40000, max_prompt_bytes: int = 24000,
                 max_output_tokens: int = 768, direct_connection: bool = False):
        if not base_url.startswith("https://"):
            raise ValueError("gateway URL must use HTTPS")
        for name, value in (("max_calls", max_calls), ("max_accounted_tokens", max_accounted_tokens)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"{name} must be a positive integer or None")
        for name, value in (("max_prompt_bytes", max_prompt_bytes), ("max_output_tokens", max_output_tokens)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.key, self.path, self.models = api_key, ledger_path, models
        self.direct_connection = direct_connection
        self.policy = dict(max_calls=max_calls, max_accounted_tokens=max_accounted_tokens,
                           max_prompt_bytes=max_prompt_bytes, max_output_tokens=max_output_tokens,
                           models=list(models), automatic_retries=0)
        if direct_connection:
            self.policy["connection_mode"] = "direct"
        if ledger_path.exists():
            self.ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            if self.ledger["policy"] != self.policy:
                raise ValueError("cannot change the budget of an existing request ledger")
        else:
            self.ledger = {"policy": self.policy, "requests": []}

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.ledger, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def retain_uncertain_reservation(self, request_id: str, *, review_note: str):

        if not review_note.strip():
            raise ValueError("a review note is required")
        row = next(row for row in self.ledger["requests"] if row["request_id"] == request_id)
        if row["status"] != "unsettled":
            raise ValueError("only unsettled requests need reservation review")
        row.update(status="uncertain_reserved", review_note=review_note,
                   accounted_tokens=row["reserved_tokens"])
        self._save()

    def complete(self, *, model: str, prompt: str, request_id: str) -> dict:
        if model not in self.models:
            raise ValueError("model was not preselected")
        previous = self.ledger["requests"]
        if any(row["request_id"] == request_id for row in previous):
            raise ValueError("request ID already used; implicit retries are prohibited")
        if any(row["status"] not in {"complete", "uncertain_reserved"} for row in previous):
            raise ValueError("unsettled request; inspect accounting before continuing")
        if self.policy["max_calls"] is not None and len(previous) >= self.policy["max_calls"]:
            raise ValueError("request budget exhausted")
        prompt_bytes = len(prompt.encode("utf-8"))
        if prompt_bytes > self.policy["max_prompt_bytes"]:
            raise ValueError("prompt exceeds the preselected size limit")


        reserved = prompt_bytes + 1024 + self.policy["max_output_tokens"]
        consumed = sum(row["accounted_tokens"] for row in previous)
        if (self.policy["max_accounted_tokens"] is not None
                and consumed + reserved > self.policy["max_accounted_tokens"]):
            raise ValueError("insufficient remaining local token budget")
        row = dict(request_id=request_id, model=model, status="pending",
                   reserved_tokens=reserved, accounted_tokens=reserved,
                   monetary_charge_cny=None)
        previous.append(row)
        self._save()
        body = {"model": model, "messages": [{"role": "user", "content": prompt}],
                "temperature": 0, "max_tokens": self.policy["max_output_tokens"],
                "stream": False}
        if "deepseek" in model.lower():
            body["thinking"] = {"type": "disabled"}
        request = Request(self.url, data=json.dumps(body).encode("utf-8"), headers={
            "Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
        started = time.perf_counter()
        try:
            handlers = [_NoRedirect()]
            if self.direct_connection:
                handlers.append(ProxyHandler({}))
            with build_opener(*handlers).open(request, timeout=90) as response:
                payload = json.loads(response.read())
            usage = payload.get("usage", {})
            inputs, outputs = usage.get("prompt_tokens"), usage.get("completion_tokens")
            if any(isinstance(x, bool) or not isinstance(x, int) or x < 0
                   for x in (inputs, outputs)):
                raise ValueError("gateway response lacks exact input/output accounting")
            total = inputs + outputs
            if usage.get("total_tokens", total) != total:
                raise ValueError("gateway total disagrees with input plus output")
            choice = payload["choices"][0]
            row.update(status="complete", accounted_tokens=total,
                       usage={"input_tokens": inputs, "output_tokens": outputs,
                              "total_tokens": total}, raw_usage=usage,
                       finish_reason=choice.get("finish_reason"),
                       returned_model=payload.get("model"),
                       content=choice["message"].get("content"))
            if total > reserved:
                row["reservation_exceeded"] = True
                if self.policy["max_accounted_tokens"] is not None:
                    row["status"] = "reservation_exceeded"
        except Exception as exc:
            row.update(status="unsettled", error_type=type(exc).__name__)
            raise
        finally:
            row["wall_seconds"] = time.perf_counter() - started
            self.ledger = json.loads(json.dumps(self.ledger).replace(self.key, "[REDACTED]"))
            self._save()
        return self.ledger["requests"][-1]
