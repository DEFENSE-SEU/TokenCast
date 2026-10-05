from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from .manifest import DshManifest
from .compatible_models import validate_compatible_reasoning


def check_hosted_configuration(
    manifest: DshManifest,
    provider: str,
    environment: Mapping[str, str],
) -> dict[str, Any]:


    errors: list[str] = []
    variables: dict[str, dict[str, bool]] = {}
    supported = {"qwen-compatible", "deepseek-official", "openai-codex"}
    runs = tuple(
        run for run in manifest.scheduled_runs()
        if run.condition.configured_provider == provider
        and run.condition.access_mode == "hosted_api_blackbox"
    )
    counts = Counter(run.condition.condition_id for run in runs)
    conditions = [
        {
            "condition_id": condition.condition_id,
            "model": condition.configured_model,
            "reasoning_effort": condition.reasoning_effort,
            "planned_run_count": counts[condition.condition_id],
        }
        for condition in manifest.model_conditions
        if condition.condition_id in counts
    ]
    required_output = max(
        (int(run.cap_profile["max_attempt_output_tokens"]) for run in runs), default=0
    )
    if provider == "qwen-compatible":
        for condition in conditions:
            try:
                validate_compatible_reasoning(condition["model"], condition["reasoning_effort"])
            except ValueError as error:
                errors.append(str(error))

    def configured(name: str) -> bool:
        present = bool(environment.get(name, "").strip())
        variables[name] = {"configured": present}
        return present

    def positive_integer(name: str, default: str | None = None) -> int | None:
        configured(name)
        value = environment.get(name) or default
        if value is None or re.fullmatch(r"[0-9]+", value.strip()) is None:
            errors.append(f"{name} must be a positive integer.")
            return None
        parsed = int(value)
        if parsed <= 0:
            errors.append(f"{name} must be a positive integer.")
            return None
        return parsed

    if not runs:
        errors.append("The manifest has no hosted rows for the selected provider.")
    credential_check = "environment_variable_presence_only"
    credential_variable: str | None = None
    if provider not in supported:
        errors.append("Select a supported hosted provider: qwen-compatible, deepseek-official, openai-codex.")
    elif provider == "openai-codex":
        credential_check = "deferred_to_subscription_preflight"
    else:
        default_key = "QWEN_API_KEY" if provider == "qwen-compatible" else "DEEPSEEK_API_KEY"
        configured("DSH_API_KEY_ENV")
        key_name = environment.get("DSH_API_KEY_ENV") or default_key
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key_name) is None:
            errors.append("DSH_API_KEY_ENV must name an environment variable.")
        else:
            credential_variable = key_name
            if not configured(key_name):
                errors.append(f"{key_name} must contain a non-empty credential.")

        configured("DSH_BASE_URL")
        endpoint = environment.get("DSH_BASE_URL")
        if provider == "deepseek-official":
            endpoint = endpoint or "https://api.deepseek.com"
        try:
            parsed_url = urlsplit(endpoint or "")
            valid_endpoint = parsed_url.scheme in {"http", "https"} and bool(parsed_url.hostname)
        except ValueError:
            valid_endpoint = False
        if not valid_endpoint:
            errors.append("DSH_BASE_URL must be an absolute HTTP(S) endpoint.")

        deepseek = provider == "deepseek-official"
        positive_integer("DSH_CONTEXT_WINDOW", "200000" if deepseek else None)
        output_ceiling = positive_integer(
            "DSH_MAX_TOKENS_CEILING", "8192" if deepseek else None
        )
        if output_ceiling is not None and output_ceiling < required_output:
            errors.append(
                "DSH_MAX_TOKENS_CEILING is below the selected manifest's "
                f"max_attempt_output_tokens ({required_output})."
            )

    return {
        "status": "configuration_invalid" if errors else "configuration_valid",
        "manifest_id": manifest.manifest_id,
        "provider": provider if provider in supported else None,
        "available_hosted_providers": sorted({
            condition.configured_provider for condition in manifest.model_conditions
            if condition.access_mode == "hosted_api_blackbox"
        }),
        "model_source": "manifest.model_conditions[].configured_model",
        "model_override_note": "The collector overrides DSH_MODEL with each selected manifest model.",
        "dsh_model_configured": bool(environment.get("DSH_MODEL", "").strip()),
        "conditions": conditions,
        "planned_run_count": len(runs),
        "required_max_attempt_output_tokens": required_output,
        "variables": variables,
        "credential_check": credential_check,
        "credential_variable": credential_variable,
        "provider_access": "unverified",
        "network_requests": 0,
        "notes": [
            "This checks configuration only; installed DSH, task packs, and backend access remain unchecked.",
            *([
                "Run the subscription preflight to check the Codex auth file and installed model catalog. "
                "This command does not read OAuth credentials."
            ] if provider == "openai-codex" else []),
        ],
        "errors": errors,
    }
