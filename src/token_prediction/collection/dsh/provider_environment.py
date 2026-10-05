from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .env_file import EnvironmentFileError


_PROFILES = {
    "openai": ("qwen-compatible", "OPENAI"),
    "qwen": ("qwen-compatible", "QWEN"),
    "gemini": ("qwen-compatible", "GEMINI"),
    "kimi": ("qwen-compatible", "KIMI"),
    "claude": ("qwen-compatible", "CLAUDE"),
    "deepseek": ("deepseek-official", "DEEPSEEK"),
    "openai-codex": ("openai-codex", "OPENAI_CODEX"),
}
_LEGACY = {"qwen-compatible", "deepseek-official"}


@dataclass(frozen=True)
class HostedEnvironment:
    provider: str
    model_provider: str
    environment: dict[str, str]
    variable_names: dict[str, str]


def resolve_hosted_environment(
    environment: Mapping[str, str],
    *,
    file_environment: Mapping[str, str] | None = None,
    provider: str | None = None,
) -> HostedEnvironment:

    file_environment = file_environment or {}
    layers = (environment, file_environment)

    def pick(public: str, legacy: str) -> tuple[str | None, str]:
        for layer in layers:
            for name in (public, legacy):
                if name in layer:
                    return layer[name], name
        return None, legacy

    selected, _ = pick("MODEL_PROVIDER", "DSH_PROVIDER")
    selected = (selected if selected is not None else "deepseek-official").strip()
    requested = provider.strip() if provider is not None else selected
    route = _PROFILES.get(requested, (requested, ""))[0]
    if not requested:
        raise EnvironmentFileError("hosted provider must be non-empty")


    profile = requested
    if requested in _LEGACY and selected in _PROFILES and _PROFILES[selected][0] == route:
        profile = selected
    prefix = _PROFILES.get(profile, (route, ""))[1]
    profile_layer = next(
        (index for index, layer in enumerate(layers)
         if "MODEL_PROVIDER" in layer or "DSH_PROVIDER" in layer),
        len(layers),
    )
    if provider is not None and requested in _PROFILES and requested != selected:
        profile_layer = -1

    def connection_pick(public: str, legacy: str) -> tuple[str | None, str]:
        for index, layer in enumerate(layers):
            if public in layer:
                return layer[public], public


            same_profile = layer.get("MODEL_PROVIDER", "").strip() == profile
            if legacy in layer and (index <= profile_layer or same_profile):
                return layer[legacy], legacy
        return None, public

    merged = {**file_environment, **environment, "DSH_PROVIDER": route}
    names: dict[str, str] = {}
    aliases = {
        "DSH_CONTEXT_WINDOW": "MODEL_CONTEXT_WINDOW",
        "DSH_MAX_TOKENS_CEILING": "MODEL_MAX_OUTPUT_TOKENS",
        "DSH_OPENAI_AUTH_FILE": "OPENAI_CODEX_AUTH_FILE",
    }
    if prefix and route != "openai-codex":
        aliases["DSH_BASE_URL"] = f"{prefix}_BASE_URL"
    for legacy, public in aliases.items():
        value, source = (connection_pick if legacy == "DSH_BASE_URL" else pick)(public, legacy)
        if value is not None:
            merged[legacy] = value
        elif legacy == "DSH_BASE_URL":
            merged.pop(legacy, None)
        names[legacy] = source if value is not None else (public if prefix else legacy)
    if prefix and route != "openai-codex":
        key_name = f"{prefix}_API_KEY"


        key_value, source = connection_pick(key_name, "DSH_API_KEY_ENV")
        selected_key = key_value if source == "DSH_API_KEY_ENV" else key_name
        merged["DSH_API_KEY_ENV"] = selected_key
    return HostedEnvironment(route, profile, merged, names)


def public_configuration_report(report: dict, resolved: HostedEnvironment) -> dict:

    report = dict(report)
    report["model_provider"] = resolved.model_provider
    report["variables"] = {
        resolved.variable_names.get(name, name): state
        for name, state in report["variables"].items()
        if name != "DSH_API_KEY_ENV" or resolved.model_provider in _LEGACY
    }
    errors = []
    for error in report["errors"]:
        for legacy, public in resolved.variable_names.items():
            error = error.replace(legacy, public)
        errors.append(error)
    report["errors"] = errors
    return report
