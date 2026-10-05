from __future__ import annotations

import re


def compatible_model_family(model: str) -> str | None:
    if re.match(r"^(?:google/)?gemini-", model, re.I):
        return "gemini"
    if re.match(r"^(?:moonshotai/)?kimi-", model, re.I):
        return "kimi"
    if re.match(r"^(?:deepseek/)?deepseek-v4-", model, re.I):
        return "deepseek"
    if re.match(r"^(?:anthropic/)?claude-", model, re.I):
        return "claude"
    return None


def validate_compatible_reasoning(model: str, effort: str) -> None:
    family = compatible_model_family(model)
    if family == "gemini":
        can_disable = re.fullmatch(r"(?:google/)?gemini-2\.5-flash(?:-lite)?", model, re.I)
        if not (can_disable and effort == "off") and effort not in {
            "minimal", "low", "medium", "high",
        }:
            raise ValueError(
                f"Gemini model {model} requires a supported reasoning effort; "
                "use low, medium, or high (off is limited to Gemini 2.5 Flash/Flash-Lite)."
            )
    if family == "kimi" and (
        not re.fullmatch(r"(?:moonshotai/)?kimi-k2\.5", model, re.I)
        or effort not in {"off", "high"}
    ):
        raise ValueError(
            "Kimi compatibility supports kimi-k2.5 with off (instant) "
            "or high (thinking enabled)."
        )
    if family == "deepseek" and effort not in {"off", "high"}:
        raise ValueError("DeepSeek V4 compatibility supports off or high reasoning.")
    if family == "claude" and effort != "off":
        raise ValueError("Claude Chat Completions compatibility currently supports off reasoning only.")
