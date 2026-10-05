from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from bitsandbytes.nn import Linear4bit
from safetensors import safe_open
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    RepetitionPenaltyLogitsProcessor,
)

from qwen_probe import (
    GenerationProbeWriter,
    build_projection_matrix,
    project_hidden_state,
    projection_policy,
)


EXPECTED_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
EXPECTED_LINEAR4BIT_MODULES = 196


class IncrementalDecoder:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.token_cache: list[int] = []
        self.print_len = 0

    def put(self, token_id: int) -> str:
        self.token_cache.append(token_id)
        text = self.tokenizer.decode(self.token_cache, skip_special_tokens=True)
        if text.endswith("\n"):
            printable = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
            return printable
        if text and "\u4e00" <= text[-1] <= "\u9fff":
            printable = text[self.print_len :]
            self.print_len = len(text)
            return printable
        boundary = text.rfind(" ") + 1
        printable = text[self.print_len : boundary]
        self.print_len = boundary
        return printable

    def end(self) -> str:
        text = self.tokenizer.decode(self.token_cache, skip_special_tokens=True)
        printable = text[self.print_len :]
        self.token_cache = []
        self.print_len = 0
        return printable


def load(
    snapshot: Path,
) -> tuple[Any, Any, torch.Tensor, torch.Tensor, dict[str, Any], dict[str, Any]]:
    if snapshot.name != EXPECTED_REVISION:
        raise RuntimeError(f"unexpected local Qwen revision: {snapshot.name}")
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
    )
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModel.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        quantization_config=quantization,
        device_map={"": 0},
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    quantized = [module for module in model.modules() if isinstance(module, Linear4bit)]
    if not getattr(model, "is_quantized", False):
        raise RuntimeError("Qwen base model is missing its quantized runtime marker")
    if len(quantized) != EXPECTED_LINEAR4BIT_MODULES:
        raise RuntimeError(f"unexpected Qwen Linear4bit module count: {len(quantized)}")
    if any(module.weight.quant_state is None for module in quantized):
        raise RuntimeError("Qwen base model did not load with active NF4 quantization")
    quantization_states = [module.weight.quant_state for module in quantized]
    if any(state.quant_type != "nf4" for state in quantization_states):
        raise RuntimeError("Qwen base model quantization type drifted from NF4")
    if any(state.nested is not True for state in quantization_states):
        raise RuntimeError("Qwen base model NF4 double quantization is inactive")
    if any(module.compute_dtype != torch.bfloat16 for module in quantized):
        raise RuntimeError("Qwen base model compute dtype drifted from bfloat16")
    if any(module.weight.device.type != "cuda" for module in quantized):
        raise RuntimeError("Qwen base model quantized weights are not resident on CUDA")
    if model.config.tie_word_embeddings is not False:
        raise RuntimeError("Qwen snapshot unexpectedly ties its LM head")
    index_path = snapshot / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    generation_path = snapshot / "generation_config.json"
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    shard = snapshot / index["weight_map"]["lm_head.weight"]
    with safe_open(shard, framework="pt", device="cpu") as weights:
        lm_head = weights.get_tensor("lm_head.weight")
    expected_shape = (int(model.config.vocab_size), int(model.config.hidden_size))
    if tuple(lm_head.shape) != expected_shape:
        raise RuntimeError(f"unexpected Qwen LM head shape: {tuple(lm_head.shape)}")
    lm_head = lm_head.to(device=next(model.parameters()).device, dtype=torch.bfloat16)
    hidden_size = int(model.config.hidden_size)
    projection = build_projection_matrix(hidden_size, next(model.parameters()).device)
    metadata = {
        "revision": snapshot.name,
        "precision": "bitsandbytes-nf4-4bit",
        "device": str(next(model.parameters()).device),
        "model_class": type(model).__name__,
        "is_quantized": bool(model.is_quantized),
        "quant_type": "nf4",
        "double_quant": True,
        "compute_dtype": str(quantized[0].compute_dtype),
        "tie_word_embeddings": bool(model.config.tie_word_embeddings),
        "linear4bit_modules": len(quantized),
        "lm_head_dtype": str(lm_head.dtype),
        "repetition_penalty": float(generation["repetition_penalty"]),
        "eos_token_ids": list(generation["eos_token_id"]),
        "generation_probe_projection": projection_policy(hidden_size),
    }
    return tokenizer, model, lm_head, projection, metadata, generation


PREFILL_CHUNK_TOKENS = 2048


def prefill(model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor, chunk: int) -> Any:

    if chunk <= 0:
        raise ValueError("prefill chunk must be positive")
    current = None
    for start in range(0, int(input_ids.shape[1]), chunk):
        stop = min(start + chunk, int(input_ids.shape[1]))
        cache = {} if current is None else {"past_key_values": current.past_key_values}
        current = model(
            input_ids=input_ids[:, start:stop],
            attention_mask=attention_mask[:, :stop],
            use_cache=True,
            **cache,
        )
    if current is None:
        raise ValueError("prefill needs at least one prompt token")
    return current


def generate(
    tokenizer: Any,
    model: Any,
    lm_head: torch.Tensor,
    projection_matrix: torch.Tensor,
    generation: dict[str, Any],
    request: dict[str, Any],
    emit: Any,
    probe_writer: GenerationProbeWriter,
) -> dict[str, Any]:
    probe_writer.begin(request.get("generation_probe"))
    tools = request.get("tools") or None
    try:
        encoded = tokenizer.apply_chat_template(
            request["messages"],
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    except Exception as error:
        raise RuntimeError(f"QWEN_CHAT_TEMPLATE: {error}") from error
    input_tokens = int(encoded["attention_mask"].sum().item())
    emit({"type": "start", "id": request["id"], "input_tokens": input_tokens})
    max_tokens = int(request["max_tokens"])
    if input_tokens + max_tokens > int(model.config.max_position_embeddings):
        raise RuntimeError("CONTEXT_WINDOW_EXCEEDED: local Qwen request exceeds its context window")
    first_device = next(model.parameters()).device
    encoded = {name: tensor.to(first_device) for name, tensor in encoded.items()}
    temperature = float(request["temperature"])
    if temperature != 0:
        raise ValueError("local Qwen DSH worker supports deterministic temperature 0")
    stop = tuple(request.get("stop", []))
    if probe_writer.active and stop:
        raise ValueError("Generation Probe collection requires empty stop strings")
    eos_token_ids = {int(value) for value in generation["eos_token_id"]}
    repetition = RepetitionPenaltyLogitsProcessor(penalty=float(generation["repetition_penalty"]))
    generated_ids: list[int] = []
    matched_stop: str | None = None
    acknowledged_deltas = 0
    decoder = IncrementalDecoder(tokenizer)
    pending_text = ""
    reserve = max((len(value) for value in stop), default=0)
    prompt_ids = encoded["input_ids"]
    last_hidden_state: torch.Tensor | None = None
    last_hidden_state_source_output_tokens: int | None = None

    def deliver(
        value: str,
        hidden_state: torch.Tensor,
        hidden_state_source_output_tokens: int,
    ) -> None:
        nonlocal acknowledged_deltas
        acknowledgement = emit(
            {
                "type": "delta",
                "id": request["id"],
                "text": value,
                "output_tokens": len(generated_ids),
            },
            True,
        )
        if probe_writer.active and acknowledgement["generation_probe_checkpoints"]:
            probe_writer.record(
                acknowledgement,
                project_hidden_state(hidden_state, projection_matrix),
                source_output_tokens=hidden_state_source_output_tokens,
            )
        acknowledged_deltas += 1

    def publish(
        value: str,
        hidden_state: torch.Tensor,
        hidden_state_source_output_tokens: int,
    ) -> None:
        nonlocal pending_text
        pending_text += value
        boundary = len(pending_text) - reserve
        if boundary <= 0:
            return
        delta = pending_text[:boundary]
        pending_text = pending_text[boundary:]
        if delta:
            deliver(delta, hidden_state, hidden_state_source_output_tokens)

    with torch.inference_mode():
        current = prefill(model, encoded["input_ids"], encoded["attention_mask"], PREFILL_CHUNK_TOKENS)
        attention_mask = encoded["attention_mask"]
        for _ in range(max_tokens):
            logits = torch.nn.functional.linear(current.last_hidden_state[:, -1, :], lm_head)
            generated_tensor = torch.tensor(
                [generated_ids], device=prompt_ids.device, dtype=prompt_ids.dtype
            )
            sequence = torch.cat([prompt_ids, generated_tensor], dim=1)
            logits = repetition(sequence, logits)
            next_token = torch.argmax(logits, dim=-1)
            token_id = int(next_token.item())
            generated_ids.append(token_id)
            decoded_delta = decoder.put(token_id)
            decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
            matched_stop = next((value for value in stop if decoded.endswith(value)), None)
            if token_id in eos_token_ids:
                break
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(attention_mask[:, :1])], dim=1
            )
            current = model(
                input_ids=next_token[:, None],
                attention_mask=attention_mask,
                past_key_values=current.past_key_values,
                use_cache=True,
            )
            last_hidden_state = current.last_hidden_state[0, -1]
            last_hidden_state_source_output_tokens = len(generated_ids)
            publish(
                decoded_delta,
                last_hidden_state,
                last_hidden_state_source_output_tokens,
            )
            if matched_stop is not None:
                break
    decoded_tail = decoder.end()
    if decoded_tail:
        if last_hidden_state is None or last_hidden_state_source_output_tokens is None:
            raise RuntimeError("local Qwen decoded output without a token state")
        publish(
            decoded_tail,
            last_hidden_state,
            last_hidden_state_source_output_tokens,
        )
    output_tokens = len(generated_ids)
    output = tokenizer.decode(generated_ids, skip_special_tokens=True)
    if matched_stop is not None:
        output = output[: -len(matched_stop)]
        pending_text = pending_text[: -len(matched_stop)]
    if pending_text:
        if last_hidden_state is None or last_hidden_state_source_output_tokens is None:
            raise RuntimeError("local Qwen decoded output without a token state")
        deliver(
            pending_text,
            last_hidden_state,
            last_hidden_state_source_output_tokens,
        )
    acknowledgement = emit(
        {
            "type": "flush",
            "id": request["id"],
            "output_tokens": output_tokens,
        },
        True,
    )
    if probe_writer.active and acknowledgement["generation_probe_checkpoints"]:
        if last_hidden_state is None or last_hidden_state_source_output_tokens is None:
            raise RuntimeError("local Qwen checkpoint has no token state")
        probe_writer.record(
            acknowledgement,
            project_hidden_state(last_hidden_state, projection_matrix),
            source_output_tokens=last_hidden_state_source_output_tokens,
        )
    hit_limit = (
        output_tokens >= max_tokens
        and (output_tokens == 0 or generated_ids[-1] not in eos_token_ids)
        and matched_stop is None
    )
    return {
        "type": "result",
        "id": request["id"],
        "output": output,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "acknowledged_deltas": acknowledged_deltas,
        "finish_reason": "max-tokens" if hit_limit else "stop",
    }


def main() -> int:
    sys.stdin.reconfigure(encoding="utf-8", errors="strict")
    sys.stdout.reconfigure(encoding="utf-8", errors="strict")
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    snapshot = args.snapshot.resolve()
    tokenizer, model, lm_head, projection_matrix, metadata, generation = load(snapshot)
    probe_writer = GenerationProbeWriter(
        model_revision=metadata["revision"],
        projection=metadata["generation_probe_projection"],
    )
    print(
        json.dumps({"type": "ready", **metadata}),
        flush=True,
    )
    try:
        for line in sys.stdin:
            request: dict[str, Any] = {}

            def emit(
                message: dict[str, Any],
                require_ack: bool = False,
            ) -> dict[str, Any] | None:
                print(json.dumps(message, ensure_ascii=False), flush=True)
                if not require_ack:
                    return None
                ack_line = sys.stdin.readline()
                if not ack_line:
                    raise RuntimeError("local Qwen stream acknowledgement is missing")
                ack = json.loads(ack_line)
                if (
                    ack.get("type") != "ack"
                    or ack.get("id") != message["id"]
                    or ack.get("output_tokens") != message["output_tokens"]
                ):
                    raise RuntimeError("local Qwen stream acknowledgement drifted")
                checkpoints = ack.get("generation_probe_checkpoints", [])
                if not isinstance(checkpoints, list):
                    raise RuntimeError("local Qwen Generation Probe acknowledgement drifted")
                ack["generation_probe_checkpoints"] = checkpoints
                return ack

            try:
                request = json.loads(line)
                response = generate(
                    tokenizer,
                    model,
                    lm_head,
                    projection_matrix,
                    generation,
                    request,
                    emit,
                    probe_writer,
                )
            except Exception as error:
                response = {
                    "type": "error",
                    "id": request.get("id"),
                    "message": str(error),
                    "code": (
                        "CONTEXT_WINDOW_EXCEEDED"
                        if str(error).startswith("CONTEXT_WINDOW_EXCEEDED:")
                        else "LOCAL_QWEN_INFERENCE"
                    ),
                }
            finally:
                probe_writer.close()
            print(json.dumps(response, ensure_ascii=False), flush=True)
    finally:
        probe_writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
