import argparse
import asyncio
import ctypes
import json
import time

DECODER = json.JSONDecoder()
SAMPLING_FIELDS = ("do_sample", "temperature", "top_p", "top_k", "min_p", "repetition_penalty", "max_new_tokens")

COMMIT_GB_PER_TOKEN = 0.22 / 1024
COMMIT_MARGIN_GB = 1.0


class _MemoryStatus(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
        (name, ctypes.c_ulonglong)
        for name in ("total_phys", "avail_phys", "total_commit", "avail_commit", "total_virtual", "avail_virtual", "avail_extended")
    ]


def free_commit_gb() -> float:
    status = _MemoryStatus(length=ctypes.sizeof(_MemoryStatus))
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return status.avail_commit / 2**30


def parse_json_tool_calls(text: str) -> tuple[str | None, list[dict]]:

    calls, pieces, kept_from = [], [], 0
    start = text.find("{")
    while start != -1:
        try:
            value, end = DECODER.raw_decode(text, start)
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        arguments = value.get("parameters", value.get("arguments")) if isinstance(value, dict) else None
        if isinstance(value, dict) and isinstance(value.get("name"), str) and arguments is not None:
            pieces.append(text[kept_from:start])
            kept_from = end
            calls.append({"name": value["name"], "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments)})
        start = text.find("{", end)
    pieces.append(text[kept_from:])
    content = "".join(pieces).strip().strip(";").strip()
    return content or None, calls


def single_call_turns(messages: list[dict]) -> list[dict]:

    rewritten = []
    for message in messages:
        calls = message.get("tool_calls")
        if "tool_calls" in message and not calls:
            message = {key: value for key, value in message.items() if key != "tool_calls"}
        elif calls and len(calls) > 1:
            text = "; ".join(
                json.dumps({"name": c["function"]["name"], "parameters": json.loads(c["function"]["arguments"])}, ensure_ascii=False)
                for c in calls
            )
            message = {"role": message["role"], "content": text}
        rewritten.append(message)
    return rewritten


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-context", type=int, required=True)
    args = parser.parse_args()

    import torch
    from fastapi import HTTPException
    from fastapi.responses import JSONResponse
    from openai.types.chat import ChatCompletionMessageToolCall
    from openai.types.completion_usage import CompletionUsage
    from transformers.cli.serve import Serve
    from transformers.cli.serving.chat_completion import ChatCompletionHandler
    from transformers.cli.serving.utils import BaseHandler
    from transformers.integrations import sdpa_attention

    sdpa_attention.use_gqa_in_sdpa = lambda *args, **kwargs: False
    to_processor_inputs = BaseHandler.get_processor_inputs_from_messages

    def get_processor_inputs_from_messages(messages, modality):
        return to_processor_inputs(single_call_turns(messages), modality)

    async def non_streaming(self, request_id, model, processor, model_id, inputs, gen_config, gen_manager,
                            tool_config=None, reasoning_config=None):
        prompt_tokens = inputs["input_ids"].shape[-1]
        requested = prompt_tokens + gen_config.max_new_tokens
        if requested > args.max_context:
            print(json.dumps({"rejected": "context_length_exceeded", "prompt_tokens": prompt_tokens}), flush=True)
            raise HTTPException(status_code=400, detail=(
                f"This model's maximum context length is {args.max_context} tokens. However, you requested "
                f"{requested} tokens ({prompt_tokens} in the messages, {gen_config.max_new_tokens} in the completion)."
            ))

        waited = time.time()
        while free_commit_gb() < prompt_tokens * COMMIT_GB_PER_TOKEN + COMMIT_MARGIN_GB:
            await asyncio.sleep(5)
        started = time.time()
        try:
            text, _, generated_ids = await gen_manager.generate_non_streaming(model, processor, inputs, gen_config, request_id=request_id)
        finally:

            torch.cuda.empty_cache()
        content, calls = parse_json_tool_calls(text)
        completion_tokens = len(generated_ids)
        if calls:
            finish_reason = "tool_calls"
        else:
            finish_reason = "length" if completion_tokens >= gen_config.max_new_tokens else "stop"
        print(json.dumps({
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "tool_calls": len(calls),
            "finish_reason": finish_reason, "seconds": round(time.time() - started, 2),
            "commit_wait_seconds": round(started - waited, 1),
            "sampling": {field: getattr(gen_config, field, None) for field in SAMPLING_FIELDS},
        }), flush=True)
        tool_calls = [
            ChatCompletionMessageToolCall(id=f"{request_id}_tool_call_{i}", type="function", function=call)
            for i, call in enumerate(calls)
        ]
        usage = CompletionUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                                total_tokens=prompt_tokens + completion_tokens)
        return JSONResponse(
            self._build_completion(request_id, content, model_id, finish_reason=finish_reason, usage=usage,
                                   tool_calls=tool_calls or None),
            media_type="application/json",
        )

    BaseHandler.get_processor_inputs_from_messages = staticmethod(get_processor_inputs_from_messages)
    ChatCompletionHandler._non_streaming = non_streaming
    Serve(force_model=args.model, device="cuda:0", dtype="bfloat16", host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
