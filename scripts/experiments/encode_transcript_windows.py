from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HF_HOME = Path("workspace/models")


def read_session(capture: Path) -> list[dict]:
    import zstandard as zstd

    raw = zstd.ZstdDecompressor().stream_reader((capture / "session.jsonl.zstd").open("rb")).read()
    return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]


def transcript_steps(records: list[dict]) -> tuple[str, dict[int, str]]:

    task_text = ""
    steps: dict[int, list[str]] = defaultdict(list)
    for record in records:
        kind, data = record.get("type"), record.get("data") or {}
        if kind == "user/message" and not task_text:
            task_text = "".join(part.get("text", "") for part in data.get("content", []) if part.get("type") == "text")
        elif kind == "assistant/message":
            step = int(data.get("step", 0))
            parts = []
            for part in (data.get("message") or {}).get("content", []):
                if part.get("type") == "text":
                    parts.append(part["text"])
                elif part.get("type") == "reasoning":
                    parts.append(part.get("text", ""))
                elif part.get("type") == "tool-call":
                    parts.append(f"<tool_call name=\"{part.get('name')}\">{part.get('arguments', '')}</tool_call>")
            steps[step].append("ASSISTANT:\n" + "\n".join(parts))
        elif kind == "tool/result":
            step = int(data.get("step", 0))
            texts = []
            for part in (data.get("message") or {}).get("content", []):
                for inner in part.get("content", []) if isinstance(part.get("content"), list) else []:
                    if inner.get("type") == "text":
                        texts.append(inner["text"])
            steps[step].append("TOOL RESULT:\n" + "\n".join(texts))
    return task_text, {step: "\n".join(parts) for step, parts in steps.items()}


def call_step(record: dict) -> int | None:

    context = record.get("prediction_context_id") or ""
    if ":call:" in context:
        suffix = context.rsplit(":call:", 1)[1]
        if suffix.isdigit():
            return int(suffix) + 1
    call_id = record.get("logical_call_id") or ""
    if ":step-" in call_id:
        return int(call_id.rsplit(":step-", 1)[1].split(":")[0])
    return None


def load_points(fit_dir: Path) -> list[dict]:
    seen, points = set(), []
    with (fit_dir / "predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["point_id"] in seen:
                continue
            seen.add(record["point_id"])
            points.append({k: record.get(k) for k in ("point_id", "experiment_id", "run_id", "task_id", "prediction_context_id", "logical_call_id", "source_event_id", "cutoff_event_seq", "target_value", "fold")})
    return points


def load_prefixes(path: Path, run_ids: set[str]) -> dict[str, str]:
    out = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            if event["event_type"] != "generation_checkpoint":
                continue
            if event["event_id"].split(":")[1] not in run_ids:
                continue
            out[event["event_id"]] = event["payload"]["visible_text_prefix"]
    return out


def build_windows(points, captures: dict[str, Path], prefixes: dict[str, str]) -> list[dict]:
    cache: dict[str, tuple[str, dict[int, str]]] = {}
    windows = []
    for point in points:
        run_id = point["run_id"]
        if run_id not in cache:
            cache[run_id] = transcript_steps(read_session(captures[run_id]))
        task_text, steps = cache[run_id]
        prefix = point["experiment_id"].split("--")[0]
        step = call_step(point)


        if prefix == "task_initialization":
            head, tail = task_text, ""
        elif prefix in ("call_start", "task_update"):
            head, tail = task_text, "\n\n".join(steps[s] for s in sorted(steps) if s < step)
        else:
            history = "\n\n".join(steps[s] for s in sorted(steps) if s < step)
            head, tail = task_text, history + "\n\nASSISTANT (in progress):\n" + prefixes.get(point["source_event_id"], "")
        windows.append({**point, "position": prefix, "head": head, "tail": tail})
    return windows


def _stream_safetensors(torch, path: Path, targets: dict, chunk_bytes: int = 64 << 20) -> set[str]:

    import json as _json
    import struct

    dtypes = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
    filled = set()
    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = _json.loads(handle.read(header_len))
        base = 8 + header_len
        for name, meta in header.items():
            if name == "__metadata__" or name not in targets:
                continue
            start, end = meta["data_offsets"]
            dtype = dtypes[meta["dtype"]]
            destination = targets[name].data
            if list(destination.shape) != list(meta["shape"]):
                raise ValueError(f"{name}: checkpoint shape {meta['shape']} differs from model shape {tuple(destination.shape)}")
            handle.seek(base + start)
            if destination.dtype != dtype:
                raw = bytearray(handle.read(end - start))
                destination.copy_(torch.frombuffer(raw, dtype=dtype).reshape(meta["shape"]).to(destination.dtype))
                filled.add(name)
                continue
            flat = destination.view(torch.uint8).view(-1)
            offset = 0
            while offset < end - start:
                chunk = handle.read(min(chunk_bytes, end - start - offset))
                flat[offset : offset + len(chunk)].copy_(torch.frombuffer(bytearray(chunk), dtype=torch.uint8))
                offset += len(chunk)
            filled.add(name)
    return filled


def proxy_states(model, input_ids, attention_mask, chunk: int = 256, layers: tuple[int, ...] = ()):

    import torch

    out = model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, output_hidden_states=bool(layers))
    hidden = out.last_hidden_state
    intermediate = {index: out.hidden_states[index].float() for index in layers} if layers else {}
    entropy = torch.empty(hidden.shape[:2], dtype=torch.float32, device=hidden.device)
    for start in range(0, hidden.shape[1], chunk):
        logits = model.lm_head(hidden[:, start : start + chunk, :]).float()
        log_probs = torch.log_softmax(logits, dim=-1)
        entropy[:, start : start + chunk] = -(log_probs.exp() * log_probs).sum(-1)
        del logits, log_probs
    return hidden.float(), entropy, intermediate


def load_causal_lm(name: str, device: str):

    import torch
    from huggingface_hub.constants import HF_HUB_CACHE
    from transformers import AutoConfig, AutoModelForCausalLM


    snapshots = sorted((Path(HF_HUB_CACHE) / f"models--{name.replace('/', '--')}" / "snapshots").iterdir())
    if not snapshots:
        raise RuntimeError(f"no cached snapshot for {name}")
    snapshot = snapshots[-1]
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(snapshot), dtype=torch.bfloat16)
    model = model.to_empty(device=device)
    targets = dict(model.named_parameters())
    targets.update(dict(model.named_buffers()))
    filled: set[str] = set()
    for shard in sorted(snapshot.glob("*.safetensors")):
        filled |= _stream_safetensors(torch, shard, targets)
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    missing = {n for n in dict(model.named_parameters()) if n not in filled and not n.startswith("lm_head.")}
    if missing:
        raise RuntimeError(f"{len(missing)} parameters not in the checkpoint, e.g. {sorted(missing)[:3]}")
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("grid", type=Path)
    parser.add_argument("--runs-root", type=Path, action="append", required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proxy", default="unsloth/Llama-3.2-3B-Instruct")
    parser.add_argument("--encoder", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--head-tokens", type=int, default=384)
    parser.add_argument("--tail-tokens", type=int, default=1536)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--layers", default="", help="comma list of hidden-state indices to keep beside the final layer")
    args = parser.parse_args()

    from scripts.experiments.run_four_position_baseline_grid import capture_dirs, capture_run_id

    captures = {}
    for root in args.runs_root:
        for capture in capture_dirs(root):
            captures[capture_run_id(capture)] = capture
    fit_dir = sorted(args.grid.glob("split-*/fit-*"))[0]
    points = [p for p in load_points(fit_dir) if p["run_id"] in captures]
    if args.limit:
        kept: dict[str, list] = defaultdict(list)
        for p in points:
            kept[p["experiment_id"]].append(p)
        points = [p for group in kept.values() for p in group[: args.limit]]
    prefixes = load_prefixes(args.events, {p["run_id"] for p in points})
    windows = build_windows(points, captures, prefixes)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "windows.jsonl").open("w", encoding="utf-8") as handle:
        for w in windows:
            handle.write(json.dumps({k: v for k, v in w.items() if k not in ("head", "tail")} | {"head_chars": len(w["head"]), "tail_chars": len(w["tail"])}) + "\n")
    print(f"{len(windows)} windows from {len({w['run_id'] for w in windows})} runs", flush=True)

    import os
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    import torch
    from transformers import AutoModel, AutoTokenizer

    layer_indices = tuple(int(x) for x in args.layers.split(",") if x.strip())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.proxy)
    model = load_causal_lm(args.proxy, device)
    enc_tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    encoder = AutoModel.from_pretrained(args.encoder).to(device).eval()

    def window_ids(head: str, tail: str) -> list[int]:
        head_ids = tokenizer(head, add_special_tokens=False)["input_ids"][: args.head_tokens]
        tail_ids = tokenizer(tail, add_special_tokens=False)["input_ids"]
        tail_ids = tail_ids[-args.tail_tokens :] if tail_ids else []
        return [tokenizer.bos_token_id] + head_ids + tail_ids if tokenizer.bos_token_id is not None else head_ids + tail_ids


    part_dir = args.output / "parts"
    part_dir.mkdir(exist_ok=True)
    done_points: set[str] = set()
    for part in sorted(part_dir.glob("part-*.npz")):
        done_points.update(np.load(part, allow_pickle=False)["point_id"].tolist())
    if done_points:
        print(f"resuming: {len(done_points)} windows already encoded", flush=True)
    timing_path = args.output / "timing.jsonl"


    by_text: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for w in windows:
        by_text[(w["head"], w["tail"])].append(w)
    pending = []
    for group in by_text.values():
        if all(w["point_id"] in done_points for w in group):
            continue
        pending.append({**group[0], "shared_points": [w["point_id"] for w in group], "shared_positions": [w["position"] for w in group]})
    print(f"{len(pending)} forward passes for {sum(len(w['shared_points']) for w in pending)} windows", flush=True)
    reps = defaultdict(list)
    timing = []

    def flush_part() -> None:
        if not reps["point_id"]:
            return
        index = len(list(part_dir.glob("part-*.npz")))
        np.savez_compressed(part_dir / f"part-{index:04d}.npz", **{k: np.asarray(v) for k, v in reps.items()})
        with timing_path.open("a", encoding="utf-8") as handle:
            handle.write("".join(json.dumps(t) + chr(10) for t in timing))
        reps.clear()
        timing.clear()

    with torch.inference_mode():
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            ids = [window_ids(w["head"], w["tail"]) for w in batch]
            longest = max(len(x) for x in ids)
            pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            input_ids = torch.full((len(ids), longest), pad, dtype=torch.long)
            attention = torch.zeros((len(ids), longest), dtype=torch.long)
            for i, x in enumerate(ids):
                input_ids[i, longest - len(x) :] = torch.tensor(x)
                attention[i, longest - len(x) :] = 1
            t0 = time.perf_counter()
            hidden, entropy, intermediate = proxy_states(model, input_ids.to(device), attention.to(device), layers=layer_indices)
            mask = attention.to(device).float()
            last = hidden[:, -1, :]
            mean = (hidden * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
            weights = torch.softmax(torch.where(mask > 0, entropy, torch.full_like(entropy, -1e9)), dim=1)
            egtp = (hidden * weights[..., None]).sum(1)
            torch.cuda.synchronize() if device == "cuda" else None
            proxy_ms = (time.perf_counter() - t0) * 1000 / len(batch)
            texts = [(w["head"] + "\n\n" + w["tail"])[-4000:] for w in batch]
            t1 = time.perf_counter()
            enc_in = enc_tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            enc_out = encoder(**enc_in).last_hidden_state.float()
            enc_mask = enc_in["attention_mask"].float()
            cls = enc_out[:, 0, :]
            enc_mean = (enc_out * enc_mask[..., None]).sum(1) / enc_mask.sum(1, keepdim=True)
            enc_max = torch.where(enc_mask[..., None] > 0, enc_out, torch.full_like(enc_out, -1e9)).max(1).values
            torch.cuda.synchronize() if device == "cuda" else None
            encoder_ms = (time.perf_counter() - t1) * 1000 / len(batch)
            ent_masked = torch.where(mask > 0, entropy, torch.zeros_like(entropy))
            for i, w in enumerate(batch):
                n = mask[i].sum().item()
                vectors = {
                    "proxy_last": last[i].cpu().numpy().astype(np.float16),
                    "proxy_mean": mean[i].cpu().numpy().astype(np.float16),
                    "proxy_entropy_pool": egtp[i].cpu().numpy().astype(np.float16),
                    "entropy_stats": np.array([ent_masked[i].sum().item() / n, entropy[i, -1].item(), n], dtype=np.float32),
                    "encoder_cls": cls[i].cpu().numpy().astype(np.float16),
                    "encoder_mean": enc_mean[i].cpu().numpy().astype(np.float16),
                    "encoder_max": enc_max[i].cpu().numpy().astype(np.float16),
                }
                for index, states in intermediate.items():
                    vectors[f"proxy_last_layer{index}"] = states[i, -1, :].cpu().numpy().astype(np.float16)
                for point_id, position in zip(w["shared_points"], w["shared_positions"]):
                    reps["point_id"].append(point_id)
                    for key, value in vectors.items():
                        reps[key].append(value)
                    timing.append({"point_id": point_id, "position": position, "tokens": int(n), "proxy_ms": proxy_ms, "encoder_ms": encoder_ms, "shared_with": len(w["shared_points"]) - 1})
            del hidden, entropy, mask, last, mean, weights, egtp, enc_out, enc_mask, cls, enc_mean, enc_max, ent_masked
            if device == "cuda":


                torch.cuda.empty_cache()
            if len(reps["point_id"]) >= 500:
                flush_part()
            if start % (args.batch_size * 50) == 0:
                print(f"{len(done_points) + start + len(batch)}/{len(windows)} windows, {proxy_ms:.1f} ms/point proxy, {encoder_ms:.1f} ms/point encoder, {torch.cuda.memory_reserved() / 2**30 if device == 'cuda' else 0:.1f} GB reserved", flush=True)
    flush_part()
    parts = [np.load(part, allow_pickle=False) for part in sorted(part_dir.glob("part-*.npz"))]
    merged = {k: np.concatenate([part[k] for part in parts]) for k in parts[0].files}
    order = {w["point_id"]: i for i, w in enumerate(windows)}
    keep = np.argsort([order[p] for p in merged["point_id"].tolist()])
    np.savez_compressed(args.output / "representations.npz", **{k: v[keep] for k, v in merged.items()})
    print("done", flush=True)


if __name__ == "__main__":
    main()
