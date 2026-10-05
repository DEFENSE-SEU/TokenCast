from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(_REPOSITORY_ROOT / "workspace" / "models"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR", "1")

from transformers import AutoModelForCausalLM, AutoTokenizer, FineGrainedFP8Config

MODEL = "Qwen/Qwen2.5-7B-Instruct"
OUT_DIR = Path("workspace/whitebox")
WORD_LEN = 5


WORDS = [
    "apple", "beach", "chair", "dance", "eagle", "flame", "grape", "house",
    "juice", "lemon", "mango", "night", "ocean", "piano", "queen", "river",
    "stone", "table", "voice", "water", "bread", "cloud", "dream", "earth",
    "fruit", "green", "heart", "light", "money", "paint", "smile", "train",
    "world", "brave", "candy", "plant", "sugar", "happy", "music", "party",
    "black", "white", "brown", "clean", "drink", "field", "glass", "horse",
    "lunch", "phone", "clock", "beard", "grand", "sport", "north", "south",
    "storm", "brick", "crown", "flour",
]

SYSTEM_OPEN = (
    "You are playing Wordle. There is a hidden secret 5-letter English word. Each "
    "turn you guess a 5-letter word and I reply with per-letter feedback:\n"
    "  green  = correct letter in the correct position\n"
    "  yellow = letter is in the word but a different position\n"
    "  gray   = letter is not in the word\n"
    "Use the feedback to deduce the word. Never repeat a previous guess, and keep "
    "each new guess consistent with all feedback so far.\n"
    "Reason briefly (a few sentences at most), then end your reply with exactly ONE "
    "action line as the LAST line, in one of:\n"
    "ACTION: guess(word)   -- a valid 5-letter word, e.g. ACTION: guess(crane)\n"
    "ACTION: giveup        -- only if you are truly stuck\n"
    "Do not write anything after the action line."
)
_ACTION_RE = re.compile(r"ACTION:\s*(guess|giveup)\s*(?:\((.*)\))?\s*$",
                        re.IGNORECASE | re.DOTALL)


def build_system(candidates: list[str] | None) -> str:
    if not candidates:
        return SYSTEM_OPEN
    lst = ", ".join(candidates)
    return (
        "You are playing Wordle. There is a hidden secret 5-letter English word, and "
        "it is exactly one of the words in this candidate list:\n"
        f"  {lst}\n"
        "Each turn you guess a 5-letter word FROM THE LIST and I reply with per-letter "
        "feedback:\n"
        "  green  = correct letter in the correct position\n"
        "  yellow = letter is in the word but a different position\n"
        "  gray   = letter is not in the word\n"
        "Use each clue to ELIMINATE words from the list that are no longer possible, "
        "and guess a word still consistent with every clue so far. Never repeat a "
        "previous guess.\n"
        "Reason briefly (a few sentences at most), then end your reply with exactly ONE "
        "action line as the LAST line, in one of:\n"
        "ACTION: guess(word)   -- a 5-letter word FROM THE LIST, e.g. ACTION: guess(crane)\n"
        "ACTION: giveup        -- only if you are truly stuck\n"
        "Do not write anything after the action line."
    )


def feedback(secret: str, guess: str) -> list[str]:
    res = ["gray"] * WORD_LEN
    pool = list(secret)
    for i in range(WORD_LEN):
        if guess[i] == secret[i]:
            res[i] = "green"
            pool[i] = None
    for i in range(WORD_LEN):
        if res[i] == "gray" and guess[i] in pool:
            res[i] = "yellow"
            pool[pool.index(guess[i])] = None
    return res


def consistent_count(candidates: list[str], history: list[tuple[str, list[str]]]) -> int:

    def ok(w: str) -> bool:
        return all(feedback(w, g) == fb for g, fb in history)
    return sum(ok(w) for w in candidates)


def make_games(n: int, seed: int, n_cand: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(WORDS))
    games = []
    for i in range(n):
        secret = WORDS[int(idx[i % len(WORDS)])]
        game = {"game_id": f"wd{seed}_{i:03d}", "secret": secret}
        if n_cand > 0:
            others = [w for w in WORDS if w != secret]
            pick = rng.choice(len(others), size=min(n_cand - 1, len(others)),
                              replace=False)
            cand = [secret] + [others[int(j)] for j in pick]
            rng.shuffle(cand)
            game["candidates"] = cand
        games.append(game)
    return games


def parse_guess(arg: str) -> str | None:
    letters = re.findall(r"[A-Za-z]", arg)
    if len(letters) < WORD_LEN:
        return None
    return "".join(letters[:WORD_LEN]).lower()


class Qwen:
    def __init__(self) -> None:
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL, quantization_config=FineGrainedFP8Config(),
            device_map="cuda", dtype=torch.bfloat16)
        self.model.eval()
        self.im_end = self.tok.convert_tokens_to_ids("<|im_end|>")

    @torch.no_grad()
    def boundary_signals(self, input_ids: torch.Tensor) -> dict:
        out = self.model(input_ids, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[-1][0, -1]
        logits = out.logits[0, -1].float()
        logp = torch.log_softmax(logits, dim=-1)
        probs = logp.exp()
        entropy = float(-(probs * logp).sum())
        top2 = torch.topk(logits, 2).values
        return {
            "entropy": entropy,
            "margin": float(top2[0] - top2[1]),
            "eos_prob": float(probs[self.im_end]),
            "hidden_norm": float(h.float().norm()),
            "hidden": h.float().cpu().numpy().astype(np.float16),
        }

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new: int, temp: float) -> torch.Tensor:
        gen = self.model.generate(
            input_ids, max_new_tokens=max_new, do_sample=temp > 0,
            temperature=temp if temp > 0 else None, top_p=0.9 if temp > 0 else None,
            pad_token_id=self.tok.eos_token_id)
        return gen[0, input_ids.shape[1]:]


def run_game(qw: Qwen, game: dict, run_id: int, max_turns: int,
             max_new: int, temp: float, verbose: bool) -> list[dict]:
    secret = game["secret"]
    candidates = game.get("candidates")
    cand_set = set(candidates) if candidates else None
    first = ("Make your first guess from the candidate list."
             if candidates else
             "Make your first guess. The secret is a hidden 5-letter English word.")
    messages = [
        {"role": "system", "content": build_system(candidates)},
        {"role": "user", "content": first},
    ]
    history: list[tuple[str, list[str]]] = []
    rows: list[dict] = []
    n_guesses = n_invalid = no_action = best_green = 0
    outcome = "truncated"
    for turn in range(max_turns):
        enc = qw.tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
            return_dict=True).to("cuda")
        ids = enc["input_ids"]
        in_tokens = int(ids.shape[1])
        n_consistent = consistent_count(candidates, history) if candidates else 0
        sig = qw.boundary_signals(ids)
        new_ids = qw.generate(ids, max_new, temp)
        out_tokens = int(new_ids.shape[0])
        text = qw.tok.decode(new_ids, skip_special_tokens=True).strip()
        messages.append({"role": "assistant", "content": text})
        if verbose:
            print(f"\n--- {game['game_id']} turn {turn} ({out_tokens} tok) ---\n{text}",
                  flush=True)

        rows.append({
            "game_id": game["game_id"], "run_id": run_id, "turn": turn,
            "in_tokens": in_tokens, "out_tokens": out_tokens,
            "n_guesses_before": n_guesses, "n_invalid_before": n_invalid,
            "best_green_before": best_green, "n_consistent_before": n_consistent,
            "entropy": sig["entropy"], "margin": sig["margin"],
            "eos_prob": sig["eos_prob"], "hidden_norm": sig["hidden_norm"],
            "_hidden": sig["hidden"],
        })

        m = None
        for line in reversed([ln for ln in text.splitlines() if ln.strip()]):
            m = _ACTION_RE.search(line.strip())
            if m:
                break
        if not m:
            no_action += 1
            if no_action >= 2:
                outcome = "no_action_stop"
                break
            messages.append({"role": "user", "content":
                             "No valid ACTION line found. Reason briefly and end your "
                             "reply with exactly: ACTION: guess(word)"})
            continue
        no_action = 0
        if m.group(1).lower() == "giveup":
            outcome = "gaveup"
            break
        guess = parse_guess(m.group(2) or "")
        if guess is None or (cand_set is not None and guess not in cand_set):
            n_invalid += 1
            hint = ("Invalid guess. Choose exactly one word FROM THE LIST, like "
                    "ACTION: guess(crane).") if cand_set is not None else (
                    "Invalid guess. Provide exactly one 5-letter word, like "
                    "ACTION: guess(crane).")
            messages.append({"role": "user", "content": hint})
            continue
        n_guesses += 1
        fb = feedback(secret, guess)
        history.append((guess, fb))
        greens = fb.count("green")
        best_green = max(best_green, greens)
        if greens == WORD_LEN:
            outcome = "solved"
            break
        detail = ", ".join(f"{c}={col}" for c, col in zip(guess.upper(), fb))
        messages.append({"role": "user", "content":
                         f"Observation: {guess.lower()} -> {detail}."})

    total_in = sum(r["in_tokens"] for r in rows)
    total_out = sum(r["out_tokens"] for r in rows)
    spent_in = spent_out = 0
    for r in rows:
        r["outcome"] = outcome
        r["n_turns"] = len(rows)
        r["remaining_out"] = total_out - spent_out
        r["remaining_total"] = (total_in + total_out) - (spent_in + spent_out)
        spent_in += r["in_tokens"]
        spent_out += r["out_tokens"]
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--max-turns", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=220)
    ap.add_argument("--temp", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--candidates", type=int, default=0,
                    help="0 = open-vocab; K>0 = guess from a K-word list (variable term.)")
    ap.add_argument("--out-tag", type=str, default="",
                    help="suffix for output files, e.g. _cand24 (keeps regimes separate)")
    args = ap.parse_args()

    verbose = bool(os.environ.get("WB_VERBOSE"))
    torch.manual_seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    games = make_games(args.games, args.seed, args.candidates)
    qw = Qwen()

    all_rows: list[dict] = []
    vecs: dict[str, np.ndarray] = {}
    for gi, game in enumerate(games):
        for run_id in range(args.runs):
            t0 = time.time()
            rows = run_game(qw, game, run_id, args.max_turns, args.max_new,
                            args.temp, verbose)
            for r in rows:
                rid = f"{r['game_id']}_r{run_id}_t{r['turn']}"
                vecs[rid] = r.pop("_hidden")
                r["row_id"] = rid
                all_rows.append(r)
            oc = rows[-1]["outcome"] if rows else "empty"
            print(f"[{gi+1}/{len(games)}] {game['game_id']} secret={game['secret']} "
                  f"run{run_id} turns={len(rows):>2} "
                  f"out={sum(r['out_tokens'] for r in rows):>4} {oc} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    tag = args.out_tag
    rows_path = OUT_DIR / f"pilot_rows{tag}.jsonl"
    with rows_path.open("w", encoding="utf-8") as fh:
        for r in all_rows:
            fh.write(json.dumps(r) + "\n")
    np.savez_compressed(OUT_DIR / f"pilot_vecs{tag}.npz", **vecs)

    oc = Counter(r["outcome"] for r in all_rows if r["turn"] == 0)
    print(f"\nwrote {len(all_rows)} rows over {len(games)} games x {args.runs} runs "
          f"-> {rows_path}")
    print("outcomes:", dict(oc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
