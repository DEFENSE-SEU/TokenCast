from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import json
import lightgbm as lgb
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

OUT_DIR = Path("workspace/whitebox")
TAG = sys.argv[1] if len(sys.argv) > 1 else ""
TARGETS = ("remaining_out", "remaining_total")
BB_BASE = ["turn", "in_tokens", "spent_total_before", "n_guesses_before",
           "n_invalid_before", "best_green_before"]
WB_SCALAR = ["entropy", "margin", "eos_prob", "hidden_norm"]
N_PCA = 16
K = 5
SEEDS = tuple(range(8))


def load() -> list[dict]:
    rows = [json.loads(ln) for ln in (OUT_DIR / f"pilot_rows{TAG}.jsonl").open(encoding="utf-8")]
    vecs = dict(np.load(OUT_DIR / f"pilot_vecs{TAG}.npz"))
    by_game: dict[tuple, list[dict]] = {}
    for r in rows:
        by_game.setdefault((r["game_id"], r["run_id"]), []).append(r)
    for group in by_game.values():
        group.sort(key=lambda r: r["turn"])
        total = group[0]["remaining_total"]
        for r in group:
            r["spent_total_before"] = total - r["remaining_total"]
            r["_hidden"] = vecs[r["row_id"]].astype(np.float32)
    return rows


def grouped_folds(groups: list[str], seed: int, k: int):

    uniq = sorted(set(groups))
    k = min(k, len(uniq))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    fold_of = {uniq[order[i]]: i % k for i in range(len(uniq))}
    g = np.array(groups)
    for f in range(k):
        te = np.array([i for i in range(len(g)) if fold_of[g[i]] == f])
        tr = np.array([i for i in range(len(g)) if fold_of[g[i]] != f])
        yield tr, te


def lgbm() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="quantile", alpha=0.5, n_estimators=300, learning_rate=0.04,
        num_leaves=15, min_child_samples=8, reg_lambda=1.0, verbose=-1, seed=0)


def build(base: np.ndarray | None, use_hidden: bool, H: np.ndarray,
          tr: np.ndarray, te: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    ptr, pte = [], []
    if base is not None:
        ptr.append(base[tr])
        pte.append(base[te])
    if use_hidden:
        sc = StandardScaler().fit(H[tr])
        n = min(N_PCA, len(tr) - 1, H.shape[1])
        pca = PCA(n_components=n).fit(sc.transform(H[tr]))
        ptr.append(pca.transform(sc.transform(H[tr])))
        pte.append(pca.transform(sc.transform(H[te])))
    return np.hstack(ptr), np.hstack(pte)


def cv_model(base, use_hidden, H, y, groups, seed) -> float:
    err = tot = 0.0
    for tr, te in grouped_folds(groups, seed, K):
        Xtr, Xte = build(base, use_hidden, H, tr, te)
        p = lgbm().fit(Xtr, y[tr]).predict(Xte)
        err += float(np.abs(p - y[te]).sum())
        tot += float(np.abs(y[te]).sum())
    return err / tot


def cv_mean(y, groups, seed) -> float:
    err = tot = 0.0
    for tr, te in grouped_folds(groups, seed, K):
        p = float(y[tr].mean())
        err += float(np.abs(p - y[te]).sum())
        tot += float(np.abs(y[te]).sum())
    return err / tot


def main() -> int:
    rows = load()
    is_cand = ("n_consistent_before" in rows[0]
               and len({r["n_consistent_before"] for r in rows}) > 1)
    if not is_cand:
        rows = [r for r in rows if r["turn"] >= 1]

    groups = [r["game_id"] for r in rows]
    H = np.stack([r["_hidden"] for r in rows])

    def mat(names): return np.array([[float(r[n]) for n in names] for r in rows], dtype=np.float32)
    Xturn, Xbb, Xscalar = mat(["turn"]), mat(BB_BASE), mat(WB_SCALAR)


    specs = [
        ("turn_only", Xturn, False),
        ("BB", Xbb, False),
        ("BB+WBscalar", np.hstack([Xbb, Xscalar]), False),
        ("WBhidden", None, True),
        ("BB+WBhidden", Xbb, True),
    ]
    if is_cand:
        Xbo = np.hstack([Xbb, mat(["n_consistent_before"])])
        specs += [("BB+cand", Xbo, False), ("BB+cand+WBhidden", Xbo, True)]


    per_game = {}
    for r in rows:
        per_game[r["game_id"]] = (r["outcome"], r["n_turns"])
    outc = Counter(o for o, _ in per_game.values())
    nturns = sorted(t for _, t in per_game.values())
    print(f"tag='{TAG or '(open-vocab)'}'  regime={'candidate' if is_cand else 'open-vocab'}"
          f"  rows={len(rows)}  games={len(per_game)}")
    print(f"outcomes(by game): {dict(outc)}  n_turns spread: "
          f"min={nturns[0]} median={nturns[len(nturns)//2]} max={nturns[-1]}")

    for target in TARGETS:
        y = np.array([float(r[target]) for r in rows])
        res = {"mean": np.array([cv_mean(y, groups, s) for s in SEEDS])}
        for name, base, uh in specs:
            res[name] = np.array([cv_model(base, uh, H, y, groups, s) for s in SEEDS])

        bb = res["BB"]
        star = "  <== PRIMARY" if target == "remaining_out" else ""
        print(f"\n=== target={target}  (mean={y.mean():.0f} cv={y.std()/y.mean():.2f}){star} ===")
        print("WAPE mean +/- std over 8 game-partitions (grouped CV, LightGBM quantile):")
        for name in ["mean", "turn_only", "BB", "BB+WBscalar", "WBhidden", "BB+WBhidden",
                     *(["BB+cand", "BB+cand+WBhidden"] if is_cand else [])]:
            w = res[name]
            tag = ""
            if ("WB" in name) and name != "BB":
                base_cmp = res["BB+cand"] if name == "BB+cand+WBhidden" else bb
                d = w - base_cmp
                wins = int((w < base_cmp).sum())
                cmp = "BB+cand" if name == "BB+cand+WBhidden" else "BB"
                tag = f"   d vs {cmp} {d.mean():+.4f}  (beats it {wins}/{len(SEEDS)})"
            print(f"  {name:18s} {w.mean():.4f} +/- {w.std():.4f}{tag}")

    print("\nPearson r of each scalar internal signal vs remaining_out:")
    yout = np.array([float(r["remaining_out"]) for r in rows])
    for n in WB_SCALAR:
        x = np.array([float(r[n]) for r in rows])
        rr = float(np.corrcoef(x, yout)[0, 1]) if x.std() > 0 else 0.0
        print(f"  {n:12s} r={rr:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
