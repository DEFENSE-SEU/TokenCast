from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "configs/collection/mmlupro_gpt54_low_eight_sufy_pilot_20260923.json"
MODEL = ROOT / "workspace/runs/composable-multidomain-history-v1/model.pt"
PROVENANCE = MODEL.with_name("provenance.json")
BASELINE = ROOT / "workspace/experiments/composable_position_ablation_20260923/history_median_diagnostic.json"
ANSWER_AUDIT = ROOT / "workspace/experiments/qwen38_domain_correctness_20260923/run_audit.jsonl"
OUTPUT = ROOT / "workspace/experiments/mmlupro_gpt54_low_eight_transfer_20260923"
POSITIONS = ("task_start", "call_start", "in_call_update", "task_update")
EXPECTED_IDS = (
    "mmlupro-10455", "mmlupro-2905", "mmlupro-7058", "mmlupro-9487",
    "mmlupro-8554", "mmlupro-10924", "mmlupro-4076", "mmlupro-1039",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def declaration() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if (manifest["manifest_id"] != "mmlupro-gpt54-low-eight-sufy-pilot-20260923"
            or tuple(manifest["schedule"]["task_order"]) != EXPECTED_IDS
            or len(manifest["tasks"]) != 8):
        raise ValueError("the frozen eight-task manifest differs")
    tasks = {task["task_id"]: task for task in manifest["tasks"]}
    if set(tasks) != set(EXPECTED_IDS):
        raise ValueError("manifest task roster differs")
    partition = json.loads(PROVENANCE.read_text(encoding="utf-8"))["task_family_partition"]
    medians = json.loads(BASELINE.read_text(encoding="utf-8"))["positions"]
    paths = {
        "manifest": MANIFEST, "model": MODEL, "training_provenance": PROVENANCE,
        "training_only_medians": BASELINE, "prior_original_key_audit": ANSWER_AUDIT,
        "freeze_script": Path(__file__),
        "causal_segment_builder": ROOT / "src/token_prediction/case_study/segments.py",
        "composable_predictor": ROOT / "src/token_prediction/estimators/composable_segment.py",
        "request_surface_builder": ROOT / "src/token_prediction/case_study/request_surface.py",
        "dsh_capture_reader": ROOT / "src/token_prediction/collection/readers/dsh_session.py",
    }
    pack_receipts = {}
    for task_id in EXPECTED_IDS:
        task = tasks[task_id]
        pack = (ROOT / task["pack_path"]).resolve()
        if not pack.is_relative_to((ROOT / "benchmarks/dsh").resolve()):
            raise ValueError(f"pack path leaves current DSH benchmark tree: {task_id}")
        pack_json = json.loads((pack / "pack.json").read_text(encoding="utf-8"))
        if pack_json["task_id"] != task_id or pack_json["oracle"] is not None:
            raise ValueError(f"pack contract differs: {task_id}")
        seed = Path(pack_json["seed_workspace"]).resolve()
        head = subprocess.run(["git", "-C", str(seed), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        if head != pack_json["source"]["base_commit"]:
            raise ValueError(f"seed HEAD differs: {task_id}")
        pack_receipts[task_id] = {
            "pack_path": task["pack_path"], "pack_json_sha256": sha256(pack / "pack.json"),
            "task_prompt_sha256": sha256(pack / "task.md"), "seed_head": head,
            "source_partition": partition.get(task_id, "source_unseen"),
        }
    if set(medians) != set(POSITIONS):
        raise ValueError("training-only baseline positions differ")
    return {
        "status": "frozen_before_eight_paid_task_calls",
        "purpose": "small descriptive same-task cross-model development pilot",
        "planned_task_count": 8, "task_order": list(EXPECTED_IDS),
        "manifest_id": manifest["manifest_id"],
        "condition_id": "gpt-5-4-low-sufy-eight", "repeat_id": "r1",
        "paths_sha256": {name: {"path": str(path.resolve()), "sha256": sha256(path)}
                         for name, path in paths.items()},
        "packs": pack_receipts,
        "baseline_constants": {point: medians[point]["training_weighted_median"]
                               for point in POSITIONS},
        "prediction_policy": {
            "model": "frozen shared composable checkpoint, no new fitting or adaptation",
            "features": "load_run(request_surface_features=True, history_cost_features=True); causal query only",
            "positions": list(POSITIONS), "checkpoint_policy": "first_per_call",
            "rows": "one whole-allowance suffix per observed nonterminal cutoff; exclude observed terminal zero rows",
            "forecast": "median of 128 draws; no calibration; fixed deterministic point-id seed",
            "task_prediction_unavailable": "if any of 128 samples has horizon_exhausted; retain row and disclose",
            "target": "provider-reported input plus output of every attempt, full current call for Call Start/In-call Update, remaining task for Task Start/Task Update",
            "score": "within-task mean absolute error, then equal-weighted mean over tasks, paired model/baseline on identical available cutoffs",
            "denominator": "all eight roster rows retained; unfinished, ineligible, missing, and unavailable points reported separately; no replacement tasks",
            "uncertainty": "paired task bootstrap, 10000 draws, seed 20260923; descriptive",
            "accuracy": "strict answer.md first-line letter and matching submission.patch first added line against preexisting original-key audit; report completion and all-eight yields",
            "prediction_export": "label-free predictions file hashed before a separate score phase joins labels and answer keys",
        },
        "limitations": ["seven of eight tasks occur in source train or validation partitions",
                        "sample too small for PDF Table 7 or F.1 Table 8",
                        "source input files contain retrospective usage; causal query excludes it"],
        "paid_provider_calls_from_freeze": 0,
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "predeclared.json"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(declaration(), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"path": str(path), "sha256": sha256(path),
                      "task_count": 8}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
