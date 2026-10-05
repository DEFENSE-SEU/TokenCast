import hashlib
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "workspace/experiments/tracebench_cross_harness_feasibility_20260923"
DEST = ROOT / "workspace/experiments/tracebench_terminus_point_repro_supplement_20260924"
ZIP = DEST.with_suffix(".zip")

FILES = {
    "protocol/terminus_point_protocol.final.lock.json": SOURCE / "terminus_point_protocol.final.lock.json",
    "protocol/terminus_artifact_index.metadata.json": SOURCE / "terminus_artifact_index.metadata.json",
    "protocol/roster.metadata.json": SOURCE / "roster.metadata.json",
    "protocol/independent_point_pretest_pass.json": SOURCE / "independent_point_pretest_pass.json",
    "protocol/independent_download_recovery_pass.json": SOURCE / "independent_download_recovery_pass.json",
    "protocol/test_download_receipt.json": SOURCE / "test_download_receipt.json",
    "fit/direct_visible.txt": SOURCE / "point_visible_fit/direct_visible.txt",
    "fit/residual_visible.txt": SOURCE / "point_visible_fit/residual_visible.txt",
    "fit/offsets.json": SOURCE / "point_visible_fit/offsets.json",
    "fit/visible_fit_report.json": SOURCE / "point_visible_fit/visible_fit_report.json",
    "fit/affine.json": SOURCE / "point_fit/affine.json",
    "predictions/point_prediction_started.json": SOURCE / "point_prediction_started.json",
    "predictions/point_predictions.jsonl": SOURCE / "point_predictions.jsonl",
    "predictions/point_prediction_coverage.jsonl": SOURCE / "point_prediction_coverage.jsonl",
    "predictions/point_prediction_sealed.json": SOURCE / "point_prediction_sealed.json",
    "score/point_score_started.json": SOURCE / "point_score_started.json",
    "score/point_scored_labels.jsonl": SOURCE / "point_scored_labels.jsonl",
    "score/point_score.json": SOURCE / "point_score.json",
    "code/qualify_tracebench_terminus_development.py": ROOT / "scripts/experiments/qualify_tracebench_terminus_development.py",
    "code/tracebench_terminus_point_runtime.py": ROOT / "scripts/experiments/tracebench_terminus_point_runtime.py",
    "code/tracebench_terminus_preflight.py": ROOT / "scripts/experiments/tracebench_terminus_preflight.py",
    "code/fit_tracebench_visible_transport.py": ROOT / "scripts/experiments/fit_tracebench_visible_transport.py",
    "code/fit_tracebench_terminus_point.py": ROOT / "scripts/experiments/fit_tracebench_terminus_point.py",
    "code/predict_tracebench_terminus_point.py": ROOT / "scripts/experiments/predict_tracebench_terminus_point.py",
    "code/score_tracebench_terminus_point.py": ROOT / "scripts/experiments/score_tracebench_terminus_point.py",
    "code/audit_tracebench_terminus_point_posttest.py": ROOT / "scripts/experiments/audit_tracebench_terminus_point_posttest.py",
    "code/package_tracebench_terminus_point.py": Path(__file__).resolve(),
    "audit/independent_point_posttest_audit.json": SOURCE / "independent_point_posttest_audit.json",
    "audit/independent_point_posttest_audit.md": SOURCE / "independent_point_posttest_audit.md",
    "notes/development_visible_no_go.md": SOURCE / "development_visible_no_go.md",
}

PREDICTION_KEYS = {
    "sequence", "task_id", "trajectory_id", "model", "episode", "I_previous",
    "O_previous", "O_visible", "prompt_proxy_tokens", "request_proxy_tokens",
    "message_count", "request_sha256", "forecasts",
}
LABEL_KEYS = {
    "sequence", "task_id", "trajectory_id", "model", "episode", "target",
    "O_previous", "O_visible", "reasoning_previous", "cached_previous", "forecasts",
}
COVERAGE_KEYS = {
    "trajectory_id", "task_id", "model", "episode_count", "candidate_pairs",
    "pre_gate_predictions", "pre_gate_reasons",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_numeric_rows(path: Path, expected: set[str]) -> int:
    count = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if set(row) != expected:
                raise RuntimeError(f"Unexpected source-bearing field in {path.name}, row {count + 1}")
            count += 1
    return count


def main() -> None:
    if DEST.exists() or ZIP.exists():
        raise RuntimeError("Supplement already exists; do not overwrite a sealed package")
    for source in FILES.values():
        if not source.is_file():
            raise RuntimeError(f"Required file missing: {source}")
    counts = {
        "point_predictions": validate_numeric_rows(SOURCE / "point_predictions.jsonl", PREDICTION_KEYS),
        "point_scored_labels": validate_numeric_rows(SOURCE / "point_scored_labels.jsonl", LABEL_KEYS),
        "point_prediction_coverage": validate_numeric_rows(SOURCE / "point_prediction_coverage.jsonl", COVERAGE_KEYS),
    }
    if counts != {"point_predictions": 19992, "point_scored_labels": 19986, "point_prediction_coverage": 563}:
        raise RuntimeError(f"Unexpected sealed row counts: {counts}")
    DEST.mkdir(parents=True)
    manifest = {}
    for rel, source in sorted(FILES.items()):
        target = DEST / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        manifest[rel] = {"sha256": digest(target), "bytes": target.stat().st_size}
    readme = """# Tracebench Terminus2 point-only local reproducibility record

This package accompanies the one-shot, target-harness new-task input-token experiment.
The frozen source revision is `7da2e4f45b330be8b6e8f1cff835247723cb3341` of
`Contextbench/Tracebench` (`full`). The development split had 60 fit and 26
calibration task identities; the 138 test identities were disjoint. The
observed-point score covers 19,986 successful calls on 137 tasks and 562
trajectories. The target is response-reported `usage.prompt_tokens`.

`predictions/` records forecasts sealed before opening current-response labels;
`score/point_scored_labels.jsonl` contains only numerical labels, diagnostics,
model names, and public task/trajectory identifiers. `score/point_score.json`
contains the prespecified five-model macro-WAPE and task-ID bootstrap tests.
The independent post-test audit is in `audit/`. The code and trained models
permit inspection of the predictor and source qualification gates. The source
archive bytes themselves are deliberately absent, so a full extraction audit
requires separately retrieving the pinned public dataset and verifying its
per-archive LFS SHA values in `protocol/`.

This is a local research package. The dataset card does not state an explicit
license for redistribution; confirm rights before publishing any derivative
record or raw source. Do not use this point-only test to claim online budget
enforcement, complete retry cost, task quality, or a hard error bound.

The `MANIFEST.json` SHA values cover each included file. The ZIP uses fixed
timestamps and sorted entries so its bytes are reproducible for these inputs.
"""
    (DEST / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    manifest["README.md"] = {"sha256": digest(DEST / "README.md"), "bytes": (DEST / "README.md").stat().st_size}
    (DEST / "MANIFEST.json").write_text(json.dumps({"files": manifest, "row_counts": counts}, indent=2) + "\n", encoding="utf-8", newline="\n")
    with zipfile.ZipFile(ZIP, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(DEST.rglob("*")):
            if path.is_file():
                info = zipfile.ZipInfo(path.relative_to(DEST.parent).as_posix(), date_time=(2026, 9, 24, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    with zipfile.ZipFile(ZIP) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"ZIP CRC failure: {bad}")
    print(json.dumps({"files": len(manifest), "rows": counts, "manifest_sha256": digest(DEST / "MANIFEST.json"), "zip_sha256": digest(ZIP), "zip_bytes": ZIP.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
