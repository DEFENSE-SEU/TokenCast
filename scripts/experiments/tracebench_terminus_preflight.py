import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

from tracebench_terminus_point_runtime import ALLOWED_ROUTES


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "workspace/experiments/tracebench_cross_harness_feasibility_20260923"
INDEX_PATH = WORK / "terminus_artifact_index.metadata.json"
LOCK_PATH = WORK / "terminus_point_protocol.final.lock.json"
PASS_PATH = WORK / "independent_point_pretest_pass.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def preflight() -> tuple[dict, list[dict], str]:
    if not LOCK_PATH.exists() or not PASS_PATH.exists():
        raise RuntimeError("Final protocol and independent pass receipt are required before test access")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    review = json.loads(PASS_PATH.read_text(encoding="utf-8"))
    lock_hash = sha256(LOCK_PATH)
    if review.get("status") != "pass" or review.get("protocol_sha256") != lock_hash:
        raise RuntimeError("Independent pretest review has not passed this exact protocol")
    if lock.get("status") != "final_point_only_locked" or lock.get("test_content_read_before_lock") is not False:
        raise RuntimeError("Final protocol status invalid")
    if sha256(INDEX_PATH) != lock["artifact_index_sha256"]:
        raise RuntimeError("Artifact index changed")
    index = json.loads(INDEX_PATH.read_text())
    if index["revision"] != lock["source_revision"]:
        raise RuntimeError("Source revision mismatch")
    fit_ids = set(lock["fit_task_ids"])
    cal_ids = set(lock["calibration_task_ids"])
    test_ids = set(lock["test_task_ids"])
    if fit_ids & cal_ids or (fit_ids | cal_ids) & test_ids or lock["excluded_probe_task_id"] in fit_ids | cal_ids | test_ids:
        raise RuntimeError("Task identity split overlaps")
    rows = [row for row in index["artifacts"] if row["task_name"] in test_ids]
    if len(rows) != 563 or len(test_ids) != 138 or any(row["group"] != "test" or row["agent"] != "Terminus2" for row in rows):
        raise RuntimeError("Pinned test roster mismatch")
    for key, relative in lock["code_files"].items():
        path = ROOT / relative
        if sha256(path) != lock["code_sha256"][key]:
            raise RuntimeError(f"Code hash changed: {key}")
    for key, relative in lock["model_files"].items():
        path = ROOT / relative
        if sha256(path) != lock["model_sha256"][key]:
            raise RuntimeError(f"Model hash changed: {key}")
    for package, version in lock["dependencies"].items():
        if importlib.metadata.version(package) != version:
            raise RuntimeError(f"Dependency version changed: {package}")
    if platform.python_version() != lock["python_version"]:
        raise RuntimeError("Python version changed")
    if lock["allowed_litellm_model_routes_by_manifest_model"] != {model: sorted(routes) for model, routes in ALLOWED_ROUTES.items()}:
        raise RuntimeError("Model-route mapping changed")
    if sha256(WORK / "point_visible_fit/visible_fit_report.json") != lock["visible_fit_report_sha256"]:
        raise RuntimeError("Development fit report changed")
    if sha256(WORK / "point_fit/fit_report.json") != lock["original_fit_report_sha256"]:
        raise RuntimeError("Original fit report changed")
    if sha256(WORK / "runtime_development_smoke_v5.json") != lock["development_smoke_sha256"]:
        raise RuntimeError("Old-identity smoke receipt changed")
    if sha256(WORK / "development_visible_no_go.md") != lock["budget_no_go_report_sha256"]:
        raise RuntimeError("Development budget no-go report changed")
    return lock, sorted(rows, key=lambda row: row["traj_id"]), lock_hash
