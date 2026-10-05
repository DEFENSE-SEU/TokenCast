import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from prepare_qwen38_segments import digest, selected_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("workspace/runs/qwen38-segment-expansion-v2")
    )
    args = parser.parse_args()
    manifest = args.dataset / "task_size_enrichment.json"
    if manifest.exists():
        raise ValueError("enrichment already recorded")
    original = args.dataset / "provenance.json"
    provenance = json.loads(original.read_text(encoding="utf-8"))
    wanted = {row["execution_id"] for row in provenance["accepted"]}
    source = Path(provenance["source"]) / "runs.jsonl.gz"
    sizes, text_hashes = {}, {}
    for row in selected_records(source, wanted):
        task = row.get("task") or row.get("job", {}).get("task")
        if not isinstance(task, str) or not task:
            raise ValueError("initial task text unavailable")
        raw = task.encode("utf-8")
        sizes[row["execution_id"]] = len(raw)
        text_hashes[row["execution_id"]] = hashlib.sha256(raw).hexdigest()
    assert set(sizes) == wanted
    records = {}
    for name in ("train", "validation", "calibration", "test", "training_segments"):
        path = args.dataset / (name + ".npz")
        before_hash = digest(path)
        with np.load(path, allow_pickle=True) as archive:
            before = {key: archive[key] for key in archive.files}
        assert "task_text_bytes" not in before
        after = dict(
            before,
            task_text_bytes=np.array([sizes[str(run)] for run in before["run_id"]], dtype=np.int64),
        )
        temporary = path.with_suffix(".enriched.npz")
        np.savez_compressed(temporary, **after)
        with np.load(temporary, allow_pickle=True) as check:


            for key, value in before.items():
                assert check[key].dtype == value.dtype
                np.testing.assert_array_equal(check[key], value, err_msg=key)
            np.testing.assert_array_equal(check["task_text_bytes"], after["task_text_bytes"])
        temporary.replace(path)
        records[name] = {
            "before_sha256": before_hash,
            "after_sha256": digest(path),
            "unchanged_prior_columns": list(before),
            "rows": len(before["n"]),
        }
    report = {
        "schema_version": 1,
        "field": "task_text_bytes",
        "definition": "UTF-8 byte length of initial actor task text, available before execution",
        "source_sha256": digest(source),
        "original_provenance_sha256": digest(original),
        "adapter_sha256": digest(Path("src/token_prediction/case_study/qwen38_adapter.py")),
        "script_sha256": digest(Path(__file__)),
        "task_text_sha256_by_run": text_hashes,
        "task_text_bytes_by_run": sizes,
        "files": records,
        "test_evaluated": False,
        "all_prior_arrays_exactly_equal": True,
        "paid_api_calls": 0,
    }
    manifest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"enrichment": str(manifest), "runs": len(sizes), "verified_files": len(records)}
        )
    )


if __name__ == "__main__":
    main()
