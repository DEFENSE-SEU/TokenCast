from __future__ import annotations

import csv
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


class EvidenceError(ValueError):
    pass


_Pair = tuple[str, str]
_METRIC_FIELDS = (
    "request_rows",
    "requests",
    "checksum_rows",
    "stale_checksum_rows",
    "invalidation_rows",
    "applied_invalidations",
    "deployment_coverages",
)
_CSV_FIELDS = ("region", "key", "impacted", *_METRIC_FIELDS)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError(f"duplicate JSON member {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise EvidenceError(f"non-standard JSON constant {value}")


def _json_value(payload: bytes, source: str, line: int) -> object:
    try:
        text = payload.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except EvidenceError as exc:
        raise EvidenceError(f"{source}:{line}: {exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise EvidenceError(f"{source}:{line}: invalid JSON record") from exc


def _jsonl(directory: Path, source: str) -> list[tuple[int, object]]:
    raw = (directory / source).read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise EvidenceError(f"{source}: UTF-8 BOM is invalid")
    if not raw:
        return []
    if not raw.endswith(b"\n"):
        raise EvidenceError(f"{source}: unterminated final record")
    records: list[tuple[int, object]] = []
    for line, physical in enumerate(raw.split(b"\n")[:-1], 1):
        payload = physical[:-1] if physical.endswith(b"\r") else physical
        if not payload:
            raise EvidenceError(f"{source}:{line}: blank record")
        records.append((line, _json_value(payload, source, line)))
    return records


def _scalar_string(value: object) -> bool:
    return (
        type(value) is str
        and bool(value)
        and "\r" not in value
        and "\n" not in value
        and not any(0xD800 <= ord(character) <= 0xDFFF for character in value)
    )


def _tick(value: object) -> bool:
    return type(value) is int and value >= 0


def _positive_integer(value: object) -> bool:
    return type(value) is int and value > 0


def _exact_object(value: object, fields: set[str], source: str, line: int) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise EvidenceError(f"{source}:{line}: record schema differs")
    return value


def _pair(region: object, key: object, source: str, line: int) -> _Pair:
    if not _scalar_string(region) or not _scalar_string(key):
        raise EvidenceError(f"{source}:{line}: invalid pair")
    return region, key


def _citation(source: str, line: int) -> dict[str, object]:
    return {"line": line, "source": source}


def _requests(directory: Path) -> list[dict[str, object]]:
    source = "requests.jsonl"
    records: list[dict[str, object]] = []
    for line, value in _jsonl(directory, source):
        record = _exact_object(value, {"key", "region", "requests", "tick"}, source, line)
        pair = _pair(record["region"], record["key"], source, line)
        if not _tick(record["tick"]) or not _positive_integer(record["requests"]):
            raise EvidenceError(f"{source}:{line}: invalid request values")
        records.append(
            {
                "key": pair[1],
                "region": pair[0],
                "requests": record["requests"],
                "tick": record["tick"],
            }
        )
    return records


def _checksum_tick(value: str, line: int) -> int:
    if not value.isascii() or not value.isdigit() or (len(value) > 1 and value[0] == "0"):
        raise EvidenceError(f"checksums.csv:{line}: invalid tick")
    return int(value)


def _checksums(directory: Path) -> list[dict[str, object]]:
    source = "checksums.csv"
    raw = (directory / source).read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise EvidenceError(f"{source}: UTF-8 BOM is invalid")
    if not raw or not raw.endswith(b"\n"):
        raise EvidenceError(f"{source}: missing header or final LF")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvidenceError(f"{source}: invalid UTF-8") from exc

    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        header = next(reader)
    except (StopIteration, csv.Error) as exc:
        raise EvidenceError(f"{source}: missing header") from exc
    if header != ["tick", "region", "key", "cache_checksum", "origin_checksum"]:
        raise EvidenceError(f"{source}: header differs")

    records: list[dict[str, object]] = []
    seen: set[tuple[int, str, str]] = set()
    previous_line = reader.line_num
    try:
        for row in reader:
            line = reader.line_num
            if line != previous_line + 1 or len(row) != 5:
                raise EvidenceError(f"{source}:{line}: record must occupy one physical line")
            previous_line = line
            tick_text, region, key, cache_checksum, origin_checksum = row
            pair = _pair(region, key, source, line)
            if not _scalar_string(cache_checksum) or not _scalar_string(origin_checksum):
                raise EvidenceError(f"{source}:{line}: invalid checksum")
            tick = _checksum_tick(tick_text, line)
            identity = (tick, *pair)
            if identity in seen:
                raise EvidenceError(f"{source}:{line}: duplicate pair observation")
            seen.add(identity)
            records.append(
                {
                    "cache_checksum": cache_checksum,
                    "key": pair[1],
                    "line": line,
                    "origin_checksum": origin_checksum,
                    "region": pair[0],
                    "tick": tick,
                }
            )
    except csv.Error as exc:
        raise EvidenceError(f"{source}: invalid CSV") from exc
    return records


def _deployments(directory: Path) -> list[dict[str, object]]:
    source = "deployments.jsonl"
    deployments: list[dict[str, object]] = []
    identifiers: set[str] = set()
    for line, value in _jsonl(directory, source):
        record = _exact_object(value, {"coverage", "deployment_id", "tick"}, source, line)
        identifier = record["deployment_id"]
        coverage = record["coverage"]
        if (
            not _scalar_string(identifier)
            or identifier in identifiers
            or not _tick(record["tick"])
            or type(coverage) is not list
            or not coverage
        ):
            raise EvidenceError(f"{source}:{line}: invalid deployment")
        identifiers.add(identifier)
        pairs: set[_Pair] = set()
        for item in coverage:
            pair_record = _exact_object(item, {"key", "region"}, source, line)
            pair = _pair(pair_record["region"], pair_record["key"], source, line)
            if pair in pairs:
                raise EvidenceError(f"{source}:{line}: duplicate coverage pair")
            pairs.add(pair)
        deployments.append(
            {
                "coverage": pairs,
                "deployment_id": identifier,
                "line": line,
                "tick": record["tick"],
            }
        )
    return deployments


def _invalidations(directory: Path) -> list[dict[str, object]]:
    source = "invalidations.jsonl"
    invalidations: list[dict[str, object]] = []
    seen: set[tuple[int, str, str]] = set()
    for line, value in _jsonl(directory, source):
        record = _exact_object(value, {"key", "region", "status", "tick"}, source, line)
        pair = _pair(record["region"], record["key"], source, line)
        status = record["status"]
        if (
            not _tick(record["tick"])
            or type(status) is not str
            or status not in {"applied", "rejected"}
        ):
            raise EvidenceError(f"{source}:{line}: invalid invalidation")
        identity = (record["tick"], *pair)
        if identity in seen:
            raise EvidenceError(f"{source}:{line}: duplicate pair invalidation")
        seen.add(identity)
        invalidations.append(
            {
                "key": pair[1],
                "line": line,
                "region": pair[0],
                "status": status,
                "tick": record["tick"],
            }
        )
    return invalidations


def _record_pair(record: dict[str, object]) -> _Pair:
    return record["region"], record["key"]


def _impact(
    checksums: list[dict[str, object]],
    invalidations: list[dict[str, object]],
) -> tuple[list[dict[str, object]], set[_Pair]]:
    checksum_groups: dict[_Pair, list[dict[str, object]]] = defaultdict(list)
    invalidation_groups: dict[_Pair, list[dict[str, object]]] = defaultdict(list)
    for record in checksums:
        checksum_groups[_record_pair(record)].append(record)
    for record in invalidations:
        invalidation_groups[_record_pair(record)].append(record)

    impacted: list[dict[str, object]] = []
    impacted_pairs: set[_Pair] = set()
    for pair, records in checksum_groups.items():
        stale = [
            record for record in records if record["cache_checksum"] != record["origin_checksum"]
        ]
        if not stale:
            continue
        start = min(stale, key=lambda record: record["tick"])
        applied = [
            record
            for record in invalidation_groups[pair]
            if record["status"] == "applied" and record["tick"] >= start["tick"]
        ]
        if not applied:
            raise EvidenceError(f"impacted pair {pair!r} has no applied invalidation")
        invalidation = min(applied, key=lambda record: record["tick"])
        matching = [
            record
            for record in records
            if record["tick"] > invalidation["tick"]
            and record["cache_checksum"] == record["origin_checksum"]
        ]
        if not matching:
            raise EvidenceError(f"impacted pair {pair!r} has no recovery checksum")
        recovery = min(matching, key=lambda record: record["tick"])
        impacted_pairs.add(pair)
        impacted.append(
            {
                "applied_invalidation_tick": invalidation["tick"],
                "citations": {
                    "applied_invalidation": _citation("invalidations.jsonl", invalidation["line"]),
                    "recovery": _citation("checksums.csv", recovery["line"]),
                    "stale_start": _citation("checksums.csv", start["line"]),
                },
                "key": pair[1],
                "recovery_tick": recovery["tick"],
                "region": pair[0],
                "stale_start_tick": start["tick"],
            }
        )
    if not impacted:
        raise EvidenceError("evidence contains no impacted pair")
    impacted.sort(key=lambda item: (item["region"], item["key"]))
    return impacted, impacted_pairs


def _causal_deployment(
    deployments: list[dict[str, object]],
    impacted_pairs: set[_Pair],
    start_tick: int,
) -> dict[str, object]:
    eligible = [
        deployment
        for deployment in deployments
        if deployment["tick"] <= start_tick and impacted_pairs <= deployment["coverage"]
    ]
    if not eligible:
        raise EvidenceError("no deployment covers every impacted pair")
    latest_tick = max(deployment["tick"] for deployment in eligible)
    latest = [deployment for deployment in eligible if deployment["tick"] == latest_tick]
    if len(latest) != 1:
        raise EvidenceError("latest causal deployment is not unique")
    selected = latest[0]
    return {
        "citation": _citation("deployments.jsonl", selected["line"]),
        "deployment_id": selected["deployment_id"],
        "tick": selected["tick"],
    }


def _reconciliation(
    requests: list[dict[str, object]],
    checksums: list[dict[str, object]],
    deployments: list[dict[str, object]],
    invalidations: list[dict[str, object]],
    impacted_pairs: set[_Pair],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    observed = {_record_pair(record) for record in requests}
    observed.update(_record_pair(record) for record in checksums)
    observed.update(_record_pair(record) for record in invalidations)
    for deployment in deployments:
        observed.update(deployment["coverage"])

    rows: dict[_Pair, dict[str, object]] = {}
    for region, key in observed:
        row: dict[str, object] = {
            "impacted": (region, key) in impacted_pairs,
            "key": key,
            "region": region,
        }
        row.update({field: 0 for field in _METRIC_FIELDS})
        rows[(region, key)] = row

    for record in requests:
        row = rows[_record_pair(record)]
        row["request_rows"] += 1
        row["requests"] += record["requests"]
    for record in checksums:
        row = rows[_record_pair(record)]
        row["checksum_rows"] += 1
        if record["cache_checksum"] != record["origin_checksum"]:
            row["stale_checksum_rows"] += 1
    for record in invalidations:
        row = rows[_record_pair(record)]
        row["invalidation_rows"] += 1
        if record["status"] == "applied":
            row["applied_invalidations"] += 1
    for deployment in deployments:
        for pair in deployment["coverage"]:
            rows[pair]["deployment_coverages"] += 1

    ordered = [rows[pair] for pair in sorted(rows)]
    totals = {
        field: sum(row[field] for row in ordered)
        for field in _METRIC_FIELDS
    }
    return ordered, totals


def analyze_evidence(evidence_dir: str | Path) -> dict[str, Any]:


    directory = Path(evidence_dir)
    if not directory.is_dir():
        raise OSError("evidence directory is unavailable")
    requests = _requests(directory)
    checksums = _checksums(directory)
    deployments = _deployments(directory)
    invalidations = _invalidations(directory)
    impacted, impacted_pairs = _impact(checksums, invalidations)
    start_tick = min(item["stale_start_tick"] for item in impacted)
    end_tick = max(item["recovery_tick"] for item in impacted)
    causal = _causal_deployment(deployments, impacted_pairs, start_tick)
    reconciliation, totals = _reconciliation(
        requests,
        checksums,
        deployments,
        invalidations,
        impacted_pairs,
    )
    return {
        "incident": {
            "causal_deployment": causal,
            "impacted_pairs": impacted,
            "interval": {"end_tick": end_tick, "start_tick": start_tick},
            "schema_version": "cache-drift-incident-v1",
            "totals": totals,
        },
        "reconciliation": reconciliation,
    }


def incident_bytes(incident: dict[str, Any]) -> bytes:


    return (
        json.dumps(incident, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def reconciliation_bytes(rows: list[dict[str, Any]]) -> bytes:


    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(_CSV_FIELDS)
    for row in rows:
        writer.writerow(
            [
                row["region"],
                row["key"],
                "true" if row["impacted"] else "false",
                *(row[field] for field in _METRIC_FIELDS),
            ]
        )
    return output.getvalue().encode("utf-8")


def write_artifacts(
    evidence_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:


    output = Path(output_dir)
    if not output.is_dir():
        raise OSError("output directory is unavailable")
    result = analyze_evidence(evidence_dir)
    (output / "incident.json").write_bytes(incident_bytes(result["incident"]))
    (output / "reconciliation.csv").write_bytes(reconciliation_bytes(result["reconciliation"]))
    return result
