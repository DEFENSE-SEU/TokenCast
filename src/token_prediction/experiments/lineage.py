from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from token_prediction.identifiers import identifier
from token_prediction.recording.redaction import redact_secrets


class ArtifactVerificationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ArtifactManifest:
    artifact_id: str
    stage_name: str
    schema_version: int
    files: dict[str, int]
    metadata: dict[str, Any]
    sha256: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "stage_name": self.stage_name,
            "schema_version": self.schema_version,
            "files": dict(sorted(self.files.items())),
            "sha256": dict(sorted(self.sha256.items())),
            "metadata": self.metadata,
        }


def _artifact_id(
    root: Path,
    *,
    stage_name: str,
    schema_version: int,
    metadata: Mapping[str, Any],
) -> str:
    reference = next(
        (
            str(metadata[name])
            for name in ("model_id", "candidate_id", "run_id", "dataset_id")
            if metadata.get(name) is not None
        ),
        root.name,
    )
    revision = next(
        (
            str(metadata[name])
            for name in ("model_revision", "epoch", "fold")
            if metadata.get(name) is not None
        ),
        "current",
    )
    return identifier("artifact", stage_name, f"schema-v{schema_version}", reference, revision)


def _is_reparse_point(metadata: os.stat_result) -> bool:
    return bool(
        getattr(metadata, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


def _payload_files(root: Path) -> tuple[dict[str, int], dict[str, str]]:
    files: dict[str, int] = {}
    hashes: dict[str, str] = {}
    directories: set[str] = set()

    def visit(directory: Path) -> None:
        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            relative = path.relative_to(root).as_posix()
            metadata = path.lstat()
            # Linked files could change outside this artifact's directory.
            if stat.S_ISLNK(metadata.st_mode) or _is_reparse_point(metadata):
                raise ArtifactVerificationError(
                    f"artifact entry is linked or reparse-backed: {relative}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                directories.add(relative)
                visit(path)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ArtifactVerificationError(
                    f"artifact entry is not a regular file: {relative}"
                )
            # The manifest stores these hashes, so it cannot hash itself.
            if relative == "manifest.json":
                continue
            digest = hashlib.sha256()
            size = 0
            with path.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    digest.update(block)
                    size += len(block)
            if size != metadata.st_size:
                raise ArtifactVerificationError(
                    f"artifact file changed while hashing: {relative}"
                )
            files[relative] = size
            hashes[relative] = digest.hexdigest()

    visit(root)
    expected_directories = {
        parent.as_posix()
        for name in files
        for parent in PurePosixPath(name).parents
        if parent != PurePosixPath(".")
    }
    if directories != expected_directories:
        raise ArtifactVerificationError(
            "artifact directory closure differs from its payload files"
        )
    return files, hashes


def _load_manifest(path: Path) -> ArtifactManifest:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactVerificationError("artifact manifest is not valid JSON") from exc
    expected = {
        "artifact_id",
        "stage_name",
        "schema_version",
        "files",
        "sha256",
        "metadata",
    }
    keys = set(document) if isinstance(document, dict) else set()
    if not isinstance(document, dict) or keys != expected:
        raise ArtifactVerificationError("artifact manifest schema does not match")
    artifact_id = document["artifact_id"]
    stage_name = document["stage_name"]
    schema_version = document["schema_version"]
    raw_files = document["files"]
    raw_hashes = document["sha256"]
    metadata = document["metadata"]
    if not isinstance(artifact_id, str) or not artifact_id.strip():
        raise ArtifactVerificationError("artifact_id is required")
    if not isinstance(stage_name, str) or not stage_name.strip():
        raise ArtifactVerificationError("stage_name is required")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version <= 0
    ):
        raise ArtifactVerificationError("schema_version must be positive")
    if not isinstance(raw_files, dict) or any(
        not isinstance(name, str)
        or not name
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        for name, size in raw_files.items()
    ):
        raise ArtifactVerificationError("artifact files must map paths to byte sizes")
    if not isinstance(metadata, dict):
        raise ArtifactVerificationError("artifact metadata must be an object")
    if (
        not isinstance(raw_hashes, dict)
        or set(raw_hashes) != set(raw_files)
        or any(
            not isinstance(name, str)
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for name, digest in raw_hashes.items()
        )
    ):
        raise ArtifactVerificationError("artifact SHA-256 inventory is invalid")
    return ArtifactManifest(
        artifact_id=artifact_id,
        stage_name=stage_name,
        schema_version=schema_version,
        files=dict(raw_files),
        metadata=dict(metadata),
        sha256=dict(raw_hashes),
    )


def publish_artifact(
    root: str | Path,
    *,
    stage_name: str,
    schema_version: int = 1,
    metadata: Mapping[str, Any] | None = None,
) -> ArtifactManifest:


    directory = Path(os.path.abspath(os.fspath(Path(root).expanduser())))
    if not directory.is_dir():
        raise ArtifactVerificationError(f"artifact directory does not exist: {directory}")
    root_metadata = directory.lstat()
    if stat.S_ISLNK(root_metadata.st_mode) or _is_reparse_point(root_metadata):
        raise ArtifactVerificationError("artifact root must be a plain directory")
    if not isinstance(stage_name, str) or not stage_name.strip():
        raise ValueError("stage_name must be a non-empty string")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version <= 0
    ):
        raise ValueError("schema_version must be a positive integer")
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"artifact already has a manifest: {directory}")
    safe_metadata = redact_secrets(dict(metadata or {}))
    files, hashes = _payload_files(directory)
    manifest = ArtifactManifest(
        artifact_id=_artifact_id(
            directory,
            stage_name=stage_name,
            schema_version=schema_version,
            metadata=safe_metadata,
        ),
        stage_name=stage_name,
        schema_version=schema_version,
        files=files,
        metadata=safe_metadata,
        sha256=hashes,
    )
    try:
        payload = json.dumps(
            manifest.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactVerificationError("artifact metadata is not finite JSON data") from exc
    manifest_path.write_text(payload + "\n", encoding="utf-8", newline="\n")
    return manifest


def verify_artifact(root: str | Path) -> ArtifactManifest:


    directory = Path(os.path.abspath(os.fspath(Path(root).expanduser())))
    if not directory.is_dir():
        raise ArtifactVerificationError(f"artifact directory does not exist: {directory}")
    root_metadata = directory.lstat()
    if stat.S_ISLNK(root_metadata.st_mode) or _is_reparse_point(root_metadata):
        raise ArtifactVerificationError("artifact root must be a plain directory")
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ArtifactVerificationError("artifact manifest is missing")
    manifest = _load_manifest(manifest_path)
    actual_files, actual_hashes = _payload_files(directory)
    if actual_files != manifest.files:
        raise ArtifactVerificationError("artifact file names or sizes differ from the manifest")
    if actual_hashes != manifest.sha256:
        raise ArtifactVerificationError("artifact SHA-256 content differs from the manifest")
    expected_id = _artifact_id(
        directory,
        stage_name=manifest.stage_name,
        schema_version=manifest.schema_version,
        metadata=manifest.metadata,
    )
    if manifest.artifact_id != expected_id:
        raise ArtifactVerificationError("artifact_id differs from its declared metadata")
    return manifest


__all__ = [
    "ArtifactManifest",
    "ArtifactVerificationError",
    "publish_artifact",
    "verify_artifact",
]
