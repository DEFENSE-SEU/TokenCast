from __future__ import annotations

import argparse
import json
import stat
import sys
import zipfile
from dataclasses import asdict, dataclass
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Sequence


EXPECTED_DISTRIBUTION = "token-prediction"
EXPECTED_IMPORT_PACKAGE = "token_prediction"
EXPECTED_CONSOLE_ENTRY = "tc = token_prediction.cli:main"
FORBIDDEN_BASENAMES = frozenset(
    {
        ".env",
        "auth.json",
        "auth-profiles.json",
        "config.toml",
        "credentials.json",
        "secrets.json",
    }
)
FORBIDDEN_SUFFIXES = (".key", ".pem", ".pyc", ".sqlite", ".sqlite3")


class DistributionCheckError(ValueError):
    pass


@dataclass(frozen=True)
class DistributionReport:
    wheel: str
    distribution: str
    version: str
    requires_python: str
    file_count: int
    package_file_count: int
    record_entries: int


def _safe_member_name(value: str) -> str:
    if not value or "\\" in value or "\x00" in value:
        raise DistributionCheckError("wheel contains an unsafe member name")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise DistributionCheckError("wheel contains an unsafe member name")
    if ":" in path.parts[0]:
        raise DistributionCheckError("wheel contains an unsafe member name")
    return path.as_posix()


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    return stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK


def _record_entry_count(payload: bytes, *, label: str) -> int:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise DistributionCheckError(f"{label} is not UTF-8") from exc
    return sum(bool(line.strip()) for line in lines)


def inspect_wheel(path: str | Path) -> DistributionReport:
    wheel = Path(path)
    if not wheel.is_file() or wheel.suffix != ".whl":
        raise DistributionCheckError("distribution input must be one wheel file")
    try:
        archive = zipfile.ZipFile(wheel)
    except (OSError, zipfile.BadZipFile) as exc:
        raise DistributionCheckError("distribution input is not a readable wheel") from exc

    with archive:
        infos = archive.infolist()
        names = [_safe_member_name(info.filename) for info in infos]
        if len(names) != len(set(names)):
            raise DistributionCheckError("wheel contains duplicate archive members")
        if any(_is_symlink(info) for info in infos):
            raise DistributionCheckError("wheel must not contain symbolic links")
        file_names = {name for name, info in zip(names, infos) if not info.is_dir()}
        if not file_names:
            raise DistributionCheckError("wheel is empty")

        dist_info_roots = {
            name.split("/", 1)[0]
            for name in file_names
            if name.split("/", 1)[0].endswith(".dist-info")
        }
        if len(dist_info_roots) != 1:
            raise DistributionCheckError("wheel must contain exactly one .dist-info directory")
        dist_info = next(iter(dist_info_roots))
        allowed_roots = {EXPECTED_IMPORT_PACKAGE, dist_info}
        unexpected_roots = sorted({name.split("/", 1)[0] for name in file_names} - allowed_roots)
        if unexpected_roots:
            raise DistributionCheckError("wheel contains repository-only top-level material")

        for name in sorted(file_names):
            member = PurePosixPath(name)
            lowered_parts = tuple(part.casefold() for part in member.parts)
            basename = lowered_parts[-1]
            if basename in FORBIDDEN_BASENAMES or basename.endswith(FORBIDDEN_SUFFIXES):
                raise DistributionCheckError(
                    "wheel contains a forbidden sensitive or generated file"
                )
            if "tests" in lowered_parts or "workspace" in lowered_parts:
                raise DistributionCheckError("wheel contains tests or local workspace material")

        required = {
            f"{EXPECTED_IMPORT_PACKAGE}/__init__.py",
            f"{dist_info}/METADATA",
            f"{dist_info}/WHEEL",
            f"{dist_info}/RECORD",
            f"{dist_info}/entry_points.txt",
        }
        if not required.issubset(file_names):
            raise DistributionCheckError("wheel is missing required package metadata")

        metadata_name = f"{dist_info}/METADATA"
        try:
            metadata = BytesParser().parsebytes(archive.read(metadata_name))
        except (UnicodeDecodeError, ValueError) as exc:
            raise DistributionCheckError("wheel METADATA is invalid") from exc
        distribution = str(metadata.get("Name") or "")
        version = str(metadata.get("Version") or "")
        requires_python = str(metadata.get("Requires-Python") or "")
        if distribution.casefold().replace("_", "-") != EXPECTED_DISTRIBUTION:
            raise DistributionCheckError("wheel distribution name is unexpected")
        if not version:
            raise DistributionCheckError("wheel version is missing")
        if requires_python != ">=3.11":
            raise DistributionCheckError("wheel must declare Requires-Python: >=3.11")

        entry_points = archive.read(f"{dist_info}/entry_points.txt").decode(
            "utf-8", errors="strict"
        )
        normalized_entries = {line.strip() for line in entry_points.splitlines()}
        if "[console_scripts]" not in normalized_entries:
            raise DistributionCheckError("wheel has no console_scripts entry-point group")
        if EXPECTED_CONSOLE_ENTRY not in normalized_entries:
            raise DistributionCheckError("wheel does not expose the expected tc command")

        record_name = f"{dist_info}/RECORD"
        record_entries = _record_entry_count(archive.read(record_name), label=record_name)
        package_file_count = sum(
            name.startswith(f"{EXPECTED_IMPORT_PACKAGE}/") for name in file_names
        )
        if package_file_count < 2:
            raise DistributionCheckError("wheel package payload is unexpectedly small")

    return DistributionReport(
        wheel=wheel.name,
        distribution=distribution,
        version=version,
        requires_python=requires_python,
        file_count=len(file_names),
        package_file_count=package_file_count,
        record_entries=record_entries,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail closed if a built wheel is incomplete or leaks repository-only files."
    )
    parser.add_argument("wheel", type=Path, help="path to the wheel produced by python -m build")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = inspect_wheel(args.wheel)
    except DistributionCheckError as exc:
        print(f"distribution check failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(asdict(report), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
