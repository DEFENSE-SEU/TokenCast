from __future__ import annotations

import hashlib
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "workspace/external/liveclawbench_c3cfa229/data"
EXPECTED_SHA256 = "580009458bb5691be0a41337a23dd4c66dc5595d5362e359be2430c6ef67126e"
EXPECTED_SIZE = 207135659
PIECES = [
    ("v0.2.1-00000-of-00001.parquet.partial", 60_000_000, 60_000_000),
    ("part_60000000_109999999.partial", 50_000_000, 50_000_000),
    ("part_110000000_159999999.partial", 50_000_000, 50_000_000),
    ("part_160000000_207135658.partial", 4_960_256, 4_960_256),
    ("tail_164960256_207135658.partial", 15_000_000, 15_000_000),
    ("range_179960256_189999999.partial", 10_039_744, 10_039_744),
    ("range_190000000_199999999.partial", 10_000_000, 10_000_000),
    ("range_200000000_207135658.partial", 7_135_659, 7_135_659),
]


def main() -> None:
    target = DIRECTORY / "v0.2.1-00000-of-00001.parquet"
    intermediate = DIRECTORY / "v0.2.1-00000-of-00001.parquet.assembling"
    if target.exists() or intermediate.exists():
        raise FileExistsError("Final or assembling source already exists")
    for name, minimum, maximum in PIECES:
        size = (DIRECTORY / name).stat().st_size
        is_prefix = name.startswith(("v0.2.1", "tail_"))
        if not (size >= minimum if is_prefix else minimum <= size <= maximum):
            raise ValueError(f"Unexpected piece length {name}: {size}")
    digest = hashlib.sha256()
    with intermediate.open("xb") as output:
        for name, length, _ in PIECES:
            with (DIRECTORY / name).open("rb") as source:
                remaining = length
                while remaining:
                    chunk = source.read(min(8 * 1024 * 1024, remaining))
                    if not chunk:
                        raise EOFError(f"Short piece: {name}")
                    output.write(chunk)
                    digest.update(chunk)
                    remaining -= len(chunk)
    if intermediate.stat().st_size != EXPECTED_SIZE or digest.hexdigest() != EXPECTED_SHA256:
        raise ValueError(f"Assembled source failed publisher checksum: {digest.hexdigest()}")
    os.replace(intermediate, target)
    print(f"{target} {target.stat().st_size} {digest.hexdigest()}")


if __name__ == "__main__":
    main()
