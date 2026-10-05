from __future__ import annotations

import array
import json
import tarfile
from pathlib import Path


def extract_jsonl(archive_path: Path, out_path: Path) -> None:
    if out_path.exists():
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".partial")
    with tarfile.open(archive_path, mode="r|gz") as tar, open(tmp_path, "wb") as out:
        for member in tar:
            if member.isfile() and member.name.endswith(".jsonl"):
                source = tar.extractfile(member)
                assert source is not None
                while True:
                    chunk = source.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                break
        else:
            raise FileNotFoundError(f"no .jsonl member found in {archive_path}")
    tmp_path.rename(out_path)


def build_offset_index(jsonl_path: Path, index_path: Path) -> None:
    if index_path.exists():
        return
    offsets = array.array("Q")
    tmp_path = index_path.with_suffix(index_path.suffix + ".partial")
    with open(jsonl_path, "rb") as handle:
        offset = handle.tell()
        for line in handle:
            if line.strip():
                offsets.append(offset)
            offset = handle.tell()
    # Publish the index only after every offset has been written.
    with open(tmp_path, "wb") as out:
        offsets.tofile(out)
    tmp_path.rename(index_path)


class IndexedCorpus:


    def __init__(self, jsonl_path: Path, index_path: Path):
        self._offsets = array.array("Q")
        with open(index_path, "rb") as handle:
            self._offsets.frombytes(handle.read())
        self._handle = open(jsonl_path, "rb")

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, idx: int) -> dict:
        self._handle.seek(self._offsets[int(idx)])
        line = self._handle.readline()
        return json.loads(line)


def build(archive_path: Path, work_dir: Path) -> IndexedCorpus:
    jsonl_path = work_dir / "wiki-18.jsonl"
    index_path = work_dir / "wiki-18.offsets"
    extract_jsonl(archive_path, jsonl_path)
    build_offset_index(jsonl_path, index_path)
    return IndexedCorpus(jsonl_path, index_path)
