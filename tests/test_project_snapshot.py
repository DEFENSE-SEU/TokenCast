import json
import subprocess
from pathlib import Path

import pytest

from token_prediction.collection import project as snapshots


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True).stdout


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    git(root, "init", "--quiet", "--template=")
    (root / ".gitignore").write_text(".env\ncache/\n")
    (root / "app.py").write_text("original\n")
    (root / "deleted.py").write_text("deleted later\n")
    git(root, "add", ".")
    (root / "app.py").write_text("uncommitted edit\n")
    (root / "deleted.py").unlink()
    (root / "new file.py").write_text("new source\n")
    (root / ".env").write_text("LOCAL_SECRET=example\n")
    (root / "cache").mkdir()
    (root / "cache/data").write_text("ignored\n")
    return root


def test_current_files_copied_independently_without_changing_source(source, tmp_path):
    index = (source / ".git/index").read_bytes()
    config = (source / ".git/config").read_bytes()
    result = snapshots.snapshot_project(source, "Fix the parser", tmp_path / "run")
    workspace = Path(result["workspace"])
    assert (workspace / "app.py").read_text() == "uncommitted edit\n"
    assert (workspace / "new file.py").read_text() == "new source\n"
    assert not (workspace / "deleted.py").exists()
    assert not (workspace / ".env").exists()
    assert not (workspace / "cache").exists()
    assert not (workspace / ".git/objects/info/alternates").exists()
    assert git(workspace, "remote") == b""
    assert git(workspace, "diff", result["baseline_tree"], "--") == b""
    (workspace / "app.py").write_text("agent change\n")
    assert b"agent change" in git(workspace, "diff", result["baseline_tree"], "--")
    assert (source / "app.py").read_text() == "uncommitted edit\n"
    assert (source / ".git/index").read_bytes() == index
    assert (source / ".git/config").read_bytes() == config
    manifest = json.loads((tmp_path / "run/snapshot.json").read_text())
    assert manifest["snapshot_id"] == result["snapshot_id"]
    second = snapshots.snapshot_project(source, "Fix the parser", tmp_path / "second")
    assert second["snapshot_id"] == result["snapshot_id"]


@pytest.mark.parametrize("case", ["nested", "existing", "subdirectory", "empty"])
def test_invalid_destinations_and_inputs_preserve_existing_files(source, tmp_path, case):
    destination = tmp_path / "output"
    project, task = source, "Fix it"
    if case == "nested":
        destination = source / "output"
    elif case == "existing":
        destination.mkdir()
        (destination / "keep").write_text("keep")
    elif case == "subdirectory":
        project = source / "cache"
    else:
        task = " "
    with pytest.raises(ValueError):
        snapshots.snapshot_project(project, task, destination)
    if case == "existing":
        assert (destination / "keep").read_text() == "keep"
    else:
        assert not destination.exists()


def test_linked_files_are_rejected_before_copy(source, tmp_path):
    link = source / "linked"
    try:
        link.symlink_to(source / "app.py")
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError, match="linked"):
        snapshots.snapshot_project(source, "Fix it", tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_source_edit_during_copy_never_produces_a_valid_snapshot(source, tmp_path, monkeypatch):
    original = snapshots.shutil.copy2

    def copy_and_edit(src, dst):
        result = original(src, dst)
        (source / "app.py").write_text("concurrent edit\n")
        return result

    monkeypatch.setattr(snapshots.shutil, "copy2", copy_and_edit)
    with pytest.raises(ValueError, match="changed while copying"):
        snapshots.snapshot_project(source, "Fix it", tmp_path / "run")
    assert not (tmp_path / "run/snapshot.json").exists()
