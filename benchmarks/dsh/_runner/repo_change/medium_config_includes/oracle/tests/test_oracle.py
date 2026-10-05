from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any

from config_stack import (
    ConfigFormatError,
    IncludeCycleError,
    IncludeOutsideRootError,
    load_config,
)


class ConfigIncludeOracleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "configuration"
        self.root.mkdir()

    def write(self, relative: str, value: Any, *, root: Path | None = None) -> Path:
        directory = self.root if root is None else root
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return path

    def test_r1_nested_paths_follow_each_including_directory(self) -> None:
        self.write("shared.json", {"selected": "root"})
        self.write("teams/shared.json", {"selected": "team", "unicode": "\u914d\u7f6e"})
        self.write(
            "teams/feature.json",
            {"$include": ["shared.json"], "feature": {"active": True}},
        )
        entry = self.write("app.json", {"$include": ["teams/feature.json"]})

        self.assertEqual(
            load_config(entry, root=self.root),
            {"selected": "team", "unicode": "\u914d\u7f6e", "feature": {"active": True}},
        )

    def test_r1_document_and_include_shapes_are_validated(self) -> None:
        cases = (
            ("array.json", []),
            ("string-include.json", {"$include": "base.json"}),
            ("mixed-include.json", {"$include": ["base.json", 1]}),
            ("null-include.json", {"$include": None}),
        )
        for name, value in cases:
            with self.subTest(name=name):
                path = self.write(name, value)
                with self.assertRaises(ConfigFormatError):
                    load_config(path, root=self.root)

    def test_r2_recursive_objects_merge_in_include_order(self) -> None:
        self.write(
            "first.json",
            {
                "db": {
                    "host": "first",
                    "limits": {"burst": 3, "steady": 1},
                    "tags": ["first"],
                },
                "source": "first",
            },
        )
        self.write(
            "second.json",
            {
                "db": {
                    "limits": {"steady": 7},
                    "port": 5432,
                    "tags": ["second"],
                },
                "source": "second",
            },
        )
        entry = self.write("entry.json", {"$include": ["first.json", "second.json"]})

        self.assertEqual(
            load_config(entry, root=self.root),
            {
                "db": {
                    "host": "first",
                    "limits": {"burst": 3, "steady": 7},
                    "port": 5432,
                    "tags": ["second"],
                },
                "source": "second",
            },
        )

    def test_r2_non_object_values_replace_whole_values(self) -> None:
        self.write("one.json", {"a": {"nested": 1}, "b": 2, "c": [1], "d": None})
        self.write("two.json", {"a": 0, "b": {"nested": 2}, "c": {}, "d": [False]})
        entry = self.write("entry.json", {"$include": ["one.json", "two.json"]})

        self.assertEqual(
            load_config(entry, root=self.root),
            {"a": 0, "b": {"nested": 2}, "c": {}, "d": [False]},
        )

    def test_r3_local_members_merge_last_with_falsy_values(self) -> None:
        self.write(
            "defaults.json",
            {
                "enabled": True,
                "label": "default",
                "count": 8,
                "items": [1],
                "optional": "set",
                "nested": {"kept": 1, "changed": "old"},
            },
        )
        entry = self.write(
            "entry.json",
            {
                "$include": ["defaults.json"],
                "enabled": False,
                "label": "",
                "count": 0,
                "items": [],
                "optional": None,
                "nested": {"changed": "", "empty": {}},
            },
        )

        self.assertEqual(
            load_config(entry, root=self.root),
            {
                "enabled": False,
                "label": "",
                "count": 0,
                "items": [],
                "optional": None,
                "nested": {"kept": 1, "changed": "", "empty": {}},
            },
        )

    def test_r3_only_top_level_include_is_control_metadata(self) -> None:
        entry = self.write(
            "entry.json",
            {"$include": [], "nested": {"$include": ["literal.json"]}},
        )

        self.assertEqual(
            load_config(entry, root=self.root),
            {"nested": {"$include": ["literal.json"]}},
        )

    def test_r4_complete_cycle_chain_uses_root_relative_posix_paths(self) -> None:
        entry = self.write("entry.json", {"$include": ["prefix.json"]})
        self.write("prefix.json", {"$include": ["cycle/a.json"]})
        self.write("cycle/a.json", {"$include": ["nested/b.json"]})
        self.write("cycle/nested/b.json", {"$include": ["c.json"]})
        self.write("cycle/nested/c.json", {"$include": ["../a.json"]})

        with self.assertRaises(IncludeCycleError) as caught:
            load_config(entry, root=self.root)

        self.assertIs(type(caught.exception.chain), tuple)
        self.assertEqual(
            caught.exception.chain,
            (
                "cycle/a.json",
                "cycle/nested/b.json",
                "cycle/nested/c.json",
                "cycle/a.json",
            ),
        )

    def test_r4_diamond_graph_is_valid_after_branch_completion(self) -> None:
        self.write("shared.json", {"shared": {"present": True}, "winner": "shared"})
        self.write(
            "left.json",
            {"$include": ["shared.json"], "left": True, "winner": "left"},
        )
        self.write(
            "right.json",
            {"$include": ["shared.json"], "right": True, "winner": "right"},
        )
        entry = self.write("entry.json", {"$include": ["left.json", "right.json"]})

        self.assertEqual(
            load_config(entry, root=self.root),
            {
                "shared": {"present": True},
                "left": True,
                "right": True,
                "winner": "right",
            },
        )

    def test_r5_entry_and_traversal_targets_stay_within_root(self) -> None:
        outside = self.write("outside.json", {"secret": True}, root=self.base)
        entry = self.write("escape.json", {"$include": ["../outside.json"]})

        with self.assertRaises(IncludeOutsideRootError):
            load_config(outside, root=self.root)
        with self.assertRaises(IncludeOutsideRootError):
            load_config(entry, root=self.root)

    def test_r5_outside_targets_are_rejected_before_any_read(self) -> None:
        outside_entry = self.base / "missing-entry.json"
        entry = self.write(
            "missing-escape.json",
            {"$include": ["../missing-include.json"]},
        )

        with self.assertRaises(IncludeOutsideRootError):
            load_config(outside_entry, root=self.root)
        with self.assertRaises(IncludeOutsideRootError):
            load_config(entry, root=self.root)

    def test_r5_relative_arguments_and_default_root_follow_cwd_and_entry_parent(self) -> None:
        self.write("shared.json", {"inside": True})
        valid_entry = self.write("valid-relative.json", {"$include": ["shared.json"]})
        self.write("outside-default.json", {"outside": True}, root=self.base)
        escaping_entry = self.write(
            "escaping-relative.json",
            {"$include": ["../outside-default.json"]},
        )
        previous_cwd = Path.cwd()
        os.chdir(self.base)
        self.addCleanup(os.chdir, previous_cwd)

        self.assertEqual(
            load_config(
                valid_entry.relative_to(self.base),
                root=self.root.relative_to(self.base),
            ),
            {"inside": True},
        )
        self.assertEqual(
            load_config(valid_entry.relative_to(self.base)),
            {"inside": True},
        )
        with self.assertRaises(IncludeOutsideRootError):
            load_config(escaping_entry.relative_to(self.base))

    def test_r5_sibling_prefix_and_absolute_includes_are_rejected(self) -> None:
        sibling = self.base / "configuration-copy"
        sibling.mkdir()
        sibling_file = self.write("secret.json", {"secret": True}, root=sibling)
        prefix_entry = self.write(
            "prefix.json", {"$include": ["../configuration-copy/secret.json"]}
        )
        inside = self.write("inside.json", {"inside": True})
        absolute_entry = self.write("absolute.json", {"$include": [str(inside)]})

        with self.assertRaises(IncludeOutsideRootError):
            load_config(prefix_entry, root=self.root)
        with self.assertRaises(IncludeOutsideRootError):
            load_config(absolute_entry, root=self.root)
        self.assertTrue(sibling_file.is_file())

    def test_r5_symbolic_link_escape_is_rejected_when_supported(self) -> None:
        outside = self.write("outside-link-target.json", {"secret": True}, root=self.base)
        link = self.root / "linked.json"
        try:
            link.symlink_to(outside)
        except (NotImplementedError, OSError) as exc:
            self.fail(f"symbolic links unavailable: {exc}")
        entry = self.write("entry.json", {"$include": ["linked.json"]})

        with self.assertRaises(IncludeOutsideRootError):
            load_config(link, root=self.root)
        with self.assertRaises(IncludeOutsideRootError):
            load_config(entry, root=self.root)


if __name__ == "__main__":
    unittest.main()
