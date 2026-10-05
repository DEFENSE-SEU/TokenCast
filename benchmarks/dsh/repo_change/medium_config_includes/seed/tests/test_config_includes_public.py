from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from config_stack import IncludeCycleError, IncludeOutsideRootError, load_config


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
COMPOSED = FIXTURES / "composed"


class PublicConfigIncludeTests(unittest.TestCase):
    def test_R1_nested_include_is_relative_to_its_declaring_file(self) -> None:
        config = load_config(COMPOSED / "app.json", root=COMPOSED)

        self.assertEqual(config["selected_shared"], "environment")

    def test_R2_includes_deep_merge_from_left_to_right(self) -> None:
        config = load_config(COMPOSED / "app.json", root=COMPOSED)

        self.assertEqual(
            config["service"],
            {
                "host": "app.internal",
                "port": 9000,
                "tls": {"ca": "environment.pem", "enabled": False},
            },
        )
        self.assertEqual(config["features"], {"audit": False, "cache": False})

    def test_R3_local_values_apply_after_all_includes(self) -> None:
        config = load_config(COMPOSED / "app.json", root=COMPOSED)

        self.assertEqual(config["items"], [])
        self.assertIsNone(config["optional"])
        self.assertEqual(config["retries"], 0)

    def test_R4_cycle_chain_is_root_relative_and_closed(self) -> None:
        cycle_root = FIXTURES / "cycle"

        with self.assertRaises(IncludeCycleError) as caught:
            load_config(cycle_root / "a.json", root=cycle_root)

        self.assertEqual(caught.exception.chain, ("a.json", "sub/b.json", "a.json"))

    def test_R5_parent_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "root"
            root.mkdir()
            (base / "outside.json").write_text("{}", encoding="utf-8")
            (root / "entry.json").write_text('{"$include":["../outside.json"]}', encoding="utf-8")

            with self.assertRaises(IncludeOutsideRootError):
                load_config(root / "entry.json", root=root)


if __name__ == "__main__":
    unittest.main()
