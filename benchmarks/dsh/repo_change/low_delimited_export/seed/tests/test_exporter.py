from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from delimited_export import export_rows


class ExportRowsPublicTests(unittest.TestCase):
    def test_r1_pipe_name_selects_pipe_delimiter(self) -> None:
        result = export_rows(
            ["name", "city"],
            [{"name": "Ada", "city": "London"}],
            delimiter="pipe",
        )

        self.assertEqual(result, "name|city\nAda|London\n")

    def test_r1_unknown_name_is_rejected_before_rows_are_requested(self) -> None:
        requested = False

        def rows():
            nonlocal requested
            requested = True
            yield {"name": "Ada"}

        with self.assertRaises(ValueError):
            export_rows(["name"], rows(), delimiter="semicolon")

        self.assertFalse(requested)

    def test_r2_omitted_delimiter_keeps_comma_output(self) -> None:
        result = export_rows(
            ["name", "note"],
            [{"name": "Ada", "note": "alpha,beta"}],
        )

        self.assertEqual(result, 'name,note\nAda,"alpha,beta"\n')

    def test_r3_tab_output_retains_csv_and_unicode_behavior(self) -> None:
        result = export_rows(
            ["city", "note"],
            [{"city": "\u6771\u4eac", "note": 'north\t"gate"'}],
            delimiter="tab",
        )

        self.assertEqual(result, 'city\tnote\n\u6771\u4eac\t"north\t""gate"""\n')


if __name__ == "__main__":
    unittest.main()
