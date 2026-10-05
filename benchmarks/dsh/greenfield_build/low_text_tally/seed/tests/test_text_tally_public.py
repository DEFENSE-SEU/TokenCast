from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from text_tally import count_terms, tally_utf8


WORKSPACE = Path(__file__).resolve().parents[1]


class TextTallyPublicTests(unittest.TestCase):
    def test_r1_tokenizes_maximal_unicode_alphanumeric_runs(self) -> None:
        self.assertEqual(
            count_terms("rock'n-roll foo_bar １２3 tōkyō"),
            {"bar": 1, "foo": 1, "n": 1, "rock": 1, "roll": 1, "tōkyō": 1, "１２3": 1},
        )

    def test_r2_counts_unicode_casefold_equivalents(self) -> None:
        self.assertEqual(
            count_terms("Straße STRASSE Σ ς ﬃ FFI"),
            {"ffi": 2, "strasse": 2, "σ": 2},
        )

    def test_r3_library_returns_exact_canonical_bytes(self) -> None:
        self.assertEqual(
            tally_utf8("β B b".encode()),
            b'{"counts":{"b":2,"\xce\xb2":1},"total_terms":3,"unique_terms":2,"version":1}\n',
        )

    def test_r3_cli_matches_library_bytes(self) -> None:
        fixture = WORKSPACE / "fixtures" / "mixed.txt"
        result = subprocess.run(
            [sys.executable, "-B", "-m", "text_tally", str(fixture)],
            cwd=WORKSPACE,
            check=False,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, tally_utf8(fixture.read_bytes()))
        self.assertEqual(result.stderr, b"")


if __name__ == "__main__":
    unittest.main()
