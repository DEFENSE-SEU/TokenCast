from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from text_tally import count_terms, tally_utf8


WORKSPACE = Path(__file__).resolve().parent.parent


def run_cli(path: Path, *, hash_seed: str = "0") -> subprocess.CompletedProcess[bytes]:
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = hash_seed
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-B", "-m", "text_tally", str(path)],
        cwd=WORKSPACE,
        env=environment,
        check=False,
        capture_output=True,
    )


class TextTallyOracleTests(unittest.TestCase):
    def test_r1_tokenizes_each_maximal_original_alphanumeric_run(self) -> None:
        text = "alpha-beta_gamma.delta\t１２3tōkyō🙂snow\r\nfall ratio½done"

        self.assertEqual(
            count_terms(text),
            {
                "alpha": 1,
                "beta": 1,
                "delta": 1,
                "fall": 1,
                "gamma": 1,
                "ratio½done": 1,
                "snow": 1,
                "１２3tōkyō": 1,
            },
        )

    def test_r1_tokenization_precedes_casefold(self) -> None:
        self.assertEqual(count_terms("İ"), {"i\u0307": 1})

    def test_r1_combining_marks_are_original_text_separators(self) -> None:
        self.assertEqual(count_terms("Cafe\u0301 NOËL"), {"cafe": 1, "noël": 1})

    def test_r1_tally_utf8_rejects_invalid_input_strictly(self) -> None:
        for data in (b"\xff", b"valid\xe2\x82", b"\x80text"):
            with self.subTest(data=data):
                with self.assertRaises(UnicodeDecodeError):
                    tally_utf8(data)

    def test_r1_cli_invalid_utf8_returns_two_without_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.txt"
            path.write_bytes(b"valid\xffinvalid")

            result = run_cli(path)

        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, b"")

    def test_r1_cli_input_io_failure_returns_one_without_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing.txt"

            result = run_cli(missing)

        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, b"")

    def test_r2_casefolds_expansions_and_contextual_forms(self) -> None:
        self.assertEqual(
            count_terms("Straße STRASSE Σ σ ς ﬃ FFI K k"),
            {"ffi": 2, "k": 2, "strasse": 2, "σ": 3},
        )

    def test_r2_casefold_collisions_accumulate_all_occurrences(self) -> None:
        self.assertEqual(
            count_terms("ẞ ß ss SS Straße strasse"),
            {"ss": 4, "strasse": 2},
        )

    def test_r2_casefold_does_not_add_unicode_normalization(self) -> None:
        self.assertEqual(count_terms("ǰ"), {"j\u030c": 1})

    def test_r2_count_dictionary_uses_sorted_canonical_keys(self) -> None:
        result = count_terms("ω Alpha β ALPHA")

        self.assertEqual(list(result), ["alpha", "β", "ω"])
        self.assertEqual(result, {"alpha": 2, "β": 1, "ω": 1})

    def test_r3_library_returns_exact_utf8_canonical_bytes(self) -> None:
        result = tally_utf8("ω Alpha β ALPHA".encode())

        self.assertIs(type(result), bytes)
        self.assertEqual(
            result,
            b'{"counts":{"alpha":2,"\xce\xb2":1,"\xcf\x89":1},'
            b'"total_terms":4,"unique_terms":3,"version":1}\n',
        )
        self.assertNotIn(b"\r\n", result)

    def test_r3_empty_input_has_one_complete_canonical_report(self) -> None:
        self.assertEqual(
            tally_utf8(b""),
            b'{"counts":{},"total_terms":0,"unique_terms":0,"version":1}\n',
        )

    def test_r3_cli_matches_library_bytes_across_runtime_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.txt"
            data = "ω Alpha β ALPHA Straße\n".encode()
            path.write_bytes(data)

            first = run_cli(path, hash_seed="1")
            second = run_cli(path, hash_seed="987654")

        expected = tally_utf8(data)
        for result in (first, second):
            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.stdout, expected)
            self.assertEqual(result.stderr, b"")
        self.assertEqual(first.stdout, second.stdout)
        self.assertTrue(first.stdout.endswith(b"\n"))
        self.assertNotIn(b"\r\n", first.stdout)


if __name__ == "__main__":
    unittest.main()
