from __future__ import annotations

import unittest

from catalog_lookup import Catalog, DuplicateItemError


PYTHON_312_STRIP_CHARACTERS = (
    "\x09\x0a\x0b\x0c\x0d"
    "\x1c\x1d\x1e\x1f\x20\x85\xa0\u1680"
    "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
    "\u2028\u2029\u202f\u205f\u3000"
)


class CatalogOracleTests(unittest.TestCase):
    def test_R1_python_strip_whitespace_is_ignored(self) -> None:
        catalog = Catalog([("delta", (4, 8))])

        self.assertEqual(catalog.lookup("\r\n\tdelta\v\f "), (4, 8))

    def test_R1_all_python_312_strip_characters_are_trimmed(self) -> None:
        for character in PYTHON_312_STRIP_CHARACTERS:
            with self.subTest(code_point=f"U+{ord(character):04X}"):
                value = object()
                catalog = Catalog([(f"{character}Comet{character}", value)])

                self.assertIs(catalog.lookup("comet"), value)
                self.assertIs(
                    catalog.lookup(f"{character}COMET{character}"),
                    value,
                )

    def test_R1_missing_lookup_retains_key_error(self) -> None:
        catalog = Catalog([("present", 1)])

        with self.assertRaises(KeyError) as caught:
            catalog.lookup(" \u2003ABSENT\t")

        self.assertEqual(caught.exception.args, ("absent",))

    def test_R2_casefold_matches_final_sigma(self) -> None:
        catalog = Catalog([("Σ", "sigma")])

        self.assertEqual(catalog.lookup("ς"), "sigma")

    def test_R2_casefold_expands_a_compatibility_ligature(self) -> None:
        catalog = Catalog([("ﬃ", "ligature")])

        self.assertEqual(catalog.lookup("ffi"), "ligature")

    def test_R3_casefold_collision_reports_both_raw_keys(self) -> None:
        with self.assertRaises(DuplicateItemError) as caught:
            Catalog([("Straße", 1), ("STRASSE", 2)])

        error = caught.exception
        self.assertEqual(error.canonical_key, "strasse")
        self.assertEqual(error.first_key, "Straße")
        self.assertEqual(error.duplicate_key, "STRASSE")

    def test_R3_strip_collision_is_rejected(self) -> None:
        with self.assertRaises(DuplicateItemError):
            Catalog([("alpha", "first"), ("  alpha\t", "second")])

    def test_R3_identical_raw_keys_are_a_collision(self) -> None:
        with self.assertRaises(DuplicateItemError) as caught:
            Catalog([("same", 1), ("same", 2)])

        self.assertEqual(caught.exception.canonical_key, "same")
        self.assertEqual(caught.exception.first_key, "same")
        self.assertEqual(caught.exception.duplicate_key, "same")

    def test_R3_collision_stops_iteration_at_the_duplicate(self) -> None:
        items_requested: list[str] = []

        def items():
            items_requested.append("first")
            yield "Alpha", 1
            items_requested.append("unrelated")
            yield "beta", 2
            items_requested.append("duplicate")
            yield "\u2003ALPHA\u2003", 3
            items_requested.append("after collision")
            self.fail("constructor requested an item after the first collision")

        with self.assertRaises(DuplicateItemError) as caught:
            Catalog(items())

        self.assertEqual(items_requested, ["first", "unrelated", "duplicate"])
        error = caught.exception
        self.assertEqual(error.canonical_key, "alpha")
        self.assertEqual(error.first_key, "Alpha")
        self.assertEqual(error.duplicate_key, "\u2003ALPHA\u2003")

    def test_R3_falsey_first_value_still_collides_immediately(self) -> None:
        items_requested: list[str] = []

        def items():
            items_requested.append("first")
            yield "Zero", 0
            items_requested.append("unrelated")
            yield "one", 1
            items_requested.append("duplicate")
            yield "ZERO", 2
            items_requested.append("after collision")
            self.fail("constructor requested an item after the falsey-value collision")

        with self.assertRaises(DuplicateItemError) as caught:
            Catalog(items())

        self.assertEqual(items_requested, ["first", "unrelated", "duplicate"])
        error = caught.exception
        self.assertEqual(error.canonical_key, "zero")
        self.assertEqual(error.first_key, "Zero")
        self.assertEqual(error.duplicate_key, "ZERO")


if __name__ == "__main__":
    unittest.main()
