from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from catalog_lookup import Catalog, DuplicateItemError


class CatalogPublicTests(unittest.TestCase):
    def test_r1_lookup_trims_surrounding_query_whitespace(self) -> None:
        catalog = Catalog([("north star", {"rank": 1})])

        self.assertEqual(catalog.lookup(" \t north star\n"), {"rank": 1})

    def test_r2_lookup_uses_unicode_casefold(self) -> None:
        catalog = Catalog([("Straße", "street")])

        self.assertEqual(catalog.lookup("STRASSE"), "street")

    def test_r3_constructor_rejects_a_canonical_collision(self) -> None:
        with self.assertRaises(DuplicateItemError):
            Catalog([("Alpha", 1), ("ALPHA", 2)])


if __name__ == "__main__":
    unittest.main()
