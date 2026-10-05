from __future__ import annotations

import unittest

from delimited_export import export_rows


class ExportRowsOracleTests(unittest.TestCase):
    def test_R1_each_exact_name_selects_its_character(self) -> None:
        fieldnames = ["left", "right"]
        rows = [{"left": "a", "right": "b"}]
        expected = {
            "comma": "left,right\na,b\n",
            "tab": "left\tright\na\tb\n",
            "pipe": "left|right\na|b\n",
        }

        for name, text in expected.items():
            with self.subTest(name=name):
                self.assertEqual(export_rows(fieldnames, rows, delimiter=name), text)

    def test_R1_unknown_values_raise_value_error_without_consuming_rows(self) -> None:
        class GuardedRows:
            def __init__(self) -> None:
                self.requested = False

            def __iter__(self):
                self.requested = True
                raise AssertionError("rows were requested before delimiter rejection")

        unknown_values = ("", "Comma", ",", "semicolon", None, 7, ["comma"])
        for value in unknown_values:
            with self.subTest(value=value):
                rows = GuardedRows()
                with self.assertRaises(ValueError):
                    export_rows(["value"], rows, delimiter=value)
                self.assertFalse(rows.requested)

    def test_R1_delimiter_is_keyword_only(self) -> None:
        with self.assertRaises(TypeError):
            export_rows(["value"], [{"value": "x"}], "comma")

    def test_R2_default_matches_explicit_comma_byte_for_byte(self) -> None:
        fieldnames = ["name", "note"]
        rows = [
            {"name": "Ada", "note": "alpha,beta"},
            {"name": "Grace", "note": "plain"},
        ]
        expected = 'name,note\nAda,"alpha,beta"\nGrace,plain\n'

        self.assertEqual(export_rows(fieldnames, rows), expected)
        self.assertEqual(export_rows(fieldnames, rows, delimiter="comma"), expected)

    def test_R2_empty_export_still_contains_comma_header_and_lf(self) -> None:
        self.assertEqual(export_rows(["beta", "alpha"], []), "beta,alpha\n")

    def test_R3_preserves_field_and_generator_row_order(self) -> None:
        requested: list[str] = []

        def rows():
            requested.append("first")
            yield {"zeta": "one", "alpha": "\u6771\u4eac"}
            requested.append("second")
            yield {"zeta": "two", "alpha": "Straße"}

        result = export_rows(["zeta", "alpha"], rows(), delimiter="pipe")

        self.assertEqual(result, "zeta|alpha\none|\u6771\u4eac\ntwo|Straße\n")
        self.assertEqual(requested, ["first", "second"])

    def test_R3_uses_minimal_dialect_specific_quoting(self) -> None:
        pipe_result = export_rows(
            ["kind", "memo"],
            [
                {"kind": "comma", "memo": "left,right"},
                {"kind": "pipe", "memo": 'left|"right"'},
            ],
            delimiter="pipe",
        )
        comma_result = export_rows(
            ["kind", "memo"],
            [{"kind": "pipe", "memo": "left|right"}],
            delimiter="comma",
        )

        self.assertEqual(
            pipe_result,
            'kind|memo\ncomma|left,right\npipe|"left|""right"""\n',
        )
        self.assertEqual(comma_result, "kind,memo\npipe,left|right\n")

    def test_R3_quotes_headers_and_doubles_embedded_quotes(self) -> None:
        result = export_rows(
            ["plain", "a|b"],
            [{"plain": 'say "yes"', "a|b": 'x|"y"'}],
            delimiter="pipe",
        )

        self.assertEqual(result, 'plain|"a|b"\n"say ""yes"""|"x|""y"""\n')

    def test_R3_preserves_lone_cr_and_quotes_newline_bearing_headers(self) -> None:
        result = export_rows(
            ['he"ad', "carriage\rreturn", "line\nfeed"],
            [
                {
                    'he"ad': "top\rbottom",
                    "carriage\rreturn": "plain",
                    "line\nfeed": "\u96ea",
                }
            ],
        )

        self.assertEqual(
            result,
            '"he""ad","carriage\rreturn","line\nfeed"\n"top\rbottom",plain,\u96ea\n',
        )

    def test_R3_preserves_unicode_and_embedded_newlines_with_lf_records(self) -> None:
        result = export_rows(
            ["city", "note"],
            [
                {"city": "\u96ea", "note": "line one\nline two"},
                {"city": "München", "note": "\u7d42"},
            ],
            delimiter="tab",
        )

        self.assertEqual(
            result,
            'city\tnote\n\u96ea\t"line one\nline two"\nMünchen\t\u7d42\n',
        )
        self.assertNotIn("\r\n", result)

    def test_R3_preserves_embedded_crlf_inside_a_quoted_field(self) -> None:
        result = export_rows(
            ["city", "note"],
            [{"city": "\u6771\u4eac", "note": "top\r\nbottom"}],
        )

        self.assertEqual(result, 'city,note\n\u6771\u4eac,"top\r\nbottom"\n')


if __name__ == "__main__":
    unittest.main()
