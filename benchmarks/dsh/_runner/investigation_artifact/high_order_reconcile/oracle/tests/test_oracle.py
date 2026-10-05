from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from order_reconcile import EvidenceError, analyze, write_artifacts


CANDIDATE_ROOT = Path(__file__).resolve().parents[2]
PUBLIC = CANDIDATE_ROOT / "fixtures" / "public"
ALTERNATE = Path(__file__).parent / "fixtures" / "alternate"
SOURCES = (
    "orders/01.jsonl",
    "scans/01.jsonl",
    "scans/02.jsonl",
    "corrections/01.jsonl",
    "adjustments/01.jsonl",
    "adjustments/02.jsonl",
)


def _payload(**fields: object) -> bytes:
    return json.dumps(fields, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _write_fixture(root: Path, records: dict[str, list[tuple[bytes, bytes]]]) -> Path:
    for source in SOURCES:
        path = root.joinpath(*source.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"".join(payload + ending for payload, ending in records.get(source, [])))
    return root


def _base_records() -> dict[str, list[tuple[bytes, bytes]]]:
    order = _payload(
        event_time="2026-03-01T00:00:00Z",
        order_id="ORD-1",
        record_id="O1",
        sequence=1,
        type="order_created",
    )
    picked = _payload(
        event_time="2026-03-01T00:01:00Z",
        order_id="ORD-1",
        record_id="S-PICK",
        sequence=1,
        state="picked",
        type="scan",
    )
    return {
        "orders/01.jsonl": [(order, b"\n")],
        "scans/01.jsonl": [(picked, b"\n")],
    }


def _orders(analysis: object) -> dict[str, dict[str, object]]:
    report = getattr(analysis, "report")
    return {row["order_id"]: row for row in report["orders"]}


def _exceptions(analysis: object, code: str) -> list[dict[str, str]]:
    return [row for row in getattr(analysis, "exceptions") if row["code"] == code]


def _read_cited_record(root: Path, source: str, line: int) -> dict[str, object]:
    raw_line = root.joinpath(*source.split("/")).read_bytes().splitlines()[line - 1]
    return json.loads(raw_line.decode("utf-8"))


class OracleTests(unittest.TestCase):
    def test_R1_joins_all_record_families_in_logical_order(self) -> None:
        analysis = analyze(PUBLIC)
        orders = _orders(analysis)
        self.assertEqual(list(orders), ["ORD-100", "ORD-200", "ORD-300"])
        self.assertEqual(orders["ORD-100"]["final_state"], "delivered")
        self.assertEqual(orders["ORD-200"]["final_state"], "delivered")
        self.assertIn(
            {"line": 1, "source": "corrections/01.jsonl"},
            orders["ORD-200"]["citations"],
        )
        self.assertIn(
            {"line": 2, "source": "adjustments/02.jsonl"},
            orders["ORD-100"]["citations"],
        )

    def test_R1_segment_filename_order_does_not_control_events(self) -> None:
        analysis = analyze(ALTERNATE)
        orders = _orders(analysis)
        self.assertEqual(orders["ORD-X"]["final_state"], "delivered")
        self.assertEqual(orders["ORD-Y"]["final_state"], "picked")

    def test_R1_full_logical_key_orders_equal_time_events(self) -> None:
        records = {
            "orders/01.jsonl": [
                (
                    _payload(
                        event_time="2026-04-01T00:00:00Z",
                        order_id="ORD-TIE",
                        record_id="O-TIE",
                        sequence=1,
                        type="order_created",
                    ),
                    b"\n",
                )
            ],
            "scans/01.jsonl": [
                (
                    _payload(
                        event_time="2026-04-01T00:01:00Z",
                        order_id="ORD-TIE",
                        record_id="B-DELIVER",
                        sequence=3,
                        state="delivered",
                        type="scan",
                    ),
                    b"\n",
                ),
                (
                    _payload(
                        event_time="2026-04-01T00:01:00Z",
                        order_id="ORD-TIE",
                        record_id="Z-PACK",
                        sequence=2,
                        state="packed",
                        type="scan",
                    ),
                    b"\n",
                ),
            ],
            "scans/02.jsonl": [
                (
                    _payload(
                        event_time="2026-04-01T00:01:00Z",
                        order_id="ORD-TIE",
                        record_id="Z-PICK",
                        sequence=1,
                        state="picked",
                        type="scan",
                    ),
                    b"\n",
                ),
                (
                    _payload(
                        event_time="2026-04-01T00:01:00Z",
                        order_id="ORD-TIE",
                        record_id="A-SHIP",
                        sequence=3,
                        state="shipped",
                        type="scan",
                    ),
                    b"\n",
                ),
            ],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            analysis = analyze(_write_fixture(Path(temporary_directory), records))
        self.assertEqual(_orders(analysis)["ORD-TIE"]["final_state"], "delivered")
        self.assertFalse(
            {
                "missing_transition",
                "regressing_transition",
                "incomplete_lifecycle",
            }
            & {row["code"] for row in analysis.exceptions}
        )

    def test_R1_later_duplicate_order_preserves_prior_conclusions(self) -> None:
        records = {
            "orders/01.jsonl": [
                (
                    _payload(
                        event_time="2026-04-02T00:00:00Z",
                        order_id="ORD-DUP",
                        record_id="O-FIRST",
                        sequence=1,
                        type="order_created",
                    ),
                    b"\n",
                ),
                (
                    _payload(
                        event_time="2026-04-02T00:03:00Z",
                        order_id="ORD-DUP",
                        record_id="O-SECOND",
                        sequence=1,
                        type="order_created",
                    ),
                    b"\n",
                ),
            ],
            "scans/01.jsonl": [
                (
                    _payload(
                        event_time="2026-04-02T00:01:00Z",
                        order_id="ORD-DUP",
                        record_id="S-DUP-PICK",
                        sequence=1,
                        state="picked",
                        type="scan",
                    ),
                    b"\n",
                ),
                (
                    _payload(
                        event_time="2026-04-02T00:04:00Z",
                        order_id="ORD-DUP",
                        record_id="S-DUP-PACK",
                        sequence=1,
                        state="packed",
                        type="scan",
                    ),
                    b"\n",
                ),
            ],
            "adjustments/01.jsonl": [
                (
                    _payload(
                        amount="2",
                        event_time="2026-04-02T00:02:00Z",
                        order_id="ORD-DUP",
                        record_id="A-DUP",
                        sequence=1,
                        type="adjustment",
                    ),
                    b"\n",
                )
            ],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            analysis = analyze(_write_fixture(Path(temporary_directory), records))
        order = _orders(analysis)["ORD-DUP"]
        self.assertEqual(order["final_state"], "packed")
        self.assertEqual(order["adjustment_total"], "2")
        self.assertEqual(
            [row["record_id"] for row in _exceptions(analysis, "duplicate_order")],
            ["O-SECOND"],
        )

    def test_R2_LF_and_CRLF_duplicates_collapse_with_all_citations(self) -> None:
        records = _base_records()
        packed = _payload(
            event_time="2026-03-01T00:02:00Z",
            order_id="ORD-1",
            record_id="S-PACK",
            sequence=1,
            state="packed",
            type="scan",
        )
        shipped = _payload(
            event_time="2026-03-01T00:03:00Z",
            order_id="ORD-1",
            record_id="S-SHIP",
            sequence=1,
            state="shipped",
            type="scan",
        )
        delivered = _payload(
            event_time="2026-03-01T00:04:00Z",
            order_id="ORD-1",
            record_id="S-DELIVER",
            sequence=1,
            state="delivered",
            type="scan",
        )
        records["scans/01.jsonl"].append((packed, b"\n"))
        records["scans/02.jsonl"] = [
            (packed, b"\r\n"),
            (shipped, b"\n"),
            (delivered, b"\n"),
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = _write_fixture(Path(temporary_directory), records)
            analysis = analyze(root)
        order = _orders(analysis)["ORD-1"]
        self.assertEqual(order["final_state"], "delivered")
        self.assertIn({"line": 2, "source": "scans/01.jsonl"}, order["citations"])
        self.assertIn({"line": 1, "source": "scans/02.jsonl"}, order["citations"])
        duplicates = _exceptions(analysis, "identical_duplicate")
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(duplicates[0]["source"], "scans/02.jsonl")

    def test_R2_final_record_without_a_terminator_is_accepted(self) -> None:
        records = _base_records()
        records["scans/01.jsonl"][0] = (records["scans/01.jsonl"][0][0], b"")
        with tempfile.TemporaryDirectory() as temporary_directory:
            analysis = analyze(_write_fixture(Path(temporary_directory), records))
        self.assertEqual(_orders(analysis)["ORD-1"]["final_state"], "picked")

    def test_R2_adjustment_duplicate_retains_citations_and_last_change(self) -> None:
        records = _base_records()
        adjustment = _payload(
            amount="7",
            event_time="2026-03-01T00:02:00Z",
            order_id="ORD-1",
            record_id="A-DUPLICATE",
            sequence=1,
            type="adjustment",
        )
        records["adjustments/01.jsonl"] = [(adjustment, b"\n")]
        records["adjustments/02.jsonl"] = [(adjustment, b"")]
        with tempfile.TemporaryDirectory() as temporary_directory:
            analysis = analyze(_write_fixture(Path(temporary_directory), records))
        order = _orders(analysis)["ORD-1"]
        self.assertEqual(order["adjustment_total"], "7")
        self.assertIn({"line": 1, "source": "adjustments/01.jsonl"}, order["citations"])
        self.assertIn({"line": 1, "source": "adjustments/02.jsonl"}, order["citations"])
        self.assertEqual(
            [row["source"] for row in _exceptions(analysis, "identical_duplicate")],
            ["adjustments/02.jsonl"],
        )
        incomplete = _exceptions(analysis, "incomplete_lifecycle")
        self.assertEqual(
            (incomplete[0]["record_id"], incomplete[0]["source"], incomplete[0]["line"]),
            ("A-DUPLICATE", "adjustments/01.jsonl", "1"),
        )

    def test_R3_equal_JSON_with_different_payload_bytes_conflicts(self) -> None:
        records = _base_records()
        fields = {
            "event_time": "2026-03-01T00:02:00Z",
            "order_id": "ORD-1",
            "record_id": "S-CONFLICT",
            "sequence": 1,
            "state": "packed",
            "type": "scan",
        }
        compact = _payload(**fields)
        spaced = json.dumps(fields, sort_keys=True, separators=(", ", ": ")).encode()
        records["scans/01.jsonl"].append((compact, b"\n"))
        records["scans/02.jsonl"] = [(spaced, b"\n")]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = _write_fixture(Path(temporary_directory), records)
            analysis = analyze(root)
        conflicts = _exceptions(analysis, "conflicting_duplicate")
        self.assertEqual(len(conflicts), 2)
        self.assertEqual({row["source"] for row in conflicts}, {"scans/01.jsonl", "scans/02.jsonl"})
        citations = _orders(analysis)["ORD-1"]["citations"]
        self.assertNotIn({"line": 2, "source": "scans/01.jsonl"}, citations)
        self.assertNotIn({"line": 1, "source": "scans/02.jsonl"}, citations)

    def test_R3_identity_reconciliation_spans_every_record_kind(self) -> None:
        records = {
            "orders/01.jsonl": [
                (
                    _payload(
                        event_time="2026-04-03T00:00:00Z",
                        order_id="ORD-1",
                        record_id="O-MAIN",
                        sequence=1,
                        type="order_created",
                    ),
                    b"\n",
                ),
                (
                    _payload(
                        event_time="2026-04-03T00:00:01Z",
                        order_id="ORD-2",
                        record_id="ID-ORDER",
                        sequence=1,
                        type="order_created",
                    ),
                    b"\n",
                ),
            ],
            "scans/01.jsonl": [
                (
                    _payload(
                        event_time="2026-04-03T00:01:00Z",
                        order_id="ORD-1",
                        record_id="ID-ORDER",
                        sequence=1,
                        state="picked",
                        type="scan",
                    ),
                    b"\n",
                ),
                (
                    _payload(
                        event_time="2026-04-03T00:01:01Z",
                        order_id="ORD-1",
                        record_id="ID-REV",
                        sequence=1,
                        state="picked",
                        type="scan",
                    ),
                    b"\n",
                ),
            ],
            "corrections/01.jsonl": [
                (
                    _payload(
                        event_time="2026-04-03T00:01:02Z",
                        from_state="created",
                        order_id="ORD-1",
                        record_id="ID-CORR",
                        sequence=1,
                        to_state="picked",
                        type="correction",
                    ),
                    b"\n",
                )
            ],
            "adjustments/01.jsonl": [
                (
                    _payload(
                        amount="5",
                        event_time="2026-04-03T00:02:00Z",
                        order_id="ORD-1",
                        record_id="ID-CORR",
                        sequence=1,
                        type="adjustment",
                    ),
                    b"\n",
                ),
                (
                    _payload(
                        event_time="2026-04-03T00:02:01Z",
                        order_id="ORD-1",
                        record_id="ID-REV",
                        reversal_of="A-NONE",
                        sequence=1,
                        type="reversal",
                    ),
                    b"\n",
                ),
            ],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            analysis = analyze(_write_fixture(Path(temporary_directory), records))
        conflicts = _exceptions(analysis, "conflicting_duplicate")
        self.assertEqual(
            {
                record_id: sum(row["record_id"] == record_id for row in conflicts)
                for record_id in {"ID-ORDER", "ID-CORR", "ID-REV"}
            },
            {"ID-ORDER": 2, "ID-CORR": 2, "ID-REV": 2},
        )
        self.assertEqual(set(_orders(analysis)), {"ORD-1"})
        self.assertEqual(_orders(analysis)["ORD-1"]["adjustment_total"], "0")

    def test_R3_adjustment_payload_whitespace_creates_a_conflict(self) -> None:
        records = _base_records()
        fields = {
            "amount": "3",
            "event_time": "2026-04-03T00:02:00Z",
            "order_id": "ORD-1",
            "record_id": "A-CONFLICT",
            "sequence": 1,
            "type": "adjustment",
        }
        records["adjustments/01.jsonl"] = [(_payload(**fields), b"\n")]
        records["adjustments/02.jsonl"] = [
            (
                json.dumps(fields, sort_keys=True, separators=(", ", ": ")).encode(),
                b"\n",
            )
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            analysis = analyze(_write_fixture(Path(temporary_directory), records))
        self.assertEqual(_orders(analysis)["ORD-1"]["adjustment_total"], "0")
        self.assertEqual(
            [row["source"] for row in _exceptions(analysis, "conflicting_duplicate")],
            ["adjustments/01.jsonl", "adjustments/02.jsonl"],
        )

    def test_R4_corrections_and_reversal_relationships_apply_in_order(self) -> None:
        analysis = analyze(PUBLIC)
        orders = _orders(analysis)
        self.assertEqual(orders["ORD-100"]["adjustment_total"], "1.2")
        self.assertEqual(orders["ORD-200"]["adjustment_total"], "5")
        codes = {row["code"] for row in analysis.exceptions}
        self.assertTrue(
            {
                "correction_state_mismatch",
                "cross_order_reversal",
                "forward_reversal",
                "missing_reversal_target",
                "second_reversal",
            }.issubset(codes)
        )

    def test_R4_reversal_target_type_is_checked(self) -> None:
        records = _base_records()
        reversal = _payload(
            event_time="2026-03-01T00:02:00Z",
            order_id="ORD-1",
            record_id="R-ORDER",
            reversal_of="O1",
            sequence=1,
            type="reversal",
        )
        records["adjustments/01.jsonl"] = [(reversal, b"\n")]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = _write_fixture(Path(temporary_directory), records)
            analysis = analyze(root)
        self.assertEqual(len(_exceptions(analysis, "invalid_reversal_target")), 1)
        self.assertEqual(_orders(analysis)["ORD-1"]["adjustment_total"], "0")

    def test_R4_correction_requires_the_immediate_successor(self) -> None:
        records = _base_records()
        correction = _payload(
            event_time="2026-03-01T00:02:00Z",
            from_state="picked",
            order_id="ORD-1",
            record_id="C-SKIP",
            sequence=1,
            to_state="shipped",
            type="correction",
        )
        records["corrections/01.jsonl"] = [(correction, b"\n")]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = _write_fixture(Path(temporary_directory), records)
            analysis = analyze(root)
        self.assertEqual(len(_exceptions(analysis, "invalid_correction")), 1)
        self.assertEqual(_orders(analysis)["ORD-1"]["final_state"], "picked")

    def test_R4_reversal_exception_priority_is_frozen(self) -> None:
        def order(order_id: str, record_id: str) -> bytes:
            return _payload(
                event_time="2026-04-04T00:00:00Z",
                order_id=order_id,
                record_id=record_id,
                sequence=1,
                type="order_created",
            )

        def adjustment(order_id: str, record_id: str, amount: str, event_time: str) -> bytes:
            return _payload(
                amount=amount,
                event_time=event_time,
                order_id=order_id,
                record_id=record_id,
                sequence=1,
                type="adjustment",
            )

        def reversal(order_id: str, record_id: str, target: str, event_time: str) -> bytes:
            return _payload(
                event_time=event_time,
                order_id=order_id,
                record_id=record_id,
                reversal_of=target,
                sequence=1,
                type="reversal",
            )

        records = {
            "orders/01.jsonl": [(order("ORD-1", "O1"), b"\n"), (order("ORD-2", "O2"), b"\n")],
            "adjustments/01.jsonl": [
                (adjustment("ORD-1", "A-VALID", "2", "2026-04-04T00:01:00Z"), b"\n"),
                (adjustment("ORD-2", "A-CROSS", "3", "2026-04-04T00:01:01Z"), b"\n"),
                (reversal("ORD-1", "R-VALID", "A-VALID", "2026-04-04T00:02:00Z"), b"\n"),
                (reversal("ORD-1", "R-SECOND", "A-VALID", "2026-04-04T00:03:00Z"), b"\n"),
                (reversal("ORD-1", "R-CROSS", "A-CROSS", "2026-04-04T00:03:01Z"), b"\n"),
                (reversal("ORD-1", "R-MISSING", "A-NONE", "2026-04-04T00:03:02Z"), b"\n"),
            ],
            "adjustments/02.jsonl": [
                (reversal("ORD-1", "R-FORWARD", "A-FUTURE", "2026-04-04T00:04:00Z"), b"\n"),
                (reversal("ORD-1", "R-TYPE", "O1", "2026-04-04T00:04:01Z"), b"\n"),
                (adjustment("ORD-2", "A-FUTURE", "4", "2026-04-04T00:05:00Z"), b"\n"),
            ],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            analysis = analyze(_write_fixture(Path(temporary_directory), records))
        codes_by_record = {row["record_id"]: row["code"] for row in analysis.exceptions}
        self.assertEqual(codes_by_record["R-SECOND"], "second_reversal")
        self.assertEqual(codes_by_record["R-CROSS"], "cross_order_reversal")
        self.assertEqual(codes_by_record["R-MISSING"], "missing_reversal_target")
        self.assertEqual(codes_by_record["R-FORWARD"], "forward_reversal")
        self.assertEqual(codes_by_record["R-TYPE"], "invalid_reversal_target")
        self.assertEqual(_orders(analysis)["ORD-1"]["adjustment_total"], "0")
        self.assertEqual(_orders(analysis)["ORD-2"]["adjustment_total"], "7")

    def test_R5_missing_regressing_and_incomplete_lifecycle_are_reported(self) -> None:
        analysis = analyze(PUBLIC)
        codes = {row["code"] for row in analysis.exceptions}
        self.assertIn("missing_transition", codes)
        self.assertIn("regressing_transition", codes)
        incomplete = _exceptions(analysis, "incomplete_lifecycle")
        self.assertEqual(len(incomplete), 1)
        self.assertEqual(incomplete[0]["order_id"], "ORD-300")
        self.assertEqual(_orders(analysis)["ORD-300"]["final_state"], "picked")

    def test_R6_decimal_adjustment_and_reversal_remain_exact(self) -> None:
        analysis = analyze(ALTERNATE)
        orders = _orders(analysis)
        self.assertEqual(
            orders["ORD-X"]["adjustment_total"],
            "10000000000000000000.00000000000000000001",
        )
        self.assertEqual(orders["ORD-Y"]["adjustment_total"], "-0.1")
        self.assertNotIn("E", orders["ORD-X"]["adjustment_total"])

    def test_R6_signed_zero_is_canonical_zero(self) -> None:
        records = _base_records()
        records["adjustments/01.jsonl"] = [
            (
                _payload(
                    amount="-0.000000",
                    event_time="2026-03-01T00:02:00Z",
                    order_id="ORD-1",
                    record_id="A-ZERO",
                    sequence=1,
                    type="adjustment",
                ),
                b"\n",
            )
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            analysis = analyze(_write_fixture(Path(temporary_directory), records))
        self.assertEqual(_orders(analysis)["ORD-1"]["adjustment_total"], "0")

    def test_R7_every_conclusion_and_exception_has_a_physical_citation(self) -> None:
        analysis = analyze(PUBLIC)
        for order in analysis.report["orders"]:
            self.assertTrue(order["citations"])
            for citation in order["citations"]:
                record = _read_cited_record(PUBLIC, citation["source"], int(citation["line"]))
                self.assertEqual(record["order_id"], order["order_id"])
        for row in analysis.exceptions:
            record = _read_cited_record(PUBLIC, row["source"], int(row["line"]))
            self.assertEqual(record["record_id"], row["record_id"])
        self.assertEqual(
            [row["exception_id"] for row in analysis.exceptions],
            [f"X{index:04d}" for index in range(1, len(analysis.exceptions) + 1)],
        )

    def test_R7_exception_lines_sort_numerically(self) -> None:
        records = _base_records()
        records["adjustments/01.jsonl"] = [
            (
                _payload(
                    event_time=f"2026-03-01T00:02:{index:02d}Z",
                    order_id="ORD-1",
                    record_id=f"R-LINE-{index}",
                    reversal_of=f"A-NONE-{index}",
                    sequence=1,
                    type="reversal",
                ),
                b"\n",
            )
            for index in range(1, 11)
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            analysis = analyze(_write_fixture(Path(temporary_directory), records))
        adjustment_rows = [
            row for row in analysis.exceptions if row["source"] == "adjustments/01.jsonl"
        ]
        self.assertEqual([int(row["line"]) for row in adjustment_rows], list(range(1, 11)))

    def test_R8_API_CLI_and_submitted_artifacts_are_byte_deterministic(self) -> None:
        analysis = analyze(PUBLIC)
        with (
            tempfile.TemporaryDirectory() as first_directory,
            tempfile.TemporaryDirectory() as second_directory,
        ):
            first_report, first_exceptions = write_artifacts(analysis, first_directory)
            second_report, second_exceptions = write_artifacts(analyze(PUBLIC), second_directory)
            self.assertEqual(first_report.read_bytes(), second_report.read_bytes())
            self.assertEqual(first_exceptions.read_bytes(), second_exceptions.read_bytes())
            self.assertEqual(
                first_report.read_bytes(),
                (CANDIDATE_ROOT / "artifacts" / "report.json").read_bytes(),
            )
            self.assertEqual(
                first_exceptions.read_bytes(),
                (CANDIDATE_ROOT / "artifacts" / "exceptions.csv").read_bytes(),
            )

            cli_output = Path(second_directory) / "cli"
            result = subprocess.run(
                [sys.executable, "-m", "order_reconcile", str(PUBLIC), str(cli_output)],
                cwd=CANDIDATE_ROOT,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8"))
            self.assertEqual(result.stdout, b"")
            self.assertEqual((cli_output / "report.json").read_bytes(), first_report.read_bytes())
            self.assertEqual(
                (cli_output / "exceptions.csv").read_bytes(),
                first_exceptions.read_bytes(),
            )

    def test_R8_alternate_evidence_changes_canonical_artifacts(self) -> None:
        with (
            tempfile.TemporaryDirectory() as public_directory,
            tempfile.TemporaryDirectory() as alternate_directory,
        ):
            public_paths = write_artifacts(analyze(PUBLIC), public_directory)
            alternate_paths = write_artifacts(analyze(ALTERNATE), alternate_directory)
            self.assertNotEqual(public_paths[0].read_bytes(), alternate_paths[0].read_bytes())
            self.assertNotEqual(public_paths[1].read_bytes(), alternate_paths[1].read_bytes())
            rows = list(csv.DictReader(io.StringIO(alternate_paths[1].read_text(encoding="utf-8"))))
            self.assertTrue(rows)

    def test_R8_invalid_evidence_raises_the_public_error_type(self) -> None:
        records = _base_records()
        records["orders/01.jsonl"] = [(b"{}", b"\n")]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = _write_fixture(Path(temporary_directory), records)
            with self.assertRaises(EvidenceError):
                analyze(root)

    def test_R8_strict_schemas_boolean_sequence_and_large_integer_are_rejected(self) -> None:
        valid_order = {
            "event_time": "2026-05-01T00:00:00Z",
            "order_id": "ORD-1",
            "record_id": "O1",
            "sequence": 1,
            "type": "order_created",
        }
        exact_shape_cases = {
            "order": (
                "orders/01.jsonl",
                _payload(**valid_order, extra="x"),
            ),
            "scan": (
                "scans/01.jsonl",
                _payload(
                    **{
                        **valid_order,
                        "extra": "x",
                        "record_id": "S1",
                        "state": "picked",
                        "type": "scan",
                    }
                ),
            ),
            "correction": (
                "corrections/01.jsonl",
                _payload(
                    **{
                        **valid_order,
                        "extra": "x",
                        "from_state": "created",
                        "record_id": "C1",
                        "to_state": "picked",
                        "type": "correction",
                    }
                ),
            ),
            "adjustment": (
                "adjustments/01.jsonl",
                _payload(
                    **{
                        **valid_order,
                        "amount": "1",
                        "extra": "x",
                        "record_id": "A1",
                        "type": "adjustment",
                    }
                ),
            ),
            "reversal": (
                "adjustments/02.jsonl",
                _payload(
                    **{
                        **valid_order,
                        "extra": "x",
                        "record_id": "R1",
                        "reversal_of": "A1",
                        "type": "reversal",
                    }
                ),
            ),
        }
        for label, (source, payload) in exact_shape_cases.items():
            with tempfile.TemporaryDirectory() as directory:
                records = {"orders/01.jsonl": [(_payload(**valid_order), b"\n")]}
                records[source] = [(payload, b"\n")]
                with self.assertRaises(EvidenceError, msg=label):
                    analyze(_write_fixture(Path(directory), records))

        malformed_orders = (
            b'{"event_time":"2026-05-01T00:00:00Z","order_id":"ORD-1",'
            b'"record_id":"O1","record_id":"O2","sequence":1,"type":"order_created"}',
            b'{"event_time":"2026-05-01T00:00:00Z","order_id":"ORD-1",'
            b'"record_id":"O1","sequence":true,"type":"order_created"}',
            b'{"event_time":"2026-05-01T00:00:00Z","order_id":"ORD-1",'
            b'"record_id":"O1","sequence":' + b"1" * 4301 + b',"type":"order_created"}',
        )
        for index, payload in enumerate(malformed_orders):
            with tempfile.TemporaryDirectory() as directory:
                records = {"orders/01.jsonl": [(payload, b"\n")]}
                with self.assertRaises(EvidenceError, msg=f"malformed case {index}"):
                    analyze(_write_fixture(Path(directory), records))

    def test_R8_report_order_and_cli_error_classification_are_frozen(self) -> None:
        records = {
            "orders/01.jsonl": [
                (
                    _payload(
                        event_time="2026-05-02T00:00:00Z",
                        order_id="ORD-Z",
                        record_id="O-Z",
                        sequence=1,
                        type="order_created",
                    ),
                    b"\n",
                ),
                (
                    _payload(
                        event_time="2026-05-02T00:00:01Z",
                        order_id="ORD-A",
                        record_id="O-A",
                        sequence=1,
                        type="order_created",
                    ),
                    b"\n",
                ),
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            evidence = _write_fixture(Path(directory) / "evidence", records)
            analysis = analyze(evidence)
            self.assertEqual(
                [row["order_id"] for row in analysis.report["orders"]],
                ["ORD-A", "ORD-Z"],
            )

            missing = Path(directory) / "missing"
            with self.assertRaises(EvidenceError):
                analyze(missing)

            output_file = Path(directory) / "output-file"
            output_file.write_bytes(b"sentinel")
            with self.assertRaises(OSError):
                write_artifacts(analysis, output_file)

            commands = (
                [sys.executable, "-m", "order_reconcile"],
                [
                    sys.executable,
                    "-m",
                    "order_reconcile",
                    str(evidence),
                    str(Path(directory) / "out"),
                    "extra",
                ],
                [
                    sys.executable,
                    "-m",
                    "order_reconcile",
                    str(missing),
                    str(Path(directory) / "missing-out"),
                ],
                [sys.executable, "-m", "order_reconcile", str(evidence), str(output_file)],
            )
            for command in commands:
                completed = subprocess.run(
                    command,
                    cwd=CANDIDATE_ROOT,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 2, completed.stderr)
                self.assertEqual(completed.stdout, b"")
                self.assertTrue(completed.stderr)


if __name__ == "__main__":
    unittest.main()
