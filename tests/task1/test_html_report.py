from __future__ import annotations

import csv
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "docs" / "task1-experiment-report.html"

SOURCE_TABLES = {
    "cnn-summary": ROOT / "results" / "evidence" / "task1" / "comparison.csv",
    "cnn-folds": ROOT / "results" / "evidence" / "task1" / "fold_metrics.csv",
    "cnn-oof": ROOT / "results" / "evidence" / "task1" / "oof_metrics.csv",
}


class ReportParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.fragment_links: list[str] = []
        self.local_links: list[str] = []
        self.tables: dict[str, list[dict[str, str]]] = {}
        self._table: str | None = None
        self._row: dict[str, str] | None = None
        self._cell_field: str | None = None
        self._cell_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(str(values["id"]))

        for attribute in ("href", "src"):
            target = values.get(attribute)
            if target and target.startswith("#"):
                self.fragment_links.append(target[1:])
            elif target and not target.startswith(("http:", "https:", "data:")):
                self.local_links.append(target)

        if tag == "table" and values.get("data-evidence-table"):
            self._table = str(values["data-evidence-table"])
            self.tables[self._table] = []
        elif tag == "tr" and self._table and values.get("data-source-row"):
            self._row = {"__row__": str(values["data-source-row"])}
        elif tag == "td" and self._row is not None and values.get("data-field"):
            self._cell_field = str(values["data-field"])
            self._cell_text = []
            self._row[self._cell_field] = str(values.get("data-value", ""))

    def handle_data(self, data: str) -> None:
        if self._cell_field is not None:
            self._cell_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "td" and self._row is not None and self._cell_field is not None:
            display_key = f"__display__{self._cell_field}"
            self._row[display_key] = "".join(self._cell_text).strip()
            self._cell_field = None
            self._cell_text = []
        elif tag == "tr" and self._table and self._row is not None:
            self.tables[self._table].append(self._row)
            self._row = None
        elif tag == "table":
            self._table = None


def parse_report() -> ReportParser:
    parser = ReportParser()
    parser.feed(REPORT.read_text(encoding="utf-8"))
    return parser


def test_report_navigation_and_local_assets_resolve() -> None:
    parser = parse_report()

    expected_sections = {
        "overview",
        "data",
        "method",
        "experiments",
        "results",
        "failures",
        "decision",
        "evidence",
    }
    assert expected_sections <= parser.ids
    assert set(parser.fragment_links) == expected_sections

    missing = [
        target
        for target in parser.local_links
        if not (REPORT.parent / target.split("#", 1)[0]).resolve().exists()
    ]
    assert missing == []


def test_report_result_tables_match_saved_evidence() -> None:
    parser = parse_report()

    assert set(SOURCE_TABLES) <= set(parser.tables)
    for table_name, source_path in SOURCE_TABLES.items():
        with source_path.open(encoding="utf-8", newline="") as handle:
            source_rows = list(csv.DictReader(handle))

        report_rows = parser.tables[table_name]
        assert len(report_rows) == len(source_rows), table_name
        for index, (report_row, source_row) in enumerate(
            zip(report_rows, source_rows, strict=True)
        ):
            assert report_row["__row__"] == str(index)
            for field, value in report_row.items():
                if field != "__row__" and not field.startswith("__display__"):
                    assert value == source_row[field], (table_name, index, field)

            for field, raw_value in source_row.items():
                display_key = f"__display__{field}"
                if display_key not in report_row:
                    continue
                if field == "fold":
                    expected_display = raw_value
                elif field.startswith(("top1_accuracy", "top5_accuracy")):
                    expected_display = f"{float(raw_value):.2%}"
                elif field.startswith(
                    ("macro_f1", "weighted_f1", "validation_loss")
                ):
                    expected_display = f"{float(raw_value):.4f}"
                else:
                    continue
                assert report_row[display_key] == expected_display, (
                    table_name,
                    index,
                    field,
                )


def test_report_freezes_the_mild_unweighted_cnn_for_stability() -> None:
    report = REPORT.read_text(encoding="utf-8")

    assert "task1_cnn_mild_aug_unweighted_v1" in report
    assert "stability" in report.lower()
    assert "Choose the mildly augmented unweighted CNN" in report
    assert "Choose the plain scratch CNN" not in report


if __name__ == "__main__":
    test_report_navigation_and_local_assets_resolve()
    test_report_result_tables_match_saved_evidence()
    test_report_freezes_the_mild_unweighted_cnn_for_stability()
