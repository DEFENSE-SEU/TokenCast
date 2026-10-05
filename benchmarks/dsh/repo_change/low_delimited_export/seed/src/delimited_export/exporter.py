from __future__ import annotations

import csv
import io
from collections.abc import Iterable, Mapping, Sequence


def export_rows(
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, str]],
) -> str:


    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=list(fieldnames),
        delimiter=",",
        quotechar='"',
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
        doublequote=True,
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()
