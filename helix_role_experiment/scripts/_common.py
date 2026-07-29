from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        destination.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean = {
                key: (
                    json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                )
                for key, value in row.items()
            }
            writer.writerow(clean)
    temporary.replace(destination)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_layer_spec(
    specification: str | None,
    available: list[int],
) -> list[int]:
    values = sorted(set(int(layer) for layer in available))
    if not values:
        raise ValueError("no layers are available")
    if specification in (None, "all"):
        return values
    if specification == "late-half":
        midpoint = (max(values) + 1) // 2
        selected = [layer for layer in values if layer >= midpoint]
    else:
        selected_set: set[int] = set()
        for item in specification.split(","):
            item = item.strip()
            if not item:
                continue
            if "-" in item:
                left, right = item.split("-", 1)
                selected_set.update(range(int(left), int(right) + 1))
            else:
                selected_set.add(int(item))
        selected = sorted(selected_set)
    missing = sorted(set(selected) - set(values))
    if missing:
        raise ValueError(f"requested layers are unavailable: {missing}")
    if not selected:
        raise ValueError("layer selection is empty")
    return selected
