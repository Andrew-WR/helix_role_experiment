from __future__ import annotations

import html
from pathlib import Path
from typing import Iterable

import numpy as np


COLORS = ("#275dad", "#e07a5f", "#3d9970", "#7b2cbf", "#d4a017", "#4d4d4d")


def _scale(values: np.ndarray, low: float, high: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    minimum = float(np.nanmin(array))
    maximum = float(np.nanmax(array))
    if maximum - minimum < 1e-12:
        return np.full_like(array, (low + high) / 2.0)
    return low + (array - minimum) * (high - low) / (maximum - minimum)


def _svg_start(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,sans-serif;fill:#222}.axis{stroke:#555;stroke-width:1}"
        ".grid{stroke:#ddd;stroke-width:1}.legend{font-size:12px}.title{font-size:18px;font-weight:600}</style>",
        f'<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="25" text-anchor="middle" class="title">{html.escape(title)}</text>',
    ]


def scatter_svg(
    path: str | Path,
    x: np.ndarray,
    y: np.ndarray,
    groups: Iterable[str] | None,
    title: str,
    x_label: str,
    y_label: str,
    width: int = 760,
    height: int = 520,
) -> None:
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    group_array = np.asarray(list(groups) if groups is not None else ["all"] * len(x_array))
    left, right, top, bottom = 70, width - 30, 50, height - 65
    sx = _scale(x_array, left, right)
    sy = _scale(y_array, bottom, top)
    lines = _svg_start(width, height, title)
    lines += [
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis"/>',
        f'<text x="{(left+right)/2}" y="{height-18}" text-anchor="middle">{html.escape(x_label)}</text>',
        f'<text x="18" y="{(top+bottom)/2}" text-anchor="middle" transform="rotate(-90 18 {(top+bottom)/2})">{html.escape(y_label)}</text>',
    ]
    unique = list(dict.fromkeys(group_array.tolist()))
    for group_index, group in enumerate(unique):
        color = COLORS[group_index % len(COLORS)]
        mask = group_array == group
        for px, py in zip(sx[mask], sy[mask]):
            lines.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="3.2" fill="{color}" fill-opacity="0.65"/>')
        lines.append(
            f'<circle cx="{right-135}" cy="{top+16*group_index}" r="4" fill="{color}"/>'
            f'<text x="{right-125}" y="{top+4+16*group_index}" class="legend">{html.escape(str(group))}</text>'
        )
    lines.append("</svg>")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def line_svg(
    path: str | Path,
    series: dict[str, tuple[np.ndarray, np.ndarray]],
    title: str,
    x_label: str,
    y_label: str,
    width: int = 760,
    height: int = 520,
) -> None:
    all_x = np.concatenate([np.asarray(value[0]) for value in series.values()])
    all_y = np.concatenate([np.asarray(value[1]) for value in series.values()])
    left, right, top, bottom = 70, width - 30, 50, height - 65
    x_min, x_max = float(all_x.min()), float(all_x.max())
    y_min, y_max = float(all_y.min()), float(all_y.max())

    def sx(value: np.ndarray) -> np.ndarray:
        return left + (value - x_min) * (right - left) / max(x_max - x_min, 1e-12)

    def sy(value: np.ndarray) -> np.ndarray:
        return bottom - (value - y_min) * (bottom - top) / max(y_max - y_min, 1e-12)

    lines = _svg_start(width, height, title)
    lines += [
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis"/>',
        f'<text x="{(left+right)/2}" y="{height-18}" text-anchor="middle">{html.escape(x_label)}</text>',
        f'<text x="18" y="{(top+bottom)/2}" text-anchor="middle" transform="rotate(-90 18 {(top+bottom)/2})">{html.escape(y_label)}</text>',
    ]
    for index, (name, (x_values, y_values)) in enumerate(series.items()):
        color = COLORS[index % len(COLORS)]
        points = " ".join(
            f"{x:.2f},{y:.2f}"
            for x, y in zip(sx(np.asarray(x_values)), sy(np.asarray(y_values)))
        )
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        lines.append(
            f'<line x1="{right-145}" y1="{top+16*index}" x2="{right-130}" y2="{top+16*index}" stroke="{color}" stroke-width="2"/>'
            f'<text x="{right-125}" y="{top+4+16*index}" class="legend">{html.escape(name)}</text>'
        )
    lines.append("</svg>")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def heatmap_svg(
    path: str | Path,
    matrix: np.ndarray,
    row_labels: list[str],
    column_labels: list[str],
    title: str,
    width: int = 760,
    height: int = 620,
) -> None:
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (len(row_labels), len(column_labels)):
        raise ValueError("matrix shape does not match labels")
    left, right, top, bottom = 130, width - 40, 65, height - 110
    cell_w = (right - left) / max(1, len(column_labels))
    cell_h = (bottom - top) / max(1, len(row_labels))
    minimum, maximum = float(np.nanmin(values)), float(np.nanmax(values))
    lines = _svg_start(width, height, title)
    for row in range(values.shape[0]):
        lines.append(
            f'<text x="{left-8}" y="{top+(row+0.65)*cell_h:.2f}" text-anchor="end" class="legend">'
            f"{html.escape(row_labels[row])}</text>"
        )
        for column in range(values.shape[1]):
            ratio = (values[row, column] - minimum) / max(maximum - minimum, 1e-12)
            red = int(245 - 170 * ratio)
            green = int(248 - 70 * ratio)
            blue = int(252 - 20 * ratio)
            lines.append(
                f'<rect x="{left+column*cell_w:.2f}" y="{top+row*cell_h:.2f}" '
                f'width="{cell_w:.2f}" height="{cell_h:.2f}" fill="rgb({red},{green},{blue})"/>'
            )
            lines.append(
                f'<text x="{left+(column+0.5)*cell_w:.2f}" y="{top+(row+0.62)*cell_h:.2f}" '
                f'text-anchor="middle" class="legend">{values[row,column]:.2f}</text>'
            )
    for column, label in enumerate(column_labels):
        x = left + (column + 0.5) * cell_w
        lines.append(
            f'<text x="{x:.2f}" y="{bottom+8}" transform="rotate(45 {x:.2f} {bottom+8})" '
            f'class="legend">{html.escape(label)}</text>'
        )
    lines.append("</svg>")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def paired_effect_svg(
    path: str | Path,
    labels: list[str],
    estimates: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    title: str,
) -> None:
    width, height = 760, 120 + 65 * len(labels)
    left, right, top, bottom = 175, width - 40, 55, height - 45
    all_values = np.concatenate((np.asarray(lower), np.asarray(upper), np.array([0.0])))
    minimum, maximum = float(all_values.min()), float(all_values.max())

    def sx(value: float) -> float:
        return left + (value - minimum) * (right - left) / max(maximum - minimum, 1e-12)

    lines = _svg_start(width, height, title)
    lines.append(f'<line x1="{sx(0):.2f}" y1="{top}" x2="{sx(0):.2f}" y2="{bottom}" stroke="#888" stroke-dasharray="4 3"/>')
    for index, label in enumerate(labels):
        y = top + 35 + index * 60
        lines.append(f'<text x="{left-10}" y="{y+4}" text-anchor="end">{html.escape(label)}</text>')
        lines.append(f'<line x1="{sx(lower[index]):.2f}" y1="{y}" x2="{sx(upper[index]):.2f}" y2="{y}" stroke="#275dad" stroke-width="3"/>')
        lines.append(f'<circle cx="{sx(estimates[index]):.2f}" cy="{y}" r="6" fill="#e07a5f"/>')
    lines.append("</svg>")
    Path(path).write_text("\n".join(lines), encoding="utf-8")

