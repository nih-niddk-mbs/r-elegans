"""Render saved chemotaxis head trajectories as a dependency-free SVG."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def points(path: np.ndarray, center_x: float, center_y: float, scale: float) -> str:
    return " ".join(
        f"{center_x + scale * x:.2f},{center_y - scale * y:.2f}"
        for x, y in path
    )


def marker(position, center_x, center_y, scale, color, radius=4) -> str:
    x, y = position
    return (
        f'<circle cx="{center_x + scale * x:.2f}" '
        f'cy="{center_y - scale * y:.2f}" r="{radius}" fill="{color}"/>'
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    with np.load(args.artifact, allow_pickle=False) as data:
        sources = data["test_sources"]
        paths = data["test_head_position"]
        success = data["test_success"]
        neural_path = data["neural_test_representative_head_position"]
        neural_success = bool(data["neural_test_success"][0])
        direct_rate = float(np.mean(success))
        neural_rate = float(np.mean(data["neural_test_success"]))

    scale = 135.0
    panels = ((250.0, 250.0), (750.0, 250.0))
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="520" '
        'viewBox="0 0 1000 520">',
        '<rect width="1000" height="520" fill="#f8f5e9"/>',
        '<style>text{font-family:system-ui,sans-serif;fill:#252821}'
        '.title{font-size:18px;font-weight:600}.metric{font-size:14px}</style>',
    ]
    for center_x, center_y in panels:
        elements.append(
            f'<circle cx="{center_x}" cy="{center_y}" r="{1.5 * scale}" '
            'fill="#fffdf6" stroke="#555a50" stroke-width="2"/>'
        )

    direct_x, direct_y = panels[0]
    for source, path, reached in zip(sources, paths, success):
        color = "#277da1" if reached else "#c44536"
        elements.append(
            f'<polyline points="{points(path, direct_x, direct_y, scale)}" '
            f'fill="none" stroke="{color}" stroke-width="1.4" opacity="0.72"/>'
        )
        elements.append(marker(source, direct_x, direct_y, scale, "#4f8a3c", 3))

    neural_x, neural_y = panels[1]
    neural_color = "#7b2cbf" if neural_success else "#c44536"
    elements.extend(
        [
            f'<polyline points="{points(neural_path, neural_x, neural_y, scale)}" '
            f'fill="none" stroke="{neural_color}" stroke-width="2.5"/>',
            marker(sources[0], neural_x, neural_y, scale, "#4f8a3c", 6),
            marker(neural_path[0], neural_x, neural_y, scale, "#242424", 4),
            f'<text x="250" y="28" text-anchor="middle" class="title">'
            f'Body-direct held-out paths ({direct_rate:.1%} success)</text>',
            f'<text x="750" y="28" text-anchor="middle" class="title">'
            'Neural/NMJ representative path</text>',
            f'<text x="750" y="49" text-anchor="middle" class="metric">'
            f'All neural held-out trials: {neural_rate:.1%} success</text>',
            '<text x="500" y="500" text-anchor="middle" class="metric">'
            'green = food source; blue/purple = reached; red = missed; black = start'
            '</text>',
            '</svg>',
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(elements), encoding="utf-8")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
