"""Display five size plots without modifying the input JSON or saving files."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = (
    Path(__file__).resolve().parents[1] / "data/images/train/image_size_blocks.json"
)


def load_sizes(path=DATA_PATH):
    # Keep missing/invalid dimensions as NaN; zero dimensions remain valid.
    with Path(path).open(encoding="utf-8") as file:
        records = json.load(file)

    keys = ("original_width", "original_height", "display_width", "display_height")
    rows = []
    for record in records:
        size = record.get("size") or {}
        row = []
        for key in keys:
            value = size.get(key)
            try:
                number = float(value) if not isinstance(value, bool) else np.nan
            except (TypeError, ValueError):
                number = np.nan
            row.append(number if np.isfinite(number) and number >= 0 else np.nan)
        rows.append(row)
    return np.asarray(rows, dtype=float).reshape(-1, 4)


def scatter(ax, x, y, title, xlabel, ylabel):
    valid = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[valid], y[valid], s=12, alpha=0.35, edgecolors="none")
    ax.set(title=f"{title} (n={valid.sum():,})", xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=0.2)


def create_figure(sizes):
    original_width, original_height, display_width, display_height = sizes.T
    original_area = original_width * original_height
    display_area = display_width * display_height
    aspect_ratio = np.divide(
        original_width,
        original_height,
        out=np.full_like(original_width, np.nan),
        where=original_height > 0,
    )

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    scatter(
        axes[0, 0],
        original_area,
        display_area,
        "Original vs browser display area (log scales)",
        "Original area (pixels²)",
        "Browser display area (CSS pixels²)",
    )
    # Compress large outliers while retaining zero-area images.
    # Areas from 0 to 1 are linear; larger areas use a base-10 log scale.
    axes[0, 0].set_xscale("symlog", linthresh=1)
    axes[0, 0].set_yscale("symlog", linthresh=1)
    axes[0, 1].set_xscale("symlog", linthresh=1)
    axes[0, 1].set_yscale("symlog", linthresh=1)
    paired = np.isfinite(original_area) & np.isfinite(display_area)
    if paired.any():
        maximum = max(original_area[paired].max(), display_area[paired].max())
        axes[0, 0].plot(
            [0, maximum],
            [0, maximum],
            "--",
            color="gray",
            linewidth=1,
            label="Equal area",
        )
        axes[0, 0].legend()

    original_valid = original_area[np.isfinite(original_area)]
    display_valid = display_area[np.isfinite(display_area)]
    combined = np.concatenate([original_valid, display_valid])
    bins = np.histogram_bin_edges(combined, bins=50) if combined.size else 50
    for values, label in (
        (original_valid, "Original"),
        (display_valid, "Browser display"),
    ):
        axes[0, 1].hist(
            values, bins=bins, alpha=0.5, label=f"{label} (n={values.size:,})"
        )
    axes[0, 1].set(title="Area distributions", xlabel="Area (pixels²)", ylabel="Count")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.2)

    scatter(
        axes[0, 2],
        original_width,
        original_height,
        "Original width vs height",
        "Original width (pixels)",
        "Original height (pixels)",
    )
    scatter(
        axes[1, 0],
        display_width,
        display_height,
        "Browser display width vs height",
        "Display width (CSS pixels)",
        "Display height (CSS pixels)",
    )
    scatter(
        axes[1, 1],
        aspect_ratio,
        original_area,
        "Original aspect ratio vs pixel area",
        "Original aspect ratio (width / height)",
        "Original area (pixels²)",
    )
    axes[1, 2].axis("off")
    fig.suptitle(f"Image sizes — {len(sizes):,} records", fontsize=16)
    return fig


def main():
    fig = create_figure(load_sizes())
    # plt.show()
    fig.savefig(Path(__file__).resolve().parent / "visualize_image_size.png")


if __name__ == "__main__":
    main()
