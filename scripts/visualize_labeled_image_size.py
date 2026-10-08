# 이미지의 크기 데이터 분포를 is_relevant 라벨별로 구분해 시각화

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = (
    Path(__file__).resolve().parents[1] / "data/images/train/image_size_blocks.json"
)
OUTPUT_PATH = Path(__file__).resolve().with_suffix(".png")


def load_sizes(path=DATA_PATH):
    with Path(path).open(encoding="utf-8") as file:
        records = json.load(file)

    keys = (
        "original_width",
        "original_height",
        "display_width",
        "display_height",
        "display_area",
        "relative_display_area",
    )
    rows, labels = [], []
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
        if record.get("page_id", 1000) > 61:
            # 공모전 제외하고 모두 제거
            continue
        rows.append(row)
        label = record.get("is_relevant")
        labels.append("UNKNOWN" if label is None else str(label))

    return np.asarray(rows, dtype=float).reshape(-1, 6), np.asarray(labels, dtype=str)


def scatter(ax, x, y, labels, title, xlabel, ylabel, colors):
    valid = np.isfinite(x) & np.isfinite(y)
    for label, color in colors.items():
        selected = valid & (labels == label)
        ax.scatter(
            x[selected],
            y[selected],
            color=color,
            s=18,
            alpha=0.55,
            edgecolors="none",
            label=f"{label} (n={selected.sum():,})",
        )
    ax.set(title=f"{title} (n={valid.sum():,})", xlabel=xlabel, ylabel=ylabel)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)


def histogram(ax, areas, title, xlabel):
    values = areas[np.isfinite(areas)]
    ax.hist(values, bins=50, color="steelblue", edgecolor="white")
    ax.set(title=f"{title} (n={values.size:,})", xlabel=xlabel, ylabel="Count")
    ax.grid(axis="y", alpha=0.2)


def create_figure(sizes, labels):
    (
        original_width,
        original_height,
        display_width,
        display_height,
        display_area,
        relative_display_area,
    ) = sizes.T
    original_area = original_width * original_height
    categories = sorted(set(labels))
    palette = plt.get_cmap("tab10")
    colors = {label: palette(index % 10) for index, label in enumerate(categories)}

    fig, axes = plt.subplots(3, 3, figsize=(18, 15), constrained_layout=True)
    plots = (
        (
            original_area,
            display_area,
            "Original vs absolute display area",
            "Original area (pixels²)",
            "Absolute display area (CSS pixels²)",
        ),
        (
            original_area,
            relative_display_area,
            "Original vs relative display area",
            "Original area (pixels²)",
            "Relative display area",
        ),
        (
            display_area,
            relative_display_area,
            "Absolute vs relative display area",
            "Absolute display area (CSS pixels²)",
            "Relative display area",
        ),
        (
            original_width,
            display_width,
            "Original vs display width",
            "Original width (pixels)",
            "Display width (CSS pixels)",
        ),
        (
            original_height,
            display_height,
            "Original vs display height",
            "Original height (pixels)",
            "Display height (CSS pixels)",
        ),
        # (
        #     original_width,
        #     original_height,
        #     "Original width vs height",
        #     "Original width (pixels)",
        #     "Original height (pixels)",
        # ),
        (
            display_width,
            display_height,
            "Display width vs height",
            "Display width (CSS pixels)",
            "Display height (CSS pixels)",
        ),
    )
    for ax, (x, y, title, xlabel, ylabel) in zip(axes.flat, plots):
        scatter(ax, x, y, labels, title, xlabel, ylabel, colors)

    histogram(
        axes[2, 0],
        original_area,
        "Original area distribution",
        "Original area (pixels²)",
    )
    histogram(
        axes[2, 1],
        display_area,
        "Display area distribution",
        "Display area (CSS pixels²)",
    )
    histogram(
        axes[2, 2],
        relative_display_area,
        "Relative display area distribution",
        "Relative display area",
    )
    # axes[2, 2].axis("off")
    fig.suptitle(f"Image sizes by is_relevant — {len(sizes):,} records", fontsize=16)
    return fig


def main():
    sizes, labels = load_sizes()
    fig = create_figure(sizes, labels)
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"Saved: {OUTPUT_PATH}")

    plt.show()
    # plt.close(fig)


if __name__ == "__main__":
    main()
