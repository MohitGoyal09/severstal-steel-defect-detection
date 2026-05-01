#!/usr/bin/env python3
"""
Generate IEEE publication-ready comparison graphs:
Baseline vs Fixed-Ratio Synthetic vs Curriculum Learning
"""

import os
import sys
from pathlib import Path

# ─── Setup ───────────────────────────────────────────────────────────────────
BASE_DIR = Path("/Users/mohit/sever-rtx5090-Unet-efficientnet-b5-BCEDiceLoss-RAdam")
SYNTH_DIR = Path("/Users/mohit/sever-synthetic-Unet-efficientnet-b5-BCEDiceLoss-RAdam")
CURR_DIR = Path("/Users/mohit/sever-curriculum-Unet-efficientnet-b5-BCEDiceLoss-RAdam")

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "comparison_graphs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ─── Install deps ────────────────────────────────────────────────────────────
def import_or_install(packages):
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            import subprocess

            subprocess.run(
                [sys.executable, "-m", "pip", "install", pkg, "-q"], check=True
            )


import_or_install(["tensorboard", "numpy", "matplotlib", "scipy"])

# ─── Load TensorBoard Data ───────────────────────────────────────────────────
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
    }
)

C = {
    "baseline": "#0072B2",  # Blue
    "fixed": "#D55E00",  # Orange-red
    "curriculum": "#009E73",  # Green
    "gray": "#999999",
}

METHODS = {
    "Baseline": ("baseline", BASE_DIR, C["baseline"]),
    "Fixed-Ratio (30%)": ("fixed", SYNTH_DIR, C["fixed"]),
    "Curriculum": ("curriculum", CURR_DIR, C["curriculum"]),
}


def load_scalars(event_dir):
    """Load all scalar metrics from event files in directory."""
    event_files = sorted(Path(event_dir).glob("*/runs/events*"))
    merged = {}
    for ef in event_files:
        try:
            ea = event_accumulator.EventAccumulator(str(ef))
            ea.Reload()
            for tag in ea.Tags().get("scalars", []):
                if tag not in merged:
                    merged[tag] = {"step": [], "value": []}
                for ev in ea.Scalars(tag):
                    merged[tag]["step"].append(ev.step)
                    merged[tag]["value"].append(ev.value)
        except Exception as e:
            print(f"  Warning: {ef.name}: {e}")
    return merged


def get_vals(data, tag_name):
    d = data.get(tag_name, {})
    return np.array(d.get("value", []))


def get_epoch_axis(vals):
    """Build epoch-based x-axis from value count."""
    return np.arange(len(vals))


print("Loading results...")
all_data = {}
for name, (key, path, color) in METHODS.items():
    if not path.exists():
        print(f"  ⚠ {name}: {path} not found (will be skipped)")
        continue
    print(f"  ✓ {name}: {path}")
    all_data[key] = load_scalars(path)
    print(f"    Tags: {len(all_data[key])}")

if not all_data:
    print("ERROR: No result directories found!")
    print("Make sure you have downloaded the results to your Mac.")
    sys.exit(1)


# ─── Extract Final Metrics ───────────────────────────────────────────────────
def get_final(data):
    """Extract final validation metrics from loaded data."""
    metrics = {}
    for tag in [
        "valid/dice_mean",
        "valid/loss",
        "valid/dice_0",
        "valid/dice_1",
        "valid/dice_2",
        "valid/dice_3",
    ]:
        vals = get_vals(data, tag)
        metrics[tag] = float(vals[-1]) if len(vals) > 0 else None
    return metrics


final_metrics = {}
for key in all_data:
    final_metrics[key] = get_final(all_data[key])

print("\nFinal Validation Metrics:")
print(
    f"{'Method':<25} {'Mean Dice':>10} {'Loss':>10} {'C1':>8} {'C2':>8} {'C3':>8} {'C4':>8}"
)
print("-" * 85)
class_labels = [
    "valid/dice_mean",
    "valid/loss",
    "valid/dice_0",
    "valid/dice_1",
    "valid/dice_2",
    "valid/dice_3",
]
for name, (key, path, color) in METHODS.items():
    if key not in final_metrics:
        continue
    m = final_metrics[key]
    vals = [m.get(t, None) for t in class_labels]
    s = f"{name:<25}"
    for v in vals:
        if v is not None:
            s += f" {v:>9.4f}"
        else:
            s += f" {'N/A':>9}"
    print(s)


# ─── Helper ──────────────────────────────────────────────────────────────────
def savefig(fig, name):
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {name}.png/pdf")


# ─── FIG 1: Mean Dice Comparison ─────────────────────────────────────────────
print("\nGenerating Figure 1: Mean Dice Comparison...")
fig, ax = plt.subplots(figsize=(7, 4.5))

for name, (key, path, color) in METHODS.items():
    if key not in all_data:
        continue
    data = all_data[key]
    train_vals = get_vals(data, "train/epoch/dice_mean")
    val_vals = get_vals(data, "valid/dice_mean")

    train_epochs = get_epoch_axis(train_vals)
    if len(train_vals) > 0:
        ax.plot(train_epochs, train_vals, color=color, alpha=0.4, linewidth=1.5)

    val_epochs = (
        np.arange(100, 100 + len(val_vals)) if len(val_vals) > 0 else np.array([])
    )
    if len(val_vals) > 0:
        ax.plot(
            val_epochs,
            val_vals,
            color=color,
            label=name,
            linewidth=2.5,
            linestyle="none",
            marker="o",
            markersize=7,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=2,
        )

ax.axhline(y=0.9, color=C["gray"], linestyle="--", linewidth=1.5, alpha=0.7)
ax.set_xlabel("Epoch")
ax.set_ylabel("Mean Dice Score")
ax.set_title("Figure 1: Mean Dice Score Comparison", fontweight="bold", pad=10)
ax.legend(frameon=False, loc="lower right")
ax.set_ylim(bottom=0, top=1.05)
ax.set_xlim(left=-2)
ax.grid(True, alpha=0.3)
savefig(fig, "fig1_comparison_mean_dice")

# ─── FIG 2: Per-Class Comparison (Final) ─────────────────────────────────────
print("Generating Figure 2: Per-Class Final Dice Comparison...")

fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(4)
width = 0.25
class_names = [
    "Class 1\n(Heavy)",
    "Class 2\n(Crazing)",
    "Class 3\n(Rolled-in)",
    "Class 4\n(Pitted)",
]

for i, (name, (key, path, color)) in enumerate(METHODS.items()):
    if key not in final_metrics:
        continue
    m = final_metrics[key]
    vals = [m.get(f"valid/dice_{j}", 0) or 0 for j in range(4)]
    offset = width * (i - 1)
    bars = ax.bar(
        x + offset,
        vals,
        width,
        label=name,
        color=color,
        edgecolor="white",
        linewidth=1.5,
    )
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{val:.2f}",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

ax.axhline(
    y=0.9,
    color=C["gray"],
    linestyle="--",
    linewidth=1.5,
    alpha=0.7,
    label="90% threshold",
)
ax.set_ylabel("Validation Dice Score")
ax.set_title(
    "Figure 2: Per-Class Dice Score Comparison (Final)", fontweight="bold", pad=10
)
ax.set_xticks(x)
ax.set_xticklabels(class_names)
ax.set_ylim(bottom=0, top=1.15)
ax.legend(frameon=False, fontsize=10)
ax.grid(True, alpha=0.3, axis="y")
savefig(fig, "fig2_comparison_per_class")

# ─── FIG 3: Class 3 (Rolled-in) Improvement Focus ────────────────────────────
print("Generating Figure 3: Class 3 (Rolled-in) Focus...")
fig, ax = plt.subplots(figsize=(7, 4.5))

for name, (key, path, color) in METHODS.items():
    if key not in all_data:
        continue
    data = all_data[key]
    val_vals = get_vals(data, "valid/dice_2")
    val_epochs = (
        np.arange(100, 100 + len(val_vals)) if len(val_vals) > 0 else np.array([])
    )
    if len(val_vals) > 0:
        ax.plot(
            val_epochs,
            val_vals,
            color=color,
            label=name,
            linewidth=2.5,
            linestyle="none",
            marker="o",
            markersize=7,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=2,
        )

ax.axhline(y=0.9, color=C["gray"], linestyle="--", linewidth=1.5, alpha=0.7)
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation Dice Score")
ax.set_title(
    "Figure 3: Class 3 (Rolled-in) Dice — Rare Class Improvement",
    fontweight="bold",
    pad=10,
)
ax.legend(frameon=False, loc="lower right")
ax.set_ylim(bottom=0, top=1.05)
ax.grid(True, alpha=0.3)
savefig(fig, "fig3_comparison_class3_focus")

# ─── FIG 4: Multi-Panel Summary ──────────────────────────────────────────────
print("Generating Figure 4: Multi-Panel Summary...")

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)


def add_panel_label(ax, letter):
    ax.text(
        -0.12,
        1.1,
        letter,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        va="top",
    )


# Panel A: Mean Dice
ax = fig.add_subplot(gs[0, 0])
for name, (key, path, color) in METHODS.items():
    if key not in all_data:
        continue
    data = all_data[key]
    val_vals = get_vals(data, "valid/dice_mean")
    val_epochs = (
        np.arange(100, 100 + len(val_vals)) if len(val_vals) > 0 else np.array([])
    )
    if len(val_vals) > 0:
        ax.plot(
            val_epochs,
            val_vals,
            color=color,
            linewidth=2,
            label=name,
            linestyle="none",
            marker="o",
            markersize=5,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=1.5,
        )
ax.axhline(y=0.9, color=C["gray"], linestyle="--", linewidth=1.5, alpha=0.7)
ax.set_xlabel("Epoch")
ax.set_ylabel("Mean Dice")
ax.set_title("A. Mean Val Dice", fontweight="bold", fontsize=12)
ax.legend(frameon=False, fontsize=9)
ax.set_ylim([0, 1.05])
ax.grid(True, alpha=0.3)
add_panel_label(ax, "A")

# Panel B: Val Loss
ax = fig.add_subplot(gs[0, 1])
for name, (key, path, color) in METHODS.items():
    if key not in all_data:
        continue
    data = all_data[key]
    val_vals = get_vals(data, "valid/loss")
    val_epochs = (
        np.arange(100, 100 + len(val_vals)) if len(val_vals) > 0 else np.array([])
    )
    if len(val_vals) > 0:
        ax.plot(
            val_epochs,
            val_vals,
            color=color,
            linewidth=2,
            label=name,
            linestyle="none",
            marker="o",
            markersize=5,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=1.5,
        )
ax.set_xlabel("Epoch")
ax.set_ylabel("Val Loss")
ax.set_title("B. Val Loss", fontweight="bold", fontsize=12)
ax.legend(frameon=False, fontsize=9)
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3)
add_panel_label(ax, "B")

# Panel C: Per-class bar
ax = fig.add_subplot(gs[0, 2])
x = np.arange(4)
width = 0.25
for i, (name, (key, path, color)) in enumerate(METHODS.items()):
    if key not in final_metrics:
        continue
    m = final_metrics[key]
    vals = [m.get(f"valid/dice_{j}", 0) or 0 for j in range(4)]
    offset = width * (i - 1)
    bars = ax.bar(
        x + offset, vals, width, label=name, color=color, edgecolor="white", linewidth=1
    )
ax.set_ylabel("Val Dice")
ax.set_title("C. Per-Class Dice", fontweight="bold", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(["C1", "C2", "C3", "C4"])
ax.set_ylim([0, 1.1])
ax.legend(frameon=False, fontsize=8)
ax.grid(True, alpha=0.3, axis="y")
add_panel_label(ax, "C")

# Panel D: Class 1
ax = fig.add_subplot(gs[1, 0])
for name, (key, path, color) in METHODS.items():
    if key not in all_data:
        continue
    data = all_data[key]
    val_vals = get_vals(data, "valid/dice_0")
    val_epochs = (
        np.arange(100, 100 + len(val_vals)) if len(val_vals) > 0 else np.array([])
    )
    if len(val_vals) > 0:
        ax.plot(
            val_epochs,
            val_vals,
            color=color,
            linewidth=1.8,
            label=name,
            linestyle="none",
            marker="o",
            markersize=4,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=1,
        )
ax.set_xlabel("Epoch")
ax.set_ylabel("Val Dice")
ax.set_title("D. Class 1 (Heavy)", fontweight="bold", fontsize=12)
ax.legend(frameon=False, fontsize=8)
ax.set_ylim([0, 1.05])
ax.grid(True, alpha=0.3)
add_panel_label(ax, "D")

# Panel E: Class 2
ax = fig.add_subplot(gs[1, 1])
for name, (key, path, color) in METHODS.items():
    if key not in all_data:
        continue
    data = all_data[key]
    val_vals = get_vals(data, "valid/dice_1")
    val_epochs = (
        np.arange(100, 100 + len(val_vals)) if len(val_vals) > 0 else np.array([])
    )
    if len(val_vals) > 0:
        ax.plot(
            val_epochs,
            val_vals,
            color=color,
            linewidth=1.8,
            label=name,
            linestyle="none",
            marker="o",
            markersize=4,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=1,
        )
ax.set_xlabel("Epoch")
ax.set_ylabel("Val Dice")
ax.set_title("E. Class 2 (Crazing)", fontweight="bold", fontsize=12)
ax.legend(frameon=False, fontsize=8)
ax.set_ylim([0, 1.05])
ax.grid(True, alpha=0.3)
add_panel_label(ax, "E")

# Panel F: Class 3 (key story)
ax = fig.add_subplot(gs[1, 2])
for name, (key, path, color) in METHODS.items():
    if key not in all_data:
        continue
    data = all_data[key]
    val_vals = get_vals(data, "valid/dice_2")
    val_epochs = (
        np.arange(100, 100 + len(val_vals)) if len(val_vals) > 0 else np.array([])
    )
    if len(val_vals) > 0:
        ax.plot(
            val_epochs,
            val_vals,
            color=color,
            linewidth=2.5,
            label=name,
            linestyle="none",
            marker="o",
            markersize=6,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=2,
        )
ax.axhline(y=0.9, color=C["gray"], linestyle="--", linewidth=1.5, alpha=0.7)
ax.set_xlabel("Epoch")
ax.set_ylabel("Val Dice")
ax.set_title("F. Class 3 (Rolled-in) — Rare Class", fontweight="bold", fontsize=12)
ax.legend(frameon=False, fontsize=9)
ax.set_ylim([0, 1.05])
ax.grid(True, alpha=0.3)
add_panel_label(ax, "F")

plt.suptitle(
    "Synthetic Data Augmentation for Steel Defect Detection — Comparison",
    fontsize=16,
    fontweight="bold",
    y=1.02,
)
savefig(fig, "fig4_comparison_summary")

# ─── FIG 5: Results Table ────────────────────────────────────────────────────
print("Generating Figure 5: Results Table...")
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis("off")

table_data = [["Method", "Mean Dice", "Loss", "C1", "C2", "C3", "C4"]]
for name, (key, path, color) in METHODS.items():
    if key not in final_metrics:
        continue
    m = final_metrics[key]
    row = [name]
    for tag in class_labels:
        v = m.get(tag)
        row.append(f"{v:.4f}" if v is not None else "N/A")
    table_data.append(row)

table = ax.table(
    cellText=table_data, loc="center", cellLoc="center", colWidths=[0.22] + [0.13] * 6
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2.2)

for j in range(7):
    table[(0, j)].set_facecolor("#0072B2")
    table[(0, j)].set_text_props(color="white", fontweight="bold")

# Color-code rows by method
row_colors = {
    "Baseline": C["baseline"],
    "Fixed-Ratio (30%)": C["fixed"],
    "Curriculum": C["curriculum"],
}
for i in range(1, len(table_data)):
    method_name = table_data[i][0]
    for j in range(7):
        table[(i, j)].set_facecolor(row_colors.get(method_name, "white"))
        table[(i, j)].set_text_props(
            color="white" if method_name in row_colors else "black", fontweight="bold"
        )

ax.set_title(
    "Comparison of Methods — Final Validation Metrics",
    fontweight="bold",
    fontsize=14,
    pad=20,
)
savefig(fig, "fig5_results_table")

# ─── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("DONE! Comparison figures saved to:")
print(f"  {OUTPUT_DIR}")
for f in sorted(OUTPUT_DIR.glob("fig*.png")):
    print(f"  {f.name} ({f.stat().st_size / 1024:.0f} KB)")
print(f"{'=' * 60}")
