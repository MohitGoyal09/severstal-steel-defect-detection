#!/usr/bin/env python3
"""
Generate IEEE publication-ready baseline graphs from TensorBoard event files.
Run locally on your Mac after downloading results from Vast.ai.

Usage:
    python scripts/generate_baseline_graphs.py
"""

import os
import sys
from pathlib import Path

# ─── Setup ───────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("/Users/mohit/sever-rtx5090-Unet-efficientnet-b5-BCEDiceLoss-RAdam")
EVENT_FILES = sorted(RESULTS_DIR.glob("*/runs/events*"))
print(f"Found event files: {len(EVENT_FILES)}")
for f in EVENT_FILES:
    print(f"  {f}")

if not EVENT_FILES:
    print("ERROR: No event files found!")
    sys.exit(1)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "baseline_graphs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ─── Install deps if needed ───────────────────────────────────────────────────
def import_or_install(packages):
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            import subprocess

            subprocess.run(
                [sys.executable, "-m", "pip", "install", pkg, "-q"], check=True
            )


import_or_install(["tensorboard", "scipy", "numpy", "matplotlib", "pandas"])

# ─── Load TensorBoard Data ────────────────────────────────────────────────────
from tensorboard.backend.event_processing import event_accumulator


def load_all_scalars(event_files):
    merged = {}
    for ef in event_files:
        try:
            ea = event_accumulator.EventAccumulator(str(ef))
            ea.Reload()
            for tag in ea.Tags().get("scalars", []):
                if tag not in merged:
                    merged[tag] = {"step": [], "value": [], "wall_time": []}
                for ev in ea.Scalars(tag):
                    merged[tag]["step"].append(ev.step)
                    merged[tag]["value"].append(ev.value)
                    merged[tag]["wall_time"].append(ev.wall_time)
        except Exception as e:
            print(f"  Warning loading {ef.name}: {e}")
    return merged


print("\nLoading TensorBoard data...")
data = load_all_scalars(EVENT_FILES)
print(f"Loaded {len(data)} metrics:")
for k in sorted(data.keys()):
    n = len(data[k]["value"])
    print(f"  {k}: {n} points, steps {data[k]['step'][:3]}...{data[k]['step'][-3:]}")

# ─── IEEE Publication Style ──────────────────────────────────────────────────
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# IEEE single-column: 3.5 inches = 89mm
# IEEE double-column: 7.2 inches = 183mm
# Font sizes must be >= 6pt at final print size
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

# Colorblind-safe Okabe-Ito
C = {
    "train": "#0072B2",
    "val": "#D55E00",
    "c0": "#E69F00",
    "c1": "#56B4E9",
    "c2": "#009E73",
    "c3": "#CC79A7",
    "gray": "#999999",
}


def savefig(fig, name):
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {name}.png/pdf")


# ─── Extract Metrics with STEPS ──────────────────────────────────────────────
def get_metric(tag_name):
    """Get metric with actual step values (epoch numbers)."""
    d = data.get(tag_name, {})
    steps = d.get("step", [])
    values = d.get("value", [])
    return np.array(steps), np.array(values)


# Training metrics (logged every epoch: steps 0, 1, 2, ..., 62)
train_loss_steps, train_loss_vals = get_metric("train/epoch/loss")
train_dice0_steps, train_dice0_vals = get_metric("train/epoch/dice_0")
train_dice1_steps, train_dice1_vals = get_metric("train/epoch/dice_1")
train_dice2_steps, train_dice2_vals = get_metric("train/epoch/dice_2")
train_dice3_steps, train_dice3_vals = get_metric("train/epoch/dice_3")
train_dice_mean_steps, train_dice_mean_vals = get_metric("train/epoch/dice_mean")

# Validation metrics (logged every 10 epochs starting at 100: steps 100, 101, ...)
val_loss_steps, val_loss_vals = get_metric("valid/loss")
val_dice0_steps, val_dice0_vals = get_metric("valid/dice_0")
val_dice1_steps, val_dice1_vals = get_metric("valid/dice_1")
val_dice2_steps, val_dice2_vals = get_metric("valid/dice_2")
val_dice3_steps, val_dice3_vals = get_metric("valid/dice_3")
val_dice_mean_steps, val_dice_mean_vals = get_metric("valid/dice_mean")

# Get actual epoch range
total_epochs = int(train_loss_steps[-1]) + 1 if len(train_loss_steps) > 0 else 0
val_start = int(val_loss_steps[0]) if len(val_loss_steps) > 0 else 100

print(f"\nTrain epochs: 0-{total_epochs - 1} ({len(train_loss_vals)} points)")
print(
    f"Val epochs: {val_start}-{int(val_loss_steps[-1]) if len(val_loss_steps) > 0 else '?'} ({len(val_loss_vals)} points)"
)

# ─── FIG 1: Loss Curves ───────────────────────────────────────────────────────
print("\nGenerating Figure 1: Loss Curves...")
fig, ax = plt.subplots(figsize=(7, 4.5))

# Plot training loss at actual epoch numbers
ax.plot(
    train_loss_steps,
    train_loss_vals,
    color=C["train"],
    label="Train Loss",
    linewidth=2,
    zorder=2,
)

# Plot validation loss at actual epoch numbers with markers
if len(val_loss_vals) > 0:
    ax.plot(
        val_loss_steps,
        val_loss_vals,
        color=C["val"],
        label="Val Loss",
        linewidth=2,
        marker="o",
        markersize=6,
        markerfacecolor=C["val"],
        markeredgecolor="white",
        markeredgewidth=1.5,
        zorder=3,
    )

ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (BCE + Dice)")
ax.set_title("Figure 1: Training and Validation Loss", fontweight="bold", pad=10)
ax.legend(frameon=False, loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_xlim(left=-2)
ax.set_ylim(bottom=0)
savefig(fig, "fig1_loss_curves")

# ─── FIG 2: Mean Dice Score ──────────────────────────────────────────────────
print("Generating Figure 2: Mean Dice Score...")
fig, ax = plt.subplots(figsize=(7, 4.5))

# Training dice mean
ax.plot(
    train_dice_mean_steps,
    train_dice_mean_vals,
    color=C["train"],
    label="Train Mean Dice",
    linewidth=2,
    zorder=2,
)

# Validation dice mean with markers
if len(val_dice_mean_vals) > 0:
    ax.plot(
        val_dice_mean_steps,
        val_dice_mean_vals,
        color=C["val"],
        label="Val Mean Dice",
        linewidth=2,
        marker="o",
        markersize=6,
        markerfacecolor=C["val"],
        markeredgecolor="white",
        markeredgewidth=1.5,
        zorder=3,
    )
    # Annotate final val dice
    final_val = float(val_dice_mean_vals[-1])
    final_step = int(val_dice_mean_steps[-1])
    ax.annotate(
        f"Final: {final_val:.3f}",
        xy=(final_step, final_val),
        xytext=(final_step - 15, final_val - 0.08),
        arrowprops=dict(arrowstyle="->", color=C["val"], lw=1.5),
        fontsize=11,
        color=C["val"],
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor=C["val"], alpha=0.9
        ),
    )

ax.axhline(
    y=0.9,
    color=C["gray"],
    linestyle="--",
    linewidth=1.5,
    alpha=0.7,
    label="90% threshold",
)
ax.set_xlabel("Epoch")
ax.set_ylabel("Mean Dice Score")
ax.set_title("Figure 2: Mean Dice Score (All Classes)", fontweight="bold", pad=10)
ax.legend(frameon=False, loc="lower right")
ax.set_ylim([0, 1.05])
ax.set_xlim(left=-2)
ax.grid(True, alpha=0.3)
savefig(fig, "fig2_mean_dice")

# ─── FIG 3: Per-Class Dice (Validation) ──────────────────────────────────────
print("Generating Figure 3: Per-Class Dice (Validation)...")
fig, ax = plt.subplots(figsize=(7, 4.5))

for steps, vals, name, color in [
    (val_dice0_steps, val_dice0_vals, "Class 1 (Heavy)", C["c0"]),
    (val_dice1_steps, val_dice1_vals, "Class 2 (Crazing)", C["c1"]),
    (val_dice2_steps, val_dice2_vals, "Class 3 (Rolled-in)", C["c2"]),
    (val_dice3_steps, val_dice3_vals, "Class 4 (Pitted)", C["c3"]),
]:
    if len(vals) > 0:
        ax.plot(
            steps,
            vals,
            color=color,
            label=name,
            linewidth=2.5,
            marker="o",
            markersize=7,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=2,
        )

ax.set_xlabel("Epoch")
ax.set_ylabel("Validation Dice Score")
ax.set_title("Figure 3: Per-Class Validation Dice Scores", fontweight="bold", pad=10)
ax.legend(frameon=False, fontsize=10, loc="lower right")
ax.set_ylim([0, 1.05])
ax.grid(True, alpha=0.3)
savefig(fig, "fig3_dice_per_class_val")

# ─── FIG 4: Per-Class Dice (Training) ────────────────────────────────────────
print("Generating Figure 4: Per-Class Dice (Training)...")
fig, ax = plt.subplots(figsize=(7, 4.5))

for steps, vals, name, color in [
    (train_dice0_steps, train_dice0_vals, "Class 1 (Heavy)", C["c0"]),
    (train_dice1_steps, train_dice1_vals, "Class 2 (Crazing)", C["c1"]),
    (train_dice2_steps, train_dice2_vals, "Class 3 (Rolled-in)", C["c2"]),
    (train_dice3_steps, train_dice3_vals, "Class 4 (Pitted)", C["c3"]),
]:
    if len(vals) > 0:
        ax.plot(steps, vals, color=color, label=name, linewidth=2)

ax.set_xlabel("Epoch")
ax.set_ylabel("Training Dice Score")
ax.set_title("Figure 4: Per-Class Training Dice Scores", fontweight="bold", pad=10)
ax.legend(frameon=False, fontsize=10, loc="lower right")
ax.set_ylim([0, 1.05])
ax.set_xlim(left=-2)
ax.grid(True, alpha=0.3)
savefig(fig, "fig4_dice_per_class_train")

# ─── FIG 5: Final Results Bar ────────────────────────────────────────────────
print("Generating Figure 5: Final Results Bar...")
fig, ax = plt.subplots(figsize=(7, 4.5))

classes = [
    "Class 1\n(Heavy)",
    "Class 2\n(Crazing)",
    "Class 3\n(Rolled-in)",
    "Class 4\n(Pitted)",
]
final_vals = []
for vals in [val_dice0_vals, val_dice1_vals, val_dice2_vals, val_dice3_vals]:
    final_vals.append(float(vals[-1]) if len(vals) > 0 else 0.0)
mean_val = float(np.mean(final_vals))

colors = [C["c0"], C["c1"], C["c2"], C["c3"]]
bars = ax.bar(
    classes, final_vals, color=colors, edgecolor="white", linewidth=2, width=0.6
)

for bar, val in zip(bars, final_vals):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{val:.3f}",
        ha="center",
        fontsize=13,
        fontweight="bold",
    )

ax.axhline(
    y=mean_val,
    color="black",
    linestyle="--",
    linewidth=2.5,
    label=f"Mean: {mean_val:.3f}",
)
ax.axhline(
    y=0.9, color=C["gray"], linestyle=":", linewidth=2, alpha=0.7, label="90% threshold"
)
ax.set_ylabel("Validation Dice Score")
ax.set_title(
    "Figure 5: Per-Class Validation Dice (Final Epoch)", fontweight="bold", pad=10
)
ax.set_ylim([0, 1.15])
ax.legend(frameon=False, fontsize=11)
ax.grid(True, alpha=0.3, axis="y")
savefig(fig, "fig5_final_dice_bar")

# ─── FIG 6: Multi-Panel Summary (IEEE double-column) ─────────────────────────
print("Generating Figure 6: Multi-Panel Summary...")

fig = plt.figure(figsize=(14, 10))  # Double-column width
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


# Panel A: Loss
ax = fig.add_subplot(gs[0, 0])
ax.plot(train_loss_steps, train_loss_vals, color=C["train"], linewidth=2, label="Train")
if len(val_loss_vals) > 0:
    ax.plot(
        val_loss_steps,
        val_loss_vals,
        color=C["val"],
        linewidth=2,
        marker="o",
        markersize=5,
        markerfacecolor=C["val"],
        markeredgecolor="white",
        markeredgewidth=1.5,
        label="Val",
    )
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("A. Loss Curves", fontweight="bold", fontsize=12)
ax.legend(frameon=False, fontsize=9)
ax.grid(True, alpha=0.3)
add_panel_label(ax, "A")

# Panel B: Mean Dice
ax = fig.add_subplot(gs[0, 1])
ax.plot(
    train_dice_mean_steps,
    train_dice_mean_vals,
    color=C["train"],
    linewidth=2,
    label="Train",
)
if len(val_dice_mean_vals) > 0:
    ax.plot(
        val_dice_mean_steps,
        val_dice_mean_vals,
        color=C["val"],
        linewidth=2,
        marker="o",
        markersize=5,
        markerfacecolor=C["val"],
        markeredgecolor="white",
        markeredgewidth=1.5,
        label="Val",
    )
ax.axhline(y=0.9, color=C["gray"], linestyle="--", linewidth=1.5, alpha=0.7)
ax.set_xlabel("Epoch")
ax.set_ylabel("Mean Dice")
ax.set_title("B. Mean Dice Score", fontweight="bold", fontsize=12)
ax.legend(frameon=False, fontsize=9)
ax.set_ylim([0, 1.05])
ax.grid(True, alpha=0.3)
add_panel_label(ax, "B")

# Panel C: Final Bar
ax = fig.add_subplot(gs[0, 2])
bars = ax.bar(
    ["C1", "C2", "C3", "C4"],
    final_vals,
    color=[C["c0"], C["c1"], C["c2"], C["c3"]],
    edgecolor="white",
    linewidth=1.5,
    width=0.6,
)
for bar, val in zip(bars, final_vals):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.015,
        f"{val:.2f}",
        ha="center",
        fontsize=10,
    )
ax.axhline(
    y=mean_val,
    color="black",
    linestyle="--",
    linewidth=1.5,
    label=f"Mean: {mean_val:.3f}",
)
ax.set_ylabel("Val Dice")
ax.set_title("C. Final Val Dice", fontweight="bold", fontsize=12)
ax.set_ylim([0, 1.1])
ax.legend(frameon=False, fontsize=9)
ax.grid(True, alpha=0.3, axis="y")
add_panel_label(ax, "C")

# Panel D: Val per class
ax = fig.add_subplot(gs[1, 0])
for steps, vals, name, color in [
    (val_dice0_steps, val_dice0_vals, "C1", C["c0"]),
    (val_dice1_steps, val_dice1_vals, "C2", C["c1"]),
    (val_dice2_steps, val_dice2_vals, "C3", C["c2"]),
    (val_dice3_steps, val_dice3_vals, "C4", C["c3"]),
]:
    if len(vals) > 0:
        ax.plot(
            steps,
            vals,
            color=color,
            linewidth=1.8,
            label=name,
            marker="o",
            markersize=4,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=1,
        )
ax.set_xlabel("Epoch")
ax.set_ylabel("Val Dice")
ax.set_title("D. Val Dice by Class", fontweight="bold", fontsize=12)
ax.legend(frameon=False, fontsize=8, ncol=2)
ax.set_ylim([0, 1.05])
ax.grid(True, alpha=0.3)
add_panel_label(ax, "D")

# Panel E: Train per class
ax = fig.add_subplot(gs[1, 1])
for steps, vals, name, color in [
    (train_dice0_steps, train_dice0_vals, "C1", C["c0"]),
    (train_dice1_steps, train_dice1_vals, "C2", C["c1"]),
    (train_dice2_steps, train_dice2_vals, "C3", C["c2"]),
    (train_dice3_steps, train_dice3_vals, "C4", C["c3"]),
]:
    if len(vals) > 0:
        ax.plot(steps, vals, color=color, linewidth=1.8, label=name)
ax.set_xlabel("Epoch")
ax.set_ylabel("Train Dice")
ax.set_title("E. Train Dice by Class", fontweight="bold", fontsize=12)
ax.legend(frameon=False, fontsize=8, ncol=2)
ax.set_ylim([0, 1.05])
ax.grid(True, alpha=0.3)
add_panel_label(ax, "E")

# Panel F: Results Table
ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
table_data = [
    ["Metric", "Value"],
    ["Total Epochs", str(total_epochs)],
    ["Val Runs", f"{len(val_dice0_vals)}"],
    [
        "Val Dice Mean",
        f"{val_dice_mean_vals[-1]:.4f}" if len(val_dice_mean_vals) > 0 else "N/A",
    ],
    ["Val Loss", f"{val_loss_vals[-1]:.4f}" if len(val_loss_vals) > 0 else "N/A"],
    [
        "Class 1 (Heavy)",
        f"{val_dice0_vals[-1]:.4f}" if len(val_dice0_vals) > 0 else "N/A",
    ],
    [
        "Class 2 (Crazing)",
        f"{val_dice1_vals[-1]:.4f}" if len(val_dice1_vals) > 0 else "N/A",
    ],
    [
        "Class 3 (Rolled-in)",
        f"{val_dice2_vals[-1]:.4f}" if len(val_dice2_vals) > 0 else "N/A",
    ],
    [
        "Class 4 (Pitted)",
        f"{val_dice3_vals[-1]:.4f}" if len(val_dice3_vals) > 0 else "N/A",
    ],
]
table = ax.table(
    cellText=table_data, loc="center", cellLoc="center", colWidths=[0.5, 0.4]
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2.0)
for j in range(2):
    table[(0, j)].set_facecolor("#0072B2")
    table[(0, j)].set_text_props(color="white", fontweight="bold")
ax.set_title("F. Final Results", fontweight="bold", fontsize=12, pad=20)
add_panel_label(ax, "F")

plt.suptitle(
    "Baseline U-Net (EfficientNet-B5) — Training Results",
    fontsize=16,
    fontweight="bold",
    y=1.02,
)
savefig(fig, "fig6_summary")

# ─── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("DONE! All figures saved to:")
print(f"  {OUTPUT_DIR}")
print(f"\nFinal Results:")
print(f"  Total Epochs:        {total_epochs}")
print(f"  Val Runs:            {len(val_dice0_vals)}")
print(
    f"  Final Val Dice Mean: {val_dice_mean_vals[-1]:.4f}"
    if len(val_dice_mean_vals) > 0
    else "  N/A"
)
print(
    f"  Final Val Loss:      {val_loss_vals[-1]:.4f}"
    if len(val_loss_vals) > 0
    else "  N/A"
)
for i, vals in enumerate(
    [val_dice0_vals, val_dice1_vals, val_dice2_vals, val_dice3_vals], 1
):
    print(
        f"  Class {i}:            {vals[-1]:.4f}"
        if len(vals) > 0
        else f"  Class {i}: N/A"
    )
print(f"{'=' * 60}")
for f in sorted(OUTPUT_DIR.glob("fig*.png")):
    print(f"  {f.name} ({f.stat().st_size / 1024:.0f} KB)")
