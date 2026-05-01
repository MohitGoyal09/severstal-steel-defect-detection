#!/usr/bin/env python3
"""
Generate baseline graphs from TensorBoard event files.
Run locally on your Mac after downloading results from Vast.ai.

Usage:
    python scripts/generate_baseline_graphs.py
"""

import os
import sys
from pathlib import Path

# ─── Setup ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent  # project root
RESULTS_DIR = Path("/Users/mohit/sever-rtx5090-Unet-efficientnet-b5-BCEDiceLoss-RAdam")

# Find event files
EVENT_FILES = sorted(RESULTS_DIR.glob("*/runs/events*"))
print(f"Found event files: {len(EVENT_FILES)}")
for f in EVENT_FILES:
    print(f"  {f}")

if not EVENT_FILES:
    print("ERROR: No event files found!")
    sys.exit(1)

OUTPUT_DIR = BASE_DIR / "baseline_graphs"
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
    """Merge scalars from multiple event files."""
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
    print(f"  {k}: {n} points")

# ─── Publication Style ───────────────────────────────────────────────────────
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "lines.linewidth": 2.0,
        "lines.markersize": 4,
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


# ─── Helpers ───────────────────────────────────────────────────────────────────
def smooth(values, window=5):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def savefig(fig, name):
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {name}.png/pdf")


# ─── Extract Metrics ───────────────────────────────────────────────────────────
# Tag names from TensorBoard event files:
#   train/epoch/loss, train/epoch/dice_0, ..., train/epoch/dice_mean
#   valid/loss, valid/dice_0, ..., valid/dice_mean
loss_train = data.get("train/epoch/loss", {}).get("value", [])
loss_val = data.get("valid/loss", {}).get("value", [])

dice0_train = data.get("train/epoch/dice_0", {}).get("value", [])
dice1_train = data.get("train/epoch/dice_1", {}).get("value", [])
dice2_train = data.get("train/epoch/dice_2", {}).get("value", [])
dice3_train = data.get("train/epoch/dice_3", {}).get("value", [])
dice_mean_train = data.get("train/epoch/dice_mean", {}).get("value", [])

dice0_val = data.get("valid/dice_0", {}).get("value", [])
dice1_val = data.get("valid/dice_1", {}).get("value", [])
dice2_val = data.get("valid/dice_2", {}).get("value", [])
dice3_val = data.get("valid/dice_3", {}).get("value", [])
dice_mean_val = data.get("valid/dice_mean", {}).get("value", [])

# Validation starts at epoch 100 (runs every 10 epochs)
val_epochs = list(range(100, 100 + len(dice0_val)))
total_epochs = len(loss_train)

print(f"\nTrain epochs: {total_epochs}")
print(
    f"Validation runs: {len(dice0_val)} (epochs {val_epochs[0] if val_epochs else '?'} - {val_epochs[-1] if val_epochs else '?'})"
)
print(
    f"Final Val Dice Mean: {dice_mean_val[-1]:.4f}"
    if dice_mean_val
    else "No val dice data"
)
print(f"Final Val Loss: {loss_val[-1]:.4f}" if loss_val else "No val loss data")

# ─── FIG 1: Loss Curves ───────────────────────────────────────────────────────
print("\nGenerating Figure 1: Loss Curves...")
fig, ax = plt.subplots(figsize=(7, 4.5))
epochs_x = list(range(total_epochs))

if loss_train:
    ax.plot(
        epochs_x[: len(loss_train)],
        loss_train,
        color=C["train"],
        label="Train Loss",
        linewidth=2,
    )
if loss_val:
    ax.plot(
        val_epochs[: len(loss_val)],
        loss_val,
        color=C["val"],
        label="Val Loss",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    ax.axvline(x=100, color=C["gray"], linestyle="--", linewidth=1, alpha=0.7)
    ax.text(
        102, max(loss_val) * 0.9, "Val starts @ ep 100", fontsize=8, color=C["gray"]
    )
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (BCE + Dice)")
ax.set_title("Figure 1: Training and Validation Loss", fontweight="bold", pad=10)
ax.legend(frameon=False, loc="upper right")
ax.grid(True, alpha=0.3)
savefig(fig, "fig1_loss_curves")

# ─── FIG 2: Mean Dice Score (Train vs Val) ───────────────────────────────────
print("Generating Figure 2: Mean Dice Score...")
fig, ax = plt.subplots(figsize=(7, 4.5))
if dice_mean_train:
    ax.plot(
        epochs_x[: len(dice_mean_train)],
        dice_mean_train,
        color=C["train"],
        label="Train Mean Dice",
        linewidth=2,
    )
if dice_mean_val:
    ax.plot(
        val_epochs[: len(dice_mean_val)],
        dice_mean_val,
        color=C["val"],
        label="Val Mean Dice",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    final_val = dice_mean_val[-1]
    ax.annotate(
        f"Final: {final_val:.3f}",
        xy=(val_epochs[len(dice_mean_val) - 1], final_val),
        xytext=(val_epochs[len(dice_mean_val) - 1] - 15, final_val - 0.05),
        arrowprops=dict(arrowstyle="->", color=C["val"]),
        fontsize=9,
        color=C["val"],
    )
ax.axhline(
    y=0.9,
    color=C["gray"],
    linestyle="--",
    linewidth=1,
    alpha=0.7,
    label="90% threshold",
)
ax.set_xlabel("Epoch")
ax.set_ylabel("Mean Dice Score")
ax.set_title("Figure 2: Mean Dice Score (All Classes)", fontweight="bold", pad=10)
ax.legend(frameon=False)
ax.set_ylim([0, 1.05])
ax.grid(True, alpha=0.3)
savefig(fig, "fig2_mean_dice")

# ─── FIG 3: Per-Class Dice (Val) ───────────────────────────────────────────
print("Generating Figure 3: Per-Class Dice (Val)...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ax = axes[0]
for vals, name, color in [
    (dice0_val, "Class 1 (Heavy)", C["c0"]),
    (dice1_val, "Class 2 (Crazing)", C["c1"]),
    (dice2_val, "Class 3 (Rolled-in)", C["c2"]),
    (dice3_val, "Class 4 (Pitted)", C["c3"]),
]:
    if vals:
        ax.plot(
            val_epochs[: len(vals)],
            vals,
            color=color,
            label=name,
            linewidth=2,
            marker="o",
            markersize=4,
        )
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation Dice Score")
ax.set_title("Validation Dice Scores", fontweight="bold")
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.set_ylim([0, 1.05])
ax.grid(True, alpha=0.3)

ax = axes[1]
for vals, name, color in [
    (dice0_train, "Class 1 (Heavy)", C["c0"]),
    (dice1_train, "Class 2 (Crazing)", C["c1"]),
    (dice2_train, "Class 3 (Rolled-in)", C["c2"]),
    (dice3_train, "Class 4 (Pitted)", C["c3"]),
]:
    if vals:
        ax.plot(epochs_x[: len(vals)], vals, color=color, label=name, linewidth=2)
ax.set_xlabel("Epoch")
ax.set_ylabel("Training Dice Score")
ax.set_title("Training Dice Scores", fontweight="bold")
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.set_ylim([0, 1.05])
ax.grid(True, alpha=0.3)

plt.tight_layout()
savefig(fig, "fig3_dice_per_class")

# ─── FIG 4: Final Results Bar ────────────────────────────────────────────────
print("Generating Figure 4: Final Results Bar...")
fig, ax = plt.subplots(figsize=(7, 4.5))
classes = [
    "Class 1\n(Heavy)",
    "Class 2\n(Crazing)",
    "Class 3\n(Rolled-in)",
    "Class 4\n(Pitted)",
]
final_vals = []
for vals in [dice0_val, dice1_val, dice2_val, dice3_val]:
    final_vals.append(float(vals[-1]) if vals else 0.0)
mean_val = float(np.mean(final_vals))

colors = [C["c0"], C["c1"], C["c2"], C["c3"]]
bars = ax.bar(
    classes, final_vals, color=colors, edgecolor="white", linewidth=1.5, width=0.6
)
for bar, val in zip(bars, final_vals):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{val:.3f}",
        ha="center",
        fontsize=11,
        fontweight="bold",
    )
ax.axhline(
    y=mean_val,
    color="black",
    linestyle="--",
    linewidth=2,
    label=f"Mean: {mean_val:.3f}",
)
ax.axhline(
    y=0.9,
    color=C["gray"],
    linestyle=":",
    linewidth=1.5,
    alpha=0.7,
    label="90% threshold",
)
ax.set_ylabel("Validation Dice Score")
ax.set_title(
    "Figure 4: Per-Class Validation Dice (Final Epoch)", fontweight="bold", pad=10
)
ax.set_ylim([0, 1.18])
ax.legend(frameon=False, fontsize=9)
ax.grid(True, alpha=0.3, axis="y")
savefig(fig, "fig4_final_dice_bar")

# ─── FIG 5: Convergence ──────────────────────────────────────────────────────
print("Generating Figure 5: Convergence...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ax = axes[0]
if loss_train:
    raw = np.array(loss_train)
    sm = smooth(raw, window=5)
    ax.plot(
        epochs_x[: len(raw)],
        raw,
        color="lightgray",
        linewidth=0.8,
        alpha=0.7,
        label="Raw",
    )
    ax.plot(
        epochs_x[: len(sm)],
        sm,
        color=C["train"],
        linewidth=2.5,
        label="Smoothed (5-ep MA)",
    )
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training Loss Convergence", fontweight="bold")
ax.legend(frameon=False, fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
if dice_mean_val:
    raw = np.array(dice_mean_val)
    sm = smooth(raw, window=3)
    ax.plot(
        val_epochs[: len(raw)],
        raw,
        color="lightgray",
        linewidth=0.8,
        alpha=0.7,
        label="Raw",
    )
    ax.plot(
        val_epochs[: len(sm)],
        sm,
        color=C["val"],
        linewidth=2.5,
        label="Smoothed (3-ep MA)",
    )
    ax.set_ylim([0, 1.05])
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation Mean Dice")
ax.set_title("Validation Dice Convergence", fontweight="bold")
ax.legend(frameon=False, fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
savefig(fig, "fig5_convergence")

# ─── FIG 6: Multi-Panel Summary ──────────────────────────────────────────────
print("Generating Figure 6: Multi-Panel Summary...")
from string import ascii_uppercase

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)


def add_panel_label(ax, letter):
    ax.text(
        -0.12,
        1.1,
        letter,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
    )


# Panel A: Loss
ax = fig.add_subplot(gs[0, 0])
ax.plot(
    epochs_x[: len(loss_train)],
    loss_train,
    color=C["train"],
    linewidth=1.8,
    label="Train",
)
if loss_val:
    ax.plot(
        val_epochs[: len(loss_val)],
        loss_val,
        color=C["val"],
        linewidth=1.8,
        label="Val",
        marker="o",
        markersize=3,
    )
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("A. Loss Curves", fontweight="bold", fontsize=10)
ax.legend(frameon=False, fontsize=8)
ax.grid(True, alpha=0.3)
add_panel_label(ax, "A")

# Panel B: Mean Dice
ax = fig.add_subplot(gs[0, 1])
if dice_mean_train:
    ax.plot(
        epochs_x[: len(dice_mean_train)],
        dice_mean_train,
        color=C["train"],
        linewidth=1.8,
        label="Train",
    )
if dice_mean_val:
    ax.plot(
        val_epochs[: len(dice_mean_val)],
        dice_mean_val,
        color=C["val"],
        linewidth=1.8,
        label="Val",
        marker="o",
        markersize=3,
    )
ax.axhline(y=0.9, color=C["gray"], linestyle="--", linewidth=1, alpha=0.7)
ax.set_xlabel("Epoch")
ax.set_ylabel("Mean Dice")
ax.set_title("B. Mean Dice Score", fontweight="bold", fontsize=10)
ax.legend(frameon=False, fontsize=8)
ax.set_ylim([0, 1.05])
ax.grid(True, alpha=0.3)
add_panel_label(ax, "B")

# Panel C: Final Val Bar
ax = fig.add_subplot(gs[0, 2])
bars = ax.bar(
    ["C1", "C2", "C3", "C4"],
    final_vals,
    color=[C["c0"], C["c1"], C["c2"], C["c3"]],
    edgecolor="white",
    linewidth=1,
    width=0.6,
)
for bar, val in zip(bars, final_vals):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.015,
        f"{val:.2f}",
        ha="center",
        fontsize=9,
    )
ax.axhline(
    y=mean_val,
    color="black",
    linestyle="--",
    linewidth=1.5,
    label=f"Mean: {mean_val:.3f}",
)
ax.set_ylabel("Val Dice")
ax.set_title("C. Final Val Dice", fontweight="bold", fontsize=10)
ax.set_ylim([0, 1.1])
ax.legend(frameon=False, fontsize=8)
ax.grid(True, alpha=0.3, axis="y")
add_panel_label(ax, "C")

# Panel D: Per-class val
ax = fig.add_subplot(gs[1, 0])
for vals, name, color in [
    (dice0_val, "C1", C["c0"]),
    (dice1_val, "C2", C["c1"]),
    (dice2_val, "C3", C["c2"]),
    (dice3_val, "C4", C["c3"]),
]:
    if vals:
        ax.plot(
            val_epochs[: len(vals)],
            vals,
            color=color,
            linewidth=1.5,
            label=name,
            marker="o",
            markersize=3,
        )
ax.set_xlabel("Epoch")
ax.set_ylabel("Val Dice")
ax.set_title("D. Val Dice by Class", fontweight="bold", fontsize=10)
ax.legend(frameon=False, fontsize=7, ncol=2)
ax.set_ylim([0, 1.05])
ax.grid(True, alpha=0.3)
add_panel_label(ax, "D")

# Panel E: Per-class train
ax = fig.add_subplot(gs[1, 1])
for vals, name, color in [
    (dice0_train, "C1", C["c0"]),
    (dice1_train, "C2", C["c1"]),
    (dice2_train, "C3", C["c2"]),
    (dice3_train, "C4", C["c3"]),
]:
    if vals:
        ax.plot(epochs_x[: len(vals)], vals, color=color, linewidth=1.5, label=name)
ax.set_xlabel("Epoch")
ax.set_ylabel("Train Dice")
ax.set_title("E. Train Dice by Class", fontweight="bold", fontsize=10)
ax.legend(frameon=False, fontsize=7, ncol=2)
ax.set_ylim([0, 1.05])
ax.grid(True, alpha=0.3)
add_panel_label(ax, "E")

# Panel F: Results Table
ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
table_data = [
    ["Metric", "Value"],
    ["Total Epochs", str(total_epochs)],
    ["Val Runs", f"{len(dice0_val)}"],
    ["Val Dice Mean", f"{dice_mean_val[-1]:.4f}" if dice_mean_val else "N/A"],
    ["Val Loss", f"{loss_val[-1]:.4f}" if loss_val else "N/A"],
    ["Class 1 (Heavy)", f"{dice0_val[-1]:.4f}" if dice0_val else "N/A"],
    ["Class 2 (Crazing)", f"{dice1_val[-1]:.4f}" if dice1_val else "N/A"],
    ["Class 3 (Rolled-in)", f"{dice2_val[-1]:.4f}" if dice2_val else "N/A"],
    ["Class 4 (Pitted)", f"{dice3_val[-1]:.4f}" if dice3_val else "N/A"],
]
table = ax.table(
    cellText=table_data, loc="center", cellLoc="center", colWidths=[0.5, 0.4]
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 2.0)
for j in range(2):
    table[(0, j)].set_facecolor("#0072B2")
    table[(0, j)].set_text_props(color="white", fontweight="bold")
ax.set_title("F. Final Results", fontweight="bold", fontsize=10, pad=20)
add_panel_label(ax, "F")

plt.suptitle(
    "Baseline U-Net (EfficientNet-B5) — Training Results",
    fontsize=14,
    fontweight="bold",
    y=1.02,
)
savefig(fig, "fig6_summary")

# ─── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("DONE! All figures saved to:")
print(f"  {OUTPUT_DIR}")
print(f"\nFinal Results:")
print(f"  Total Epochs:       {total_epochs}")
print(f"  Val Runs:           {len(dice0_val)}")
print(
    f"  Final Val Dice Mean: {dice_mean_val[-1]:.4f}"
    if dice_mean_val
    else "  Val Dice Mean: N/A"
)
print(f"  Final Val Loss:     {loss_val[-1]:.4f}" if loss_val else "  Val Loss: N/A")
for i, vals in enumerate([dice0_val, dice1_val, dice2_val, dice3_val], 1):
    print(f"  Class {i}:           {vals[-1]:.4f}" if vals else f"  Class {i}: N/A")
print(f"{'=' * 60}")
for f in sorted(OUTPUT_DIR.glob("fig*.png")):
    print(f"  {f.name} ({f.stat().st_size / 1024:.0f} KB)")
