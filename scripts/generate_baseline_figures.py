#!/usr/bin/env python3
"""
Baseline Training Results Visualization
Creates publication-quality figures for IEEE paper
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
from datetime import datetime

# ─── Publication Style Setup ───
plt.style.use("default")
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica"],
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

# Colorblind-safe palette (Okabe-Ito)
COLORS = {
    "train": "#0072B2",  # Blue
    "val": "#D55E00",  # Orange
    "class0": "#E69F00",  # Yellow
    "class1": "#56B4E9",  # Sky blue
    "class2": "#009E73",  # Teal
    "class3": "#CC79A7",  # Pink
    "mean": "#E69F00",  # Gold
    "loss": "#000000",  # Black
}

OUTPUT_DIR = Path("/workspace/code/baseline_results")
OUTPUT_DIR.mkdir(exist_ok=True)


# ─── Load TensorBoard Data ───
def load_tensorboard_data(log_dir):
    """Load all scalars from TensorBoard event files."""
    ea = event_accumulator.EventAccumulator(str(log_dir))
    ea.Reload()

    data = {}
    for tag in ea.Tags()["scalars"]:
        events = ea.Scalars(tag)
        data[tag] = {
            "step": [e.step for e in events],
            "value": [e.value for e in events],
            "wall_time": [e.wall_time for e in events],
        }

    return data


# ─── Plot 1: Loss Curves (Train vs Validation) ───
def plot_loss_curves(data, save_dir):
    """Figure 1: Training and validation loss over epochs."""
    fig, ax = plt.subplots(figsize=(6, 4))

    epochs = data["epoch_loss"]["step"]
    ax.plot(
        epochs,
        data["epoch_loss"]["value"],
        color=COLORS["train"],
        label="Train Loss",
        linewidth=2,
    )

    if "epoch_val_loss" in data:
        val_epochs = data["epoch_val_loss"]["step"]
        val_start = min(val_epochs)
        ax.plot(
            val_epochs,
            data["epoch_val_loss"]["value"],
            color=COLORS["val"],
            label="Val Loss",
            linewidth=2,
        )
        ax.axvline(
            x=val_start,
            color="gray",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label=f"Val starts @ ep {val_start}",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (BCE + Dice)")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Figure 1: Training and Validation Loss", fontweight="bold", pad=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "fig1_loss_curves.png", dpi=300, bbox_inches="tight")
    plt.savefig(save_dir / "fig1_loss_curves.pdf", bbox_inches="tight")
    print(f"✓ Saved fig1_loss_curves.png/pdf")
    plt.close()


# ─── Plot 2: Dice Score Curves (All Classes) ───
def plot_dice_curves(data, save_dir):
    """Figure 2: Per-class dice scores over epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    epochs = data["epoch_dice_0"]["step"]
    class_names = [
        "Class 1 (Heavy)",
        "Class 2 (Crazing)",
        "Class 3 (Rolled-in)",
        "Class 4 (Pitted)",
    ]
    colors = [COLORS["class0"], COLORS["class1"], COLORS["class2"], COLORS["class3"]]

    # Left: Training dice
    ax = axes[0]
    for i, (key, name, color) in enumerate(
        zip(
            ["epoch_dice_0", "epoch_dice_1", "epoch_dice_2", "epoch_dice_3"],
            class_names,
            colors,
        )
    ):
        if key in data:
            ax.plot(
                data[key]["step"],
                data[key]["value"],
                color=color,
                label=name,
                linewidth=1.8,
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice Score")
    ax.set_title("Training Dice Scores", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)

    # Right: Validation dice
    ax = axes[1]
    for i, (key, name, color) in enumerate(
        zip(
            [
                "epoch_val_dice_0",
                "epoch_val_dice_1",
                "epoch_val_dice_2",
                "epoch_val_dice_3",
            ],
            class_names,
            colors,
        )
    ):
        if key in data:
            ax.plot(
                data[key]["step"],
                data[key]["value"],
                color=color,
                label=name,
                linewidth=1.8,
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice Score")
    ax.set_title("Validation Dice Scores", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "fig2_dice_per_class.png", dpi=300, bbox_inches="tight")
    plt.savefig(save_dir / "fig2_dice_per_class.pdf", bbox_inches="tight")
    print(f"✓ Saved fig2_dice_per_class.png/pdf")
    plt.close()


# ─── Plot 3: Mean Dice Score (Train vs Val) ───
def plot_mean_dice(data, save_dir):
    """Figure 3: Mean dice score comparison."""
    fig, ax = plt.subplots(figsize=(6, 4))

    if "epoch_dice_mean" in data:
        ax.plot(
            data["epoch_dice_mean"]["step"],
            data["epoch_dice_mean"]["value"],
            color=COLORS["train"],
            label="Train Mean Dice",
            linewidth=2,
        )

    if "epoch_val_dice_mean" in data:
        ax.plot(
            data["epoch_val_dice_mean"]["step"],
            data["epoch_val_dice_mean"]["value"],
            color=COLORS["val"],
            label="Val Mean Dice",
            linewidth=2,
        )

        # Mark final values
        final_val = data["epoch_val_dice_mean"]["value"][-1]
        final_epoch = data["epoch_val_dice_mean"]["step"][-1]
        ax.annotate(
            f"Final: {final_val:.3f}",
            xy=(final_epoch, final_val),
            xytext=(final_epoch - 20, final_val - 0.05),
            arrowprops=dict(arrowstyle="->", color=COLORS["val"]),
            fontsize=9,
            color=COLORS["val"],
        )

    ax.axhline(
        y=0.9,
        color="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label="90% threshold",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Dice Score")
    ax.set_title("Figure 3: Mean Dice Score (All Classes)", fontweight="bold", pad=10)
    ax.legend(frameon=False)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "fig3_mean_dice.png", dpi=300, bbox_inches="tight")
    plt.savefig(save_dir / "fig3_mean_dice.pdf", bbox_inches="tight")
    print(f"✓ Saved fig3_mean_dice.png/pdf")
    plt.close()


# ─── Plot 4: Validation Dice per Class (Bar Chart) ───
def plot_final_dice_bar(data, save_dir):
    """Figure 4: Final validation dice score per class (bar chart)."""
    classes = [
        "Class 1\n(Heavy)",
        "Class 2\n(Crazing)",
        "Class 3\n(Rolled-in)",
        "Class 4\n(Pitted)",
    ]

    final_vals = []
    for key in [
        "epoch_val_dice_0",
        "epoch_val_dice_1",
        "epoch_val_dice_2",
        "epoch_val_dice_3",
    ]:
        if key in data:
            final_vals.append(data[key]["value"][-1])
        else:
            final_vals.append(0)

    mean_val = np.mean(final_vals)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        classes,
        final_vals,
        color=[COLORS["class0"], COLORS["class1"], COLORS["class2"], COLORS["class3"]],
        edgecolor="white",
        linewidth=1.5,
        width=0.6,
    )

    # Value labels on bars
    for bar, val in zip(bars, final_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Mean line
    ax.axhline(
        y=mean_val,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {mean_val:.3f}",
    )

    ax.set_ylabel("Validation Dice Score")
    ax.set_title(
        "Figure 4: Per-Class Validation Dice Score (Final Epoch)",
        fontweight="bold",
        pad=10,
    )
    ax.set_ylim([0, 1.15])
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_dir / "fig4_final_dice_bar.png", dpi=300, bbox_inches="tight")
    plt.savefig(save_dir / "fig4_final_dice_bar.pdf", bbox_inches="tight")
    print(f"✓ Saved fig4_final_dice_bar.png/pdf")
    plt.close()


# ─── Plot 5: Convergence Behavior (Smoothed Loss) ───
def plot_convergence(data, save_dir):
    """Figure 5: Smoothed loss showing convergence rate."""
    from scipy.ndimage import uniform_filter1d

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: Raw vs smoothed train loss
    ax = axes[0]
    epochs = data["epoch_loss"]["step"]
    raw_loss = data["epoch_loss"]["value"]
    smoothed = uniform_filter1d(raw_loss, size=5)

    ax.plot(epochs, raw_loss, color="lightgray", linewidth=0.8, alpha=0.7, label="Raw")
    ax.plot(
        epochs,
        smoothed,
        color=COLORS["train"],
        linewidth=2.5,
        label="Smoothed (5-ep MA)",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Convergence", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    # Right: Loss improvement rate
    ax = axes[1]
    loss_reduction = [(raw_loss[0] - v) / raw_loss[0] * 100 for v in raw_loss]
    ax.fill_between(epochs, loss_reduction, alpha=0.3, color=COLORS["train"])
    ax.plot(epochs, loss_reduction, color=COLORS["train"], linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss Reduction (%)")
    ax.set_title("Cumulative Loss Reduction from Epoch 0", fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "fig5_convergence.png", dpi=300, bbox_inches="tight")
    plt.savefig(save_dir / "fig5_convergence.pdf", bbox_inches="tight")
    print(f"✓ Saved fig5_convergence.png/pdf")
    plt.close()


# ─── Plot 6: Class Difficulty Heatmap ───
def plot_class_difficulty(data, save_dir):
    """Figure 6: Heatmap showing learning difficulty per class over epochs."""
    import seaborn as sns

    # Collect val dice per class at key epochs
    val_epochs = data["epoch_val_dice_0"]["step"]
    n_epochs = len(val_epochs)
    n_classes = 4

    dice_matrix = np.zeros((n_classes, n_epochs))
    for col, key in enumerate(
        ["epoch_val_dice_0", "epoch_val_dice_1", "epoch_val_dice_2", "epoch_val_dice_3"]
    ):
        if key in data:
            dice_matrix[col] = data[key]["value"]

    # Sample every N epochs for readable heatmap
    sample_rate = max(1, n_epochs // 50)
    dice_sampled = dice_matrix[:, ::sample_rate]
    epoch_sampled = val_epochs[::sample_rate]

    class_labels = ["Class 1", "Class 2", "Class 3", "Class 4"]

    fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow(dice_sampled, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_yticks(range(4))
    ax.set_yticklabels(class_labels)
    ax.set_xlabel("Epoch")
    ax.set_xticks(range(len(epoch_sampled)))
    ax.set_xticklabels([str(e) for e in epoch_sampled], rotation=45, fontsize=8)
    ax.set_title(
        "Figure 6: Per-Class Validation Dice Over Training", fontweight="bold", pad=10
    )

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Dice Score")

    plt.tight_layout()
    plt.savefig(
        save_dir / "fig6_class_difficulty_heatmap.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(save_dir / "fig6_class_difficulty_heatmap.pdf", bbox_inches="tight")
    print(f"✓ Saved fig6_class_difficulty_heatmap.png/pdf")
    plt.close()


# ─── Plot 7: Multi-panel Summary Figure ───
def plot_summary_figure(data, save_dir):
    """Figure 7: Multi-panel summary figure for paper."""
    from string import ascii_uppercase

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)

    epochs = data["epoch_loss"]["step"]

    # Panel A: Loss
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(epochs, data["epoch_loss"]["value"], color=COLORS["train"], linewidth=1.5)
    if "epoch_val_loss" in data:
        ax.plot(
            data["epoch_val_loss"]["step"],
            data["epoch_val_loss"]["value"],
            color=COLORS["val"],
            linewidth=1.5,
        )
        ax.axvline(
            x=min(data["epoch_val_loss"]["step"]),
            color="gray",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("A. Loss Curves", fontweight="bold", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.text(
        -0.1,
        1.05,
        "A",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )

    # Panel B: Mean Dice
    ax = fig.add_subplot(gs[0, 1])
    if "epoch_dice_mean" in data:
        ax.plot(
            data["epoch_dice_mean"]["step"],
            data["epoch_dice_mean"]["value"],
            color=COLORS["train"],
            linewidth=1.5,
            label="Train",
        )
    if "epoch_val_dice_mean" in data:
        ax.plot(
            data["epoch_val_dice_mean"]["step"],
            data["epoch_val_dice_mean"]["value"],
            color=COLORS["val"],
            linewidth=1.5,
            label="Val",
        )
    ax.axhline(y=0.9, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Dice")
    ax.set_title("B. Mean Dice Score", fontweight="bold", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    ax.text(
        -0.1,
        1.05,
        "B",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )

    # Panel C: Final Val Dice Bar
    ax = fig.add_subplot(gs[0, 2])
    final_vals = []
    for key in [
        "epoch_val_dice_0",
        "epoch_val_dice_1",
        "epoch_val_dice_2",
        "epoch_val_dice_3",
    ]:
        if key in data:
            final_vals.append(data[key]["value"][-1])
        else:
            final_vals.append(0)

    bars = ax.bar(
        ["C1", "C2", "C3", "C4"],
        final_vals,
        color=[COLORS["class0"], COLORS["class1"], COLORS["class2"], COLORS["class3"]],
        edgecolor="white",
        linewidth=1,
        width=0.6,
    )
    for bar, val in zip(bars, final_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.axhline(
        y=np.mean(final_vals),
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Mean: {np.mean(final_vals):.3f}",
    )
    ax.set_ylabel("Val Dice")
    ax.set_title("C. Final Val Dice by Class", fontweight="bold", fontsize=10)
    ax.set_ylim([0, 1.1])
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.text(
        -0.1,
        1.05,
        "C",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )

    # Panel D: Per-class train dice
    ax = fig.add_subplot(gs[1, 0])
    for i, (key, color) in enumerate(
        zip(
            ["epoch_dice_0", "epoch_dice_1", "epoch_dice_2", "epoch_dice_3"],
            [COLORS["class0"], COLORS["class1"], COLORS["class2"], COLORS["class3"]],
        )
    ):
        if key in data:
            ax.plot(
                data[key]["step"],
                data[key]["value"],
                color=color,
                linewidth=1.5,
                label=f"Class {i + 1}",
            )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Dice")
    ax.set_title("D. Training Dice by Class", fontweight="bold", fontsize=10)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    ax.text(
        -0.1,
        1.05,
        "D",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )

    # Panel E: Per-class val dice
    ax = fig.add_subplot(gs[1, 1])
    for i, (key, color) in enumerate(
        zip(
            [
                "epoch_val_dice_0",
                "epoch_val_dice_1",
                "epoch_val_dice_2",
                "epoch_val_dice_3",
            ],
            [COLORS["class0"], COLORS["class1"], COLORS["class2"], COLORS["class3"]],
        )
    ):
        if key in data:
            ax.plot(
                data[key]["step"],
                data[key]["value"],
                color=color,
                linewidth=1.5,
                label=f"Class {i + 1}",
            )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Dice")
    ax.set_title("E. Validation Dice by Class", fontweight="bold", fontsize=10)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    ax.text(
        -0.1,
        1.05,
        "E",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )

    # Panel F: Results table text
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")

    final_vals = []
    for key in [
        "epoch_val_dice_0",
        "epoch_val_dice_1",
        "epoch_val_dice_2",
        "epoch_val_dice_3",
    ]:
        if key in data:
            final_vals.append(data[key]["value"][-1])

    final_dice_mean = (
        data["epoch_val_dice_mean"]["value"][-1]
        if "epoch_val_dice_mean" in data
        else np.mean(final_vals)
    )
    final_val_loss = (
        data["epoch_val_loss"]["value"][-1] if "epoch_val_loss" in data else 0
    )
    total_epochs = epochs[-1] if len(epochs) > 0 else 0

    table_data = [
        ["Metric", "Value"],
        ["Total Epochs", str(total_epochs)],
        ["Val Loss (final)", f"{final_val_loss:.4f}"],
        ["Val Dice Mean", f"{final_dice_mean:.4f}"],
        ["Class 1 (Heavy)", f"{final_vals[0]:.4f}"],
        ["Class 2 (Crazing)", f"{final_vals[1]:.4f}"],
        ["Class 3 (Rolled-in)", f"{final_vals[2]:.4f}"],
        ["Class 4 (Pitted)", f"{final_vals[3]:.4f}"],
    ]

    table = ax.table(
        cellText=table_data, loc="center", cellLoc="center", colWidths=[0.5, 0.4]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)

    # Header styling
    for j in range(2):
        table[(0, j)].set_facecolor("#0072B2")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    ax.set_title("F. Final Results Summary", fontweight="bold", fontsize=10, pad=20)
    ax.text(
        -0.1,
        1.05,
        "F",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )

    plt.suptitle(
        "Baseline U-Net (EfficientNet-B5) — Training Results",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.savefig(save_dir / "fig7_summary.png", dpi=300, bbox_inches="tight")
    plt.savefig(save_dir / "fig7_summary.pdf", bbox_inches="tight")
    print(f"✓ Saved fig7_summary.png/pdf")
    plt.close()


# ─── Main ───
def main():
    log_dirs = [
        "/workspace/code/runs",
        "/workspace/code/saved/sever-rtx5090-Unet-efficientnet-b5-BCEDiceLoss-RAdam",
    ]

    log_dir = None
    for ld in log_dirs:
        if os.path.exists(ld):
            log_dir = ld
            break

    if log_dir is None:
        print("ERROR: No TensorBoard log directory found.")
        print("Tried:", log_dirs)
        return

    print(f"Loading data from: {log_dir}")
    data = load_tensorboard_data(log_dir)
    print(f"Loaded {len(data)} metrics: {list(data.keys())}")

    # Generate all figures
    print("\nGenerating figures...")
    plot_loss_curves(data, OUTPUT_DIR)
    plot_dice_curves(data, OUTPUT_DIR)
    plot_mean_dice(data, OUTPUT_DIR)
    plot_final_dice_bar(data, OUTPUT_DIR)
    plot_convergence(data, OUTPUT_DIR)
    plot_class_difficulty(data, OUTPUT_DIR)
    plot_summary_figure(data, OUTPUT_DIR)

    # Save data as CSV
    import csv

    csv_path = OUTPUT_DIR / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "epoch", "value"])
        for metric_name, metric_data in data.items():
            for step, value in zip(metric_data["step"], metric_data["value"]):
                writer.writerow([metric_name, step, value])
    print(f"✓ Saved metrics.csv")

    # Print summary
    print(f"\n{'=' * 60}")
    print("GENERATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"\nFigures generated:")
    for f in sorted(OUTPUT_DIR.glob("fig*.png")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name} ({size_kb:.0f} KB)")

    # Final metrics
    if "epoch_val_dice_mean" in data:
        final = data["epoch_val_dice_mean"]["value"][-1]
        print(f"\nFinal Val Dice Mean: {final:.4f}")

    print(f"\nOpen in browser to view: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
