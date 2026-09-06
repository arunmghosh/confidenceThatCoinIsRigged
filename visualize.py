import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Style configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

COLORS = {
    "Model_A": "#2563EB",       # Vibrant Royal Blue
    "Model_B": "#EA580C",       # Vibrant Orange / Amber
    "Model_C": "#059669",       # Emerald Green
    "Bayes": "#475569"          # Slate
}

def generate_all_plots(results_file="results/evaluation_results.json", summary_csv="results/evaluation_summary.csv"):
    os.makedirs("figures", exist_ok=True)

    with open(results_file, "r") as f:
        data = json.load(f)

    df = pd.read_csv(summary_csv)
    p_values = sorted(df["p_heads"].unique())
    n_values = sorted(df["flips_N"].unique())

    print("Generating Figure 1: Confidence vs Riggedness across Sample Sizes...")
    plot_confidence_vs_riggedness(df, p_values, n_values)

    print("Generating Figure 2: Confidence vs Sample Size across Riggedness Levels...")
    plot_confidence_vs_sample_size(df, p_values, n_values)

    print("Generating Figure 3: Comparative 2D Heatmaps...")
    plot_heatmaps(df, p_values, n_values)

    print("Generating Figure 4: Model B Information Horizon Bottleneck...")
    plot_information_gap(df, p_values, n_values)

    print("Generating Figure 5: Brier Score Error Comparison...")
    plot_brier_scores(df, p_values, n_values)

    print("All static plots successfully generated in 'figures/' directory.")


def plot_confidence_vs_riggedness(df, p_values, n_values):
    """
    Subplots for each N, plotting Confidence vs True P(heads).
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharey=True)
    axes = axes.flatten()

    for i, n in enumerate(n_values):
        ax = axes[i]
        sub = df[df["flips_N"] == n].sort_values("p_heads")

        # Model A
        ax.plot(sub["p_heads"], sub["Model_A_Mean_Conf"], color=COLORS["Model_A"], marker="o", linewidth=2.2, label="Model A (Full History)")
        ax.fill_between(sub["p_heads"], sub["Model_A_Mean_Conf"] - 1.96 * sub["Model_A_SE"], sub["Model_A_Mean_Conf"] + 1.96 * sub["Model_A_SE"], color=COLORS["Model_A"], alpha=0.15)

        # Model B
        ax.plot(sub["p_heads"], sub["Model_B_Mean_Conf"], color=COLORS["Model_B"], marker="s", linewidth=2.2, label="Model B (Last 20 Flips)")
        ax.fill_between(sub["p_heads"], sub["Model_B_Mean_Conf"] - 1.96 * sub["Model_B_SE"], sub["Model_B_Mean_Conf"] + 1.96 * sub["Model_B_SE"], color=COLORS["Model_B"], alpha=0.15)

        # Model C
        ax.plot(sub["p_heads"], sub["Model_C_Mean_Conf"], color=COLORS["Model_C"], marker="^", linewidth=2.2, label="Model C (Summary Stats)")
        ax.fill_between(sub["p_heads"], sub["Model_C_Mean_Conf"] - 1.96 * sub["Model_C_SE"], sub["Model_C_Mean_Conf"] + 1.96 * sub["Model_C_SE"], color=COLORS["Model_C"], alpha=0.15)

        # Bayesian Benchmark
        ax.plot(sub["p_heads"], sub["Bayes_Mean_Conf"], color=COLORS["Bayes"], linestyle="--", linewidth=1.8, label="Bayesian Benchmark (Optimal)")

        ax.axhline(0.5, color="#94A3B8", linestyle=":", linewidth=1.2, alpha=0.8)
        ax.axvline(0.5, color="#94A3B8", linestyle=":", linewidth=1.2, alpha=0.8)

        ax.set_title(f"Trial Length N = {n} flips", fontsize=13, fontweight="bold", pad=8)
        ax.set_xlabel("True P(Heads)", fontsize=11)
        if i % 4 == 0:
            ax.set_ylabel("Predicted P(Coin Rigged)", fontsize=11)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xticks(p_values)
        ax.grid(True, linestyle="--", alpha=0.5)

    # Use the 8th subplot for a clean explanatory legend & summary card
    ax_legend = axes[7]
    ax_legend.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    ax_legend.legend(handles, labels, loc="center", fontsize=11, frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1", title="Model Architectures", title_fontsize=12)

    fig.suptitle("AI Confidence in Rigged Coin Detection vs. True Coin Bias P(Heads)\nAcross Flips per Trial (N)", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("figures/confidence_vs_riggedness.png", dpi=300)
    plt.close()


def plot_confidence_vs_sample_size(df, p_values, n_values):
    """
    Subplots for each bias p, showing how confidence scales as sample size N grows.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
    axes = axes.flatten()

    for i, p in enumerate(p_values):
        ax = axes[i]
        sub = df[df["p_heads"] == p].sort_values("flips_N")

        ax.plot(sub["flips_N"], sub["Model_A_Mean_Conf"], color=COLORS["Model_A"], marker="o", linewidth=2.2, label="Model A (Full History)")
        ax.plot(sub["flips_N"], sub["Model_B_Mean_Conf"], color=COLORS["Model_B"], marker="s", linewidth=2.2, label="Model B (Last 20 Flips)")
        ax.plot(sub["flips_N"], sub["Model_C_Mean_Conf"], color=COLORS["Model_C"], marker="^", linewidth=2.2, label="Model C (Summary Stats)")
        ax.plot(sub["flips_N"], sub["Bayes_Mean_Conf"], color=COLORS["Bayes"], linestyle="--", linewidth=1.8, label="Bayesian Optimal")

        ax.axhline(0.5, color="#94A3B8", linestyle=":", linewidth=1.2, alpha=0.8)

        ax.set_title(f"Coin Riggedness P(Heads) = {p:.2f}" + (" (Fair Coin)" if p == 0.5 else " (Biased)"), fontsize=13, fontweight="bold", pad=8)
        ax.set_xlabel("Number of Flips (N)", fontsize=11)
        if i % 3 == 0:
            ax.set_ylabel("Predicted P(Coin Rigged)", fontsize=11)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xscale("log")
        ax.set_xticks(n_values)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

        if i == 0:
            ax.legend(loc="upper right", fontsize=9, frameon=True)

    fig.suptitle("Scaling of AI Confidence as Sample Size (N) Increases\nNotice Model B Saturation vs Model A/C Scaling", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("figures/confidence_vs_sample_size.png", dpi=300)
    plt.close()


def plot_heatmaps(df, p_values, n_values):
    """
    Side-by-side heatmaps comparing Model A, B, C and Bayesian Optimal.
    """
    models = [
        ("Model_A_Mean_Conf", "Model A: Full History (Order Aware)"),
        ("Model_B_Mean_Conf", "Model B: Last 20 Flips Only"),
        ("Model_C_Mean_Conf", "Model C: Summary Statistics (% Heads, N)"),
        ("Bayes_Mean_Conf", "Theoretical Bayesian Posterior")
    ]

    fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharey=True)

    for idx, (col, title) in enumerate(models):
        ax = axes[idx]
        pivot = df.pivot(index="p_heads", columns="flips_N", values=col)
        # Invert rows so p=0.9 is at top
        pivot = pivot.sort_index(ascending=False)

        im = ax.imshow(pivot.values, cmap="magma", vmin=0.0, vmax=1.0, aspect="auto")

        # Set ticks
        ax.set_xticks(np.arange(len(n_values)))
        ax.set_xticklabels(n_values, fontsize=10)
        ax.set_xlabel("Number of Flips (N)", fontsize=11)

        if idx == 0:
            ax.set_yticks(np.arange(len(p_values)))
            ax.set_yticklabels(sorted(p_values, reverse=True), fontsize=10)
            ax.set_ylabel("True P(Heads)", fontsize=11)

        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

        # Annotate cell values
        for r in range(pivot.shape[0]):
            for c in range(pivot.shape[1]):
                val = pivot.values[r, c]
                text_color = "white" if val < 0.55 else "black"
                ax.text(c, r, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=9, fontweight="bold")

    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.18, 0.015, 0.68])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Predicted Confidence P(Rigged)", fontsize=11)

    fig.suptitle("Confidence Heatmaps: P(Rigged) Across All Experimental Conditions (N × P(heads))", fontsize=15, fontweight="bold", y=0.98)
    plt.savefig("figures/heatmaps_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_information_gap(df, p_values, n_values):
    """
    Plots the penalty incurred by Model B (restricted to last 20 flips)
    compared to Model A/C across different biases as N grows.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for p in [0.55, 0.6, 0.7, 0.8]:
        sub = df[df["p_heads"] == p].sort_values("flips_N")
        gap = sub["Model_A_Mean_Conf"] - sub["Model_B_Mean_Conf"]
        ax.plot(sub["flips_N"], gap, marker="o", linewidth=2.2, label=f"True P(Heads) = {p:.2f}")

    ax.axvline(20, color="#EF4444", linestyle="--", linewidth=1.5, label="Model B Memory Limit (20 Flips)")
    ax.axhline(0.0, color="#94A3B8", linestyle=":", linewidth=1.2)

    ax.set_xscale("log")
    ax.set_xticks(n_values)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel("Trial Length N (Total Flips)", fontsize=12)
    ax.set_ylabel("Information Gap (Model A Conf - Model B Conf)", fontsize=12)
    ax.set_title("Information Bottleneck: Confidence Lost by Truncating to Last 20 Flips", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, frameon=True)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("figures/model_b_penalty.png", dpi=300)
    plt.close()


def plot_brier_scores(df, p_values, n_values):
    """
    Compares the mean Brier score (lower is better) across models.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Average Brier score across all N for each p
    avg_brier = df.groupby("p_heads")[["Model_A_Brier", "Model_B_Brier", "Model_C_Brier", "Bayes_Brier"]].mean().reset_index()

    bar_width = 0.2
    x = np.arange(len(p_values))

    ax.bar(x - 1.5 * bar_width, avg_brier["Model_A_Brier"], width=bar_width, color=COLORS["Model_A"], label="Model A (Full History)")
    ax.bar(x - 0.5 * bar_width, avg_brier["Model_B_Brier"], width=bar_width, color=COLORS["Model_B"], label="Model B (Last 20)")
    ax.bar(x + 0.5 * bar_width, avg_brier["Model_C_Brier"], width=bar_width, color=COLORS["Model_C"], label="Model C (Summary Stats)")
    ax.bar(x + 1.5 * bar_width, avg_brier["Bayes_Brier"], width=bar_width, color=COLORS["Bayes"], label="Bayes Benchmark")

    ax.set_xticks(x)
    ax.set_xticklabels(p_values)
    ax.set_xlabel("True P(Heads)", fontsize=12)
    ax.set_ylabel("Mean Brier Score Loss (Lower is Better)", fontsize=12)
    ax.set_title("Model Calibration & Error (Brier Score) Across Coin Biases", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, frameon=True)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("figures/brier_loss_breakdown.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    generate_all_plots()
