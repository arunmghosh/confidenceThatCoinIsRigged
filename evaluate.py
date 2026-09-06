import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import numpy as np
import pandas as pd
import torch

from models import (
    ModelA_FullHistory,
    ModelB_Last20,
    ModelC_SummaryStats,
    BrierScoreLoss,
    bayesian_posterior_prob_rigged
)
from data import FLIP_COUNTS, EVAL_PROBABILITIES, generate_evaluation_batch

def evaluate_models(num_trials=500, checkpoint_dir="checkpoints"):
    os.makedirs("results", exist_ok=True)
    device = torch.device("cpu")

    print("Loading models...")
    model_a = ModelA_FullHistory().to(device)
    model_b = ModelB_Last20().to(device)
    model_c = ModelC_SummaryStats().to(device)

    model_a.load_state_dict(torch.load(os.path.join(checkpoint_dir, "model_a.pth"), map_location=device))
    model_b.load_state_dict(torch.load(os.path.join(checkpoint_dir, "model_b.pth"), map_location=device))
    model_c.load_state_dict(torch.load(os.path.join(checkpoint_dir, "model_c.pth"), map_location=device))

    model_a.eval()
    model_b.eval()
    model_c.eval()

    criterion = BrierScoreLoss()

    records = []
    detailed_results = {}

    print(f"\nRunning evaluation across {len(EVAL_PROBABILITIES)} probabilities and {len(FLIP_COUNTS)} trial lengths ({num_trials} trials each)...")

    with torch.no_grad():
        for p in EVAL_PROBABILITIES:
            p_key = f"p_{p:.2f}"
            detailed_results[p_key] = {}

            for n in FLIP_COUNTS:
                n_key = f"n_{n}"
                seed = 10000 + int(p * 1000) + n
                batch = generate_evaluation_batch(p, n, num_trials=num_trials, seed=seed)

                labels = batch["labels"].to(device)
                target_val = 1.0 if p > 0.5 else 0.0

                # Model A
                pred_a = model_a(batch["x_a"].to(device), batch["lengths_a"].to(device)).cpu().numpy().flatten()
                brier_a = float(np.mean((pred_a - target_val) ** 2))

                # Model B
                pred_b = model_b(batch["x_b"].to(device), batch["valid_b"].to(device)).cpu().numpy().flatten()
                brier_b = float(np.mean((pred_b - target_val) ** 2))

                # Model C
                pred_c = model_c(batch["head_ratio_c"].to(device), batch["total_flips_c"].to(device)).cpu().numpy().flatten()
                brier_c = float(np.mean((pred_c - target_val) ** 2))

                # Bayesian Benchmark
                pred_bayes = bayesian_posterior_prob_rigged(batch["k_counts"], n)
                brier_bayes = float(np.mean((pred_bayes - target_val) ** 2))

                cell_stats = {
                    "p": p,
                    "n": n,
                    "Model_A": {
                        "mean": float(np.mean(pred_a)),
                        "std": float(np.std(pred_a)),
                        "se": float(np.std(pred_a) / np.sqrt(num_trials)),
                        "median": float(np.median(pred_a)),
                        "p10": float(np.percentile(pred_a, 10)),
                        "p90": float(np.percentile(pred_a, 90)),
                        "brier": brier_a
                    },
                    "Model_B": {
                        "mean": float(np.mean(pred_b)),
                        "std": float(np.std(pred_b)),
                        "se": float(np.std(pred_b) / np.sqrt(num_trials)),
                        "median": float(np.median(pred_b)),
                        "p10": float(np.percentile(pred_b, 10)),
                        "p90": float(np.percentile(pred_b, 90)),
                        "brier": brier_b
                    },
                    "Model_C": {
                        "mean": float(np.mean(pred_c)),
                        "std": float(np.std(pred_c)),
                        "se": float(np.std(pred_c) / np.sqrt(num_trials)),
                        "median": float(np.median(pred_c)),
                        "p10": float(np.percentile(pred_c, 10)),
                        "p90": float(np.percentile(pred_c, 90)),
                        "brier": brier_c
                    },
                    "Bayesian_Benchmark": {
                        "mean": float(np.mean(pred_bayes)),
                        "std": float(np.std(pred_bayes)),
                        "se": float(np.std(pred_bayes) / np.sqrt(num_trials)),
                        "median": float(np.median(pred_bayes)),
                        "p10": float(np.percentile(pred_bayes, 10)),
                        "p90": float(np.percentile(pred_bayes, 90)),
                        "brier": brier_bayes
                    }
                }
                detailed_results[p_key][n_key] = cell_stats

                # Flatten into tabular record
                records.append({
                    "p_heads": p,
                    "flips_N": n,
                    "Model_A_Mean_Conf": cell_stats["Model_A"]["mean"],
                    "Model_A_SE": cell_stats["Model_A"]["se"],
                    "Model_A_Brier": cell_stats["Model_A"]["brier"],
                    "Model_B_Mean_Conf": cell_stats["Model_B"]["mean"],
                    "Model_B_SE": cell_stats["Model_B"]["se"],
                    "Model_B_Brier": cell_stats["Model_B"]["brier"],
                    "Model_C_Mean_Conf": cell_stats["Model_C"]["mean"],
                    "Model_C_SE": cell_stats["Model_C"]["se"],
                    "Model_C_Brier": cell_stats["Model_C"]["brier"],
                    "Bayes_Mean_Conf": cell_stats["Bayesian_Benchmark"]["mean"],
                    "Bayes_Brier": cell_stats["Bayesian_Benchmark"]["brier"]
                })

    df = pd.DataFrame(records)
    df.to_csv("results/evaluation_summary.csv", index=False)

    with open("results/evaluation_results.json", "w") as f:
        json.dump(detailed_results, f, indent=2)

    print("\nEvaluation complete! Results saved to:")
    print("  - results/evaluation_summary.csv")
    print("  - results/evaluation_results.json")

    print("\n--- SAMPLE RESULTS PREVIEW (Mean Confidence That P(heads) > 0.5) ---")
    preview_df = df[["p_heads", "flips_N", "Model_A_Mean_Conf", "Model_B_Mean_Conf", "Model_C_Mean_Conf", "Bayes_Mean_Conf"]]
    print(preview_df.head(14).to_string(index=False))

    return df, detailed_results

if __name__ == "__main__":
    evaluate_models()
