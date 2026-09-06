import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from models import BrierScoreLoss, ModelA_FullHistory, ModelB_Last20, ModelC_SummaryStats, bayesian_posterior_prob_rigged
from data import CoinFlipDataset, collate_coin_flips, generate_evaluation_batch

def main():
    print("Testing data generation...")
    dataset = CoinFlipDataset(num_samples=32, seed=1)
    batch = collate_coin_flips([dataset[i] for i in range(16)])
    print("Data shapes:")
    print("  x_a:", batch["x_a"].shape)
    print("  x_b:", batch["x_b"].shape)
    print("  head_ratio_c:", batch["head_ratio_c"].shape)
    print("  labels:", batch["labels"].shape)

    model_a = ModelA_FullHistory()
    model_b = ModelB_Last20()
    model_c = ModelC_SummaryStats()
    criterion = BrierScoreLoss()

    print("Running forward passes...")
    pred_a = model_a(batch["x_a"], batch["lengths_a"])
    pred_b = model_b(batch["x_b"], batch["valid_b"])
    pred_c = model_c(batch["head_ratio_c"], batch["total_flips_c"])

    loss_a = criterion(pred_a, batch["labels"])
    loss_b = criterion(pred_b, batch["labels"])
    loss_c = criterion(pred_c, batch["labels"])

    print(f"Brier loss A: {loss_a.item():.4f}")
    print(f"Brier loss B: {loss_b.item():.4f}")
    print(f"Brier loss C: {loss_c.item():.4f}")

    eval_batch = generate_evaluation_batch(0.7, 50, num_trials=10)
    p_bayes = bayesian_posterior_prob_rigged(eval_batch["k_counts"], 50)
    print("Bayes posterior sample:", p_bayes[:3])
    print("ALL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
