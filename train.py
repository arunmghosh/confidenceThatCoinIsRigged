import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models import BrierScoreLoss, ModelA_FullHistory, ModelB_Last20, ModelC_SummaryStats
from data import FastCoinFlipDataset, collate_coin_flips

def train():
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device for training: {device}", flush=True)

    print("Generating training dataset (30,000 samples)...", flush=True)
    train_dataset = FastCoinFlipDataset(num_samples=30000, seed=42)
    print("Generating validation dataset (4,000 samples)...", flush=True)
    val_dataset = FastCoinFlipDataset(num_samples=4000, seed=1234)

    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        collate_fn=collate_coin_flips,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=512,
        shuffle=False,
        collate_fn=collate_coin_flips,
        num_workers=0
    )

    model_a = ModelA_FullHistory().to(device)
    model_b = ModelB_Last20().to(device)
    model_c = ModelC_SummaryStats().to(device)

    criterion = BrierScoreLoss()

    optimizer_a = optim.AdamW(model_a.parameters(), lr=2e-3, weight_decay=1e-4)
    optimizer_b = optim.AdamW(model_b.parameters(), lr=2e-3, weight_decay=1e-4)
    optimizer_c = optim.AdamW(model_c.parameters(), lr=3e-3, weight_decay=1e-4)

    num_epochs = 12
    history = {
        "epoch": [],
        "loss_a": [], "loss_b": [], "loss_c": [],
        "val_brier_a": [], "val_brier_b": [], "val_brier_c": []
    }

    best_val_brier_a = float("inf")
    best_val_brier_b = float("inf")
    best_val_brier_c = float("inf")

    print("\nStarting training across Models A, B, and C with Brier Score Loss...", flush=True)
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        model_a.train()
        model_b.train()
        model_c.train()

        total_loss_a = 0.0
        total_loss_b = 0.0
        total_loss_c = 0.0
        num_batches = 0

        for batch in train_loader:
            labels = batch["labels"].to(device)

            # Model A
            optimizer_a.zero_grad()
            pred_a = model_a(batch["x_a"].to(device), batch["lengths_a"].to(device))
            loss_a = criterion(pred_a, labels)
            loss_a.backward()
            optimizer_a.step()

            # Model B
            optimizer_b.zero_grad()
            pred_b = model_b(batch["x_b"].to(device), batch["valid_b"].to(device))
            loss_b = criterion(pred_b, labels)
            loss_b.backward()
            optimizer_b.step()

            # Model C
            optimizer_c.zero_grad()
            pred_c = model_c(batch["head_ratio_c"].to(device), batch["total_flips_c"].to(device))
            loss_c = criterion(pred_c, labels)
            loss_c.backward()
            optimizer_c.step()

            total_loss_a += loss_a.item()
            total_loss_b += loss_b.item()
            total_loss_c += loss_c.item()
            num_batches += 1

        avg_loss_a = total_loss_a / num_batches
        avg_loss_b = total_loss_b / num_batches
        avg_loss_c = total_loss_c / num_batches

        # Validation
        model_a.eval()
        model_b.eval()
        model_c.eval()

        val_brier_a = 0.0
        val_brier_b = 0.0
        val_brier_c = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                labels = batch["labels"].to(device)

                pred_a = model_a(batch["x_a"].to(device), batch["lengths_a"].to(device))
                pred_b = model_b(batch["x_b"].to(device), batch["valid_b"].to(device))
                pred_c = model_c(batch["head_ratio_c"].to(device), batch["total_flips_c"].to(device))

                val_brier_a += criterion(pred_a, labels).item()
                val_brier_b += criterion(pred_b, labels).item()
                val_brier_c += criterion(pred_c, labels).item()
                val_batches += 1

        val_brier_a /= val_batches
        val_brier_b /= val_batches
        val_brier_c /= val_batches

        history["epoch"].append(epoch)
        history["loss_a"].append(avg_loss_a)
        history["loss_b"].append(avg_loss_b)
        history["loss_c"].append(avg_loss_c)
        history["val_brier_a"].append(val_brier_a)
        history["val_brier_b"].append(val_brier_b)
        history["val_brier_c"].append(val_brier_c)

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch:02d}/{num_epochs:02d} [{epoch_time:.1f}s] | "
            f"Val Brier -> Model A: {val_brier_a:.4f} | Model B: {val_brier_b:.4f} | Model C: {val_brier_c:.4f}",
            flush=True
        )

        if val_brier_a < best_val_brier_a:
            best_val_brier_a = val_brier_a
            torch.save(model_a.state_dict(), "checkpoints/model_a.pth")

        if val_brier_b < best_val_brier_b:
            best_val_brier_b = val_brier_b
            torch.save(model_b.state_dict(), "checkpoints/model_b.pth")

        if val_brier_c < best_val_brier_c:
            best_val_brier_c = val_brier_c
            torch.save(model_c.state_dict(), "checkpoints/model_c.pth")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s.", flush=True)
    print(f"Best Validation Brier Scores: Model A: {best_val_brier_a:.4f}, Model B: {best_val_brier_b:.4f}, Model C: {best_val_brier_c:.4f}", flush=True)

    with open("results/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("Checkpoints and training history saved successfully.", flush=True)

if __name__ == "__main__":
    train()
