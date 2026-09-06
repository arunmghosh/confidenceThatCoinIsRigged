import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import beta


class BrierScoreLoss(nn.Module):
    """
    Brier Score Loss:
    L = (1/N) * sum_i (p_hat_i - y_i)^2
    where p_hat_i is predicted probability in [0, 1], and y_i in {0, 1}.
    Strictly proper scoring rule for probabilistic calibration.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred_prob, target):
        pred_prob = pred_prob.view(-1)
        target = target.view(-1).float()
        return torch.mean((pred_prob - target) ** 2)


class ModelA_FullHistory(nn.Module):
    """
    Model A: Has access to the entire history of flips (including order).
    Processes variable-length binary sequences up to N=500.
    Architecture:
      - 2 input channels: (flip values, validity mask)
      - 1D Convolutions with kernel sizes 3 and 5 to capture sequence order, transitions, and runs
      - Masked Dual Global Pooling (Mean + Max)
      - Dense MLP with Sigmoid output.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.fc = nn.Sequential(
            nn.Linear(64 * 2 + 1, 64),  # mean + max pooling + normalized length
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x, lengths):
        """
        x: (batch_size, max_seq_len, 1) or (batch_size, max_seq_len) float tensor
        lengths: (batch_size,) int tensor of actual trial lengths
        """
        if x.dim() == 3 and x.size(-1) == 1:
            x = x.squeeze(-1)
        batch_size, max_len = x.shape
        device = x.device

        mask = (torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)).float()
        inp = torch.stack([x * mask, mask], dim=1)  # (B, 2, L)

        h = F.relu(self.conv1(inp))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))  # (B, 64, L)

        # Masked Average Pooling
        mask_exp = mask.unsqueeze(1)  # (B, 1, L)
        sum_pooled = torch.sum(h * mask_exp, dim=2)  # (B, 64)
        lengths_clamped = torch.clamp(lengths.unsqueeze(1).float(), min=1.0)
        avg_pooled = sum_pooled / lengths_clamped

        # Masked Max Pooling
        h_masked = h.masked_fill(mask_exp == 0, -1e9)
        max_pooled = torch.max(h_masked, dim=2)[0]

        norm_len = lengths.unsqueeze(1).float() / 500.0
        features = torch.cat([avg_pooled, max_pooled, norm_len], dim=1)
        prob = self.fc(features)
        return prob


class ModelB_Last20(nn.Module):
    """
    Model B: Has access ONLY to the last 20 flips (including order).
    Input:
      - Window of 20 flips (0 or 1)
      - Mask indicating how many valid flips are in the window (for N < 20)
    Architecture:
      - 1D Convolution + Dense layers with Sigmoid output.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(64 + 1, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x_last20, valid_counts):
        """
        x_last20: (batch_size, 20) float tensor
        valid_counts: (batch_size,) float tensor
        """
        mask = torch.arange(20, device=x_last20.device).unsqueeze(0) >= (20 - valid_counts.unsqueeze(1))
        mask = mask.float()

        inp = torch.stack([x_last20 * mask, mask], dim=1)  # (B, 2, 20)
        h = F.relu(self.conv1(inp))
        h = F.relu(self.conv2(h))
        pooled = self.pool(h).squeeze(-1)  # (B, 64)

        norm_counts = (valid_counts / 20.0).unsqueeze(1)
        features = torch.cat([pooled, norm_counts], dim=1)
        prob = self.fc(features)
        return prob


class ModelC_SummaryStats(nn.Module):
    """
    Model C: Has access ONLY to summary statistics:
      - % heads (k / N)
      - total number of flips (N)
    Architecture:
      - Dense MLP with Sigmoid output.
    """
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, head_ratio, total_flips):
        """
        head_ratio: (batch_size, 1) float
        total_flips: (batch_size, 1) float
        """
        norm_n = total_flips / 500.0
        se = 0.5 / torch.sqrt(torch.clamp(total_flips, min=1.0))
        features = torch.cat([head_ratio, norm_n, se], dim=1)
        prob = self.fc(features)
        return prob


def bayesian_posterior_prob_rigged(k, n, prior_alpha=1.0, prior_beta=1.0):
    """
    Exact analytical Bayesian probability that P(p > 0.5 | k heads out of n flips).
    Under Beta(alpha, beta) prior on p, the posterior is Beta(alpha + k, beta + n - k).
    P(p > 0.5 | data) = 1 - Beta_CDF(0.5; alpha + k, beta + n - k).
    """
    post_alpha = prior_alpha + k
    post_beta = prior_beta + (n - k)
    prob = 1.0 - beta.cdf(0.5, post_alpha, post_beta)
    return prob
