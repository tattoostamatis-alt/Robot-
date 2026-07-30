#!/usr/bin/env python3
"""Train a small binary classifier on openWakeWord (16,96) embedding
windows to detect "Έι ρομπότ"/"Hey robot", and export it to ONNX in the same
input/output shape as openWakeWord's pretrained models ((1,16,96) ->
(1,1) sigmoid score), so it drops straight into `Model(wakeword_model_paths=[...])`.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split


class WakeWordClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 96, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


N_SEEDS = 6      # seed sweep is cheap insurance; mini-batch runs agree closely
N_EPOCHS = 40
BATCH_SIZE = 128


def train_one(seed, X_train_t, y_train_t, X_val_t, y_val_t, clean_val_t, verbose=False):
    """Train one model from a fixed seed; return (best_val_loss, best_state)."""
    torch.manual_seed(seed)
    model = WakeWordClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # Class weighting, added 2026-07-30. The negative set grew to ~6x the
    # positives once 260 sentence-final "…ρομπότ" clips went in; unweighted BCE
    # then buys negative accuracy with positive recall, which is the one thing
    # that must not regress — a wake word you have to repeat is worse than one
    # that occasionally over-fires. Weight each class by its inverse frequency
    # so both sides carry the same total weight regardless of clip counts.
    pos_frac = float(y_train_t.mean())
    w_pos = 0.5 / max(pos_frac, 1e-6)
    w_neg = 0.5 / max(1.0 - pos_frac, 1e-6)

    def criterion(out, target):
        w = target * w_pos + (1.0 - target) * w_neg
        return nn.functional.binary_cross_entropy(out, target, weight=w)

    # Mini-batch, not full-batch: 300 full-batch Adam steps left the model
    # barely converged and wildly seed-dependent (identical data gave anything
    # between "negatives max 0.06" and "negatives max 0.85"). Shuffled batches
    # give thousands of steps per run and land in the same place every time.
    n_train = X_train_t.shape[0]
    best_val_loss = float('inf')
    best_state = None
    for epoch in range(N_EPOCHS):
        model.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb = X_train_t[idx]
            xb = xb + torch.randn_like(xb) * 0.01
            optimizer.zero_grad()
            loss = criterion(model(xb).squeeze(-1), y_train_t[idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(X_val_t).squeeze(-1)
            val_loss = criterion(val_out, y_val_t).item()
            val_acc = ((val_out > 0.5).float() == y_val_t).float().mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose and (epoch % 10 == 0 or epoch == N_EPOCHS - 1):
            print(f'  epoch {epoch:3d}: train_loss={loss.item():.4f} '
                  f'val_loss={val_loss:.4f} val_acc={val_acc:.3f}')

    # Rank runs by what actually matters on the robot, measured on the CLEAN
    # (non-augmented) validation clips — the same population evaluate.py
    # reports: how many cross/miss the 0.5 detection threshold, then the worst
    # negative score, then val_loss. val_loss alone picks models that still
    # fire on near-miss negatives like "my robot" / "το ρομπότ".
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_out = model(X_val_t).squeeze(-1)
    sel = clean_val_t > 0.5
    out_c, y_c = val_out[sel], y_val_t[sel]
    neg, pos = out_c[y_c < 0.5], out_c[y_c > 0.5]
    n_fp = int((neg > 0.5).sum())
    n_fn = int((pos < 0.5).sum())
    worst_neg = float(neg.max())
    # Rank by separation margin, not error counts: the gap between the 10th
    # percentile of positives and the worst negative. Counting errors alone
    # picks over-conservative models that keep every score bunched near the
    # threshold (observed: all positives 0.3-0.7 with negatives up to 0.6),
    # which is fragile once real mic noise widens the distributions.
    margin = float(np.quantile(pos.numpy(), 0.10)) - worst_neg
    return margin, (n_fp, n_fn, worst_neg, best_val_loss), best_state


def main():
    data = np.load('features.npz')
    X, y = data['X'], data['y']
    clean = data['clean'] if 'clean' in data else np.ones_like(y)

    X_train, X_val, y_train, y_val, _, clean_val = train_test_split(
        X, y, clean, test_size=0.2, random_state=42, stratify=y)

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val)
    clean_val_t = torch.from_numpy(clean_val)

    # Seed sweep: a bad init leaves the model keyed on a single word of the
    # phrase (observed: "robot" alone scoring 0.8), which val_loss reliably
    # flags — so train several and keep the lowest-val_loss run.
    best_margin, best_stats, best_state, best_seed = -float('inf'), None, None, None
    for seed in range(N_SEEDS):
        margin, stats, state = train_one(seed, X_train_t, y_train_t, X_val_t, y_val_t,
                                         clean_val_t)
        n_fp, n_fn, worst_neg, vl = stats
        print(f'seed {seed:2d}: margin={margin:+.3f} clean_fp={n_fp} clean_fn={n_fn} '
              f'worst_clean_neg={worst_neg:.2f} val_loss={vl:.4f}')
        if margin > best_margin:
            best_margin, best_stats, best_state, best_seed = margin, stats, state, seed
    best_val_loss = best_stats[3]
    print(f'Picked seed {best_seed}: margin={best_margin:+.3f}')

    model = WakeWordClassifier()
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        val_out = model(X_val_t).squeeze(-1)
        val_acc = ((val_out > 0.5).float() == y_val_t).float().mean().item()
        train_out = model(X_train_t).squeeze(-1)
        train_acc = ((train_out > 0.5).float() == y_train_t).float().mean().item()
    print(f'Final (seed {best_seed}): best_val_loss={best_val_loss:.4f} '
          f'val_acc={val_acc:.3f} train_acc={train_acc:.3f}')

    dummy = torch.zeros(1, 16, 96, dtype=torch.float32)
    torch.onnx.export(model, dummy, 'hey_robot.onnx',
                       input_names=['input'], output_names=['output'],
                       opset_version=17)
    print('Exported hey_robot.onnx')


if __name__ == '__main__':
    main()
