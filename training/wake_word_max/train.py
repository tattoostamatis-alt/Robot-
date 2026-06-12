#!/usr/bin/env python3
"""Train a small binary classifier on openWakeWord (16,96) embedding
windows to detect "Max"/"Μαξ", and export it to ONNX in the same
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


def main():
    data = np.load('features.npz')
    X, y = data['X'], data['y']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val)

    model = WakeWordClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCELoss()

    best_val_loss = float('inf')
    best_state = None
    for epoch in range(300):
        model.train()
        noise = torch.randn_like(X_train_t) * 0.01
        optimizer.zero_grad()
        out = model(X_train_t + noise).squeeze(-1)
        loss = criterion(out, y_train_t)
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

        if epoch % 20 == 0 or epoch == 299:
            print(f'epoch {epoch:3d}: train_loss={loss.item():.4f} '
                  f'val_loss={val_loss:.4f} val_acc={val_acc:.3f}')

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        val_out = model(X_val_t).squeeze(-1)
        val_acc = ((val_out > 0.5).float() == y_val_t).float().mean().item()
        train_out = model(X_train_t).squeeze(-1)
        train_acc = ((train_out > 0.5).float() == y_train_t).float().mean().item()
    print(f'Final: best_val_loss={best_val_loss:.4f} val_acc={val_acc:.3f} train_acc={train_acc:.3f}')

    dummy = torch.zeros(1, 16, 96, dtype=torch.float32)
    torch.onnx.export(model, dummy, 'max.onnx',
                       input_names=['input'], output_names=['output'],
                       opset_version=17)
    print('Exported max.onnx')


if __name__ == '__main__':
    main()
