"""
Étude des paramètres qui maximisent la GÉNÉRALISATION en test.

Trois balayages, chacun mesuré sur un test set fixe de 1000 points :
  1. Capacité du réseau (largeur)        -> figures/capacity.png
  2. Batch size                          -> figures/batch_size.png
  3. Régularisation (dropout vs L2)      -> figures/regularization.png

Base commune : Overlap (bruit 0.5), ReLU, Adam lr=0.01, 200 époques, seed=0.
La taille du train est choisie par expérience pour que l'effet étudié soit
visible (overfitting net quand on veut observer capacité/régularisation).
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import fixed_test_loader, train_and_eval
from data.datasets import get_dataset

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)


def study_capacity(test_loader):
    """Largeur croissante sur peu de points → bias-variance (overfit qui grandit)."""
    X_tr, y_tr = get_dataset("Overlap", n_samples=150, noise=0.5, seed=42)
    widths = [2, 4, 8, 16, 32, 64]
    acc_tr, acc_te = [], []
    print("1) Capacité (Overlap 150 pts, 2 couches, largeur variable)")
    for w in widths:
        a, b = train_and_eval(X_tr, y_tr, test_loader, [w, w])
        acc_tr.append(a * 100)
        acc_te.append(b * 100)
        print(f"  largeur={w:3d}: train={a*100:5.1f}%  test={b*100:5.1f}%  écart={(a-b)*100:+5.1f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(widths, acc_tr, "o-", color="#F44336", label="train", lw=2)
    ax.plot(widths, acc_te, "o-", color="#2196F3", label="test", lw=2)
    ax.fill_between(widths, acc_te, acc_tr, color="gray", alpha=0.18, label="écart")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Largeur (neurones par couche)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Capacité du réseau vs généralisation (Overlap, 150 pts)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "capacity.png"), dpi=105)


def study_batch(test_loader):
    """Batch size variable (effet souvent faible sur petits problèmes 2D)."""
    X_tr, y_tr = get_dataset("Overlap", n_samples=500, noise=0.5, seed=42)
    batches = [8, 16, 32, 64, 128]
    acc_tr, acc_te = [], []
    print("2) Batch size (Overlap 500 pts, 2x32)")
    for bs in batches:
        a, b = train_and_eval(X_tr, y_tr, test_loader, [32, 32], batch_size=bs)
        acc_tr.append(a * 100)
        acc_te.append(b * 100)
        print(f"  batch={bs:3d}: train={a*100:5.1f}%  test={b*100:5.1f}%  écart={(a-b)*100:+5.1f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(batches, acc_tr, "o-", color="#F44336", label="train", lw=2)
    ax.plot(batches, acc_te, "o-", color="#2196F3", label="test", lw=2)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Batch size vs généralisation (Overlap, 500 pts)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "batch_size.png"), dpi=105)


def study_regularization(test_loader):
    """Compare dropout et weight decay sur une config qui overfit."""
    X_tr, y_tr = get_dataset("Overlap", n_samples=100, noise=0.5, seed=42)
    configs = [
        ("Aucune", {}),
        ("Dropout 0.3", {"dropout": 0.3}),
        ("Weight decay 1e-3", {"weight_decay": 1e-3}),
        ("Weight decay 1e-2", {"weight_decay": 1e-2}),
    ]
    labels, tests, gaps = [], [], []
    print("3) Régularisation (Overlap 100 pts, 2x64 — config qui overfit)")
    for name, kw in configs:
        a, b = train_and_eval(X_tr, y_tr, test_loader, [64, 64], **kw)
        labels.append(name)
        tests.append(b * 100)
        gaps.append((a - b) * 100)
        print(f"  {name:18s}: train={a*100:5.1f}%  test={b*100:5.1f}%  écart={(a-b)*100:+5.1f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    xx = np.arange(len(labels))
    ax.bar(xx, tests, color=["#888888", "#2196F3", "#4CAF50", "#9C27B0"])
    for i, (t, g) in enumerate(zip(tests, gaps)):
        ax.text(i, t + 0.4, f"{t:.0f}%\nécart {g:+.0f}", ha="center", fontsize=9)
    ax.set_xticks(xx)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Accuracy test (%)")
    ax.set_ylim(60, 85)
    ax.set_title("Régularisation : dropout vs weight decay (Overlap 100 pts, 2x64)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "regularization.png"), dpi=105)


def main():
    test_loader = fixed_test_loader()
    study_capacity(test_loader)
    study_batch(test_loader)
    study_regularization(test_loader)
    print(f"-> figures dans {OUT}/ (capacity, batch_size, regularization)")


if __name__ == "__main__":
    main()
