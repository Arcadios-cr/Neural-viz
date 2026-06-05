"""
Étude de l'effet de la DENSITÉ de points sur la généralisation.

On entraîne le même réseau sur des jeux d'entraînement de tailles croissantes
et on mesure la performance sur un test set fixe de 1000 points. Cela produit
une « courbe d'apprentissage » : plus on a de données, plus l'écart train↔test
(overfitting) se résorbe.

Config : Overlap (bruit 0.5), MLP 4×32 ReLU, Adam lr=0.01, 200 époques, sans
dropout, seed des poids = 0.

Génère : experiments/figures/density_learning_curve.png
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


def main():
    test_loader = fixed_test_loader()
    sizes = [30, 60, 120, 250, 500, 1000]
    acc_tr, acc_te = [], []

    print("Densité de points (Overlap, 4x32, sans dropout) — test sur set fixe 1000 pts")
    for n in sizes:
        X_tr, y_tr = get_dataset("Overlap", n_samples=n, noise=0.5, seed=42)
        a, b = train_and_eval(X_tr, y_tr, test_loader, [32, 32, 32, 32])
        acc_tr.append(a * 100)
        acc_te.append(b * 100)
        print(f"  n_train={n:4d}: train={a*100:5.1f}%  test={b*100:5.1f}%  écart={(a-b)*100:+5.1f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, acc_tr, "o-", color="#F44336", label="train", lw=2)
    ax.plot(sizes, acc_te, "o-", color="#2196F3", label="test (set fixe)", lw=2)
    ax.fill_between(sizes, acc_te, acc_tr, color="gray", alpha=0.18, label="écart = overfitting")
    ax.set_xscale("log")
    ax.set_xlabel("Nombre de points d'entraînement (échelle log)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Densité de points vs généralisation (Overlap)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = os.path.join(OUT, "density_learning_curve.png")
    fig.savefig(path, dpi=105)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
