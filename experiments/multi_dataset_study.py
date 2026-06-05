"""
Confirmation MULTI-DATASETS : les conclusions sur la généralisation
(densité, capacité) tiennent-elles au-delà d'Overlap ?

Pour chaque dataset, deux balayages (test mesuré sur un set fixe de 1000 points
du même dataset) :
  - densité  : taille du train variable, réseau fixe (4×32)
  - capacité : largeur du réseau variable, train fixe (300 points)

On s'attend à ce que :
  - « plus de données → meilleure généralisation » tienne PARTOUT ;
  - l'effet de la capacité DÉPENDE de la complexité du dataset :
    datasets simples (Overlap, Moons) → pas d'underfit, overfit qui grandit ;
    datasets complexes (Spirals, Checkerboard) → underfit à faible capacité
    puis amélioration (vraie courbe en U du bias-variance).

Génère : figures/multi_density.png, figures/multi_capacity.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import train_and_eval, fixed_test_loader
from data.datasets import get_dataset

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

# Niveau de bruit adapté à chaque dataset (assez pour créer du chevauchement,
# donc rendre l'overfitting possible et observable).
DATASETS = [
    ("Overlap", 0.5),
    ("Moons", 0.2),
    ("Circles", 0.15),
    ("Spirals", 0.2),
    ("Checkerboard", 0.1),
]


def study_density():
    sizes = [30, 60, 120, 250, 500, 1000]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()
    for ax, (name, noise) in zip(axes, DATASETS):
        tl = fixed_test_loader(name, noise=noise)
        acc_tr, acc_te = [], []
        for n in sizes:
            X_tr, y_tr = get_dataset(name, n_samples=n, noise=noise, seed=42)
            a, b = train_and_eval(X_tr, y_tr, tl, [32, 32, 32, 32])
            acc_tr.append(a * 100)
            acc_te.append(b * 100)
        print(f"  {name:12s} densité  test={[round(x) for x in acc_te]}")
        ax.plot(sizes, acc_tr, "o-", color="#F44336", label="train", lw=2)
        ax.plot(sizes, acc_te, "o-", color="#2196F3", label="test", lw=2)
        ax.fill_between(sizes, acc_te, acc_tr, color="gray", alpha=0.15)
        ax.set_xscale("log")
        ax.set_title(name)
        ax.set_xlabel("nb points train")
        ax.set_ylabel("accuracy (%)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    axes[-1].axis("off")
    fig.suptitle("Effet de la densité de points sur 5 datasets", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "multi_density.png"), dpi=100)
    plt.close(fig)


def study_capacity():
    widths = [2, 4, 8, 16, 32, 64]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()
    for ax, (name, noise) in zip(axes, DATASETS):
        tl = fixed_test_loader(name, noise=noise)
        X_tr, y_tr = get_dataset(name, n_samples=300, noise=noise, seed=42)
        acc_tr, acc_te = [], []
        for w in widths:
            a, b = train_and_eval(X_tr, y_tr, tl, [w, w])
            acc_tr.append(a * 100)
            acc_te.append(b * 100)
        print(f"  {name:12s} capacité test={[round(x) for x in acc_te]}")
        ax.plot(widths, acc_tr, "o-", color="#F44336", label="train", lw=2)
        ax.plot(widths, acc_te, "o-", color="#2196F3", label="test", lw=2)
        ax.fill_between(widths, acc_te, acc_tr, color="gray", alpha=0.15)
        ax.set_xscale("log", base=2)
        ax.set_title(name)
        ax.set_xlabel("largeur (neurones/couche)")
        ax.set_ylabel("accuracy (%)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    axes[-1].axis("off")
    fig.suptitle("Effet de la capacité (largeur) sur 5 datasets — train 300 pts", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "multi_capacity.png"), dpi=100)
    plt.close(fig)


def main():
    print("=== DENSITÉ multi-datasets ===")
    study_density()
    print("=== CAPACITÉ multi-datasets ===")
    study_capacity()
    print(f"-> {OUT}/multi_density.png, multi_capacity.png")


if __name__ == "__main__":
    main()
