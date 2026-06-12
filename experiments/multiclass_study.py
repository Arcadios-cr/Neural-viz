"""
Performance du mode MULTI-CLASSES (vraie config, test set fixe et indépendant).

Contrairement au balayage de robustesse (config minimale, but : ne pas planter),
ce script mesure les VRAIES performances du multi-classes avec une config adaptée
(MLP 2×32 ReLU, Adam lr 0,01, 200 époques, init seed 0). Le test est toujours
évalué sur un set FIXE de 1000 points (seed=999) du même dataset, pour des chiffres
comparables et reproductibles.

Pour Blobs (centres sur un cercle) et Gaussian Quantiles (anneaux concentriques),
en K = 3, 4, 5 classes, on rapporte : accuracy train, accuracy test, F1 macro.

Les deux datasets sont pris au MÊME niveau de bruit (0,5) pour une comparaison
équitable. L'effet attendu, robuste : à config et bruit égaux, l'accuracy DÉCROÎT
quand K augmente (plus de classes = plus de frontières à placer et moins de points
par classe). La difficulté absolue dépend aussi de la géométrie propre au dataset
— on ne cherche pas à classer un dataset « plus dur » que l'autre.

Génère : figures/multiclass_perf.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import train_and_report, fixed_test_loader
from data.datasets import get_dataset

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

# (dataset, niveau de bruit) — les deux familles multi-capables
DATASETS = [("Blobs", 0.5, "#2196F3"), ("Gaussian Quantiles", 0.5, "#F44336")]
KS = [3, 4, 5]
N_TRAIN = 600        # train fixe
LAYERS = [32, 32]    # config "réelle"
N_EPOCHS = 200


def main():
    results = {}   # name -> list de (K, acc_tr, acc_te, f1)
    print("=== PERFORMANCE MULTI-CLASSES (config 2×32, 200 ép, test fixe 1000 pts) ===")
    print(f"  {'dataset':20s} {'K':>2s}  {'train':>7s} {'test':>7s} {'F1macro':>8s}")
    for name, noise, _ in DATASETS:
        results[name] = []
        for K in KS:
            tl = fixed_test_loader(name, n=1000, noise=noise, seed=999, n_classes=K)
            X_tr, y_tr = get_dataset(name, n_samples=N_TRAIN, noise=noise, seed=42, n_classes=K)
            acc_tr, rep = train_and_report(X_tr, y_tr, tl, LAYERS, n_classes=K, n_epochs=N_EPOCHS)
            results[name].append((K, acc_tr * 100, rep.accuracy * 100, rep.f1 * 100))
            print(f"  {name:20s} {K:>2d}  {acc_tr*100:6.1f}% {rep.accuracy*100:6.1f}% {rep.f1*100:7.1f}%")

    # ─── Figure : accuracy test (+ F1 macro) en fonction de K ───
    fig, (ax_acc, ax_f1) = plt.subplots(1, 2, figsize=(13, 5))
    for name, noise, color in DATASETS:
        ks = [r[0] for r in results[name]]
        te = [r[2] for r in results[name]]   # acc_test
        f1 = [r[3] for r in results[name]]   # f1 macro
        ax_acc.plot(ks, te, "o-", color=color, lw=2, label=name)
        ax_f1.plot(ks, f1, "o-", color=color, lw=2, label=name)
    for ax, ylab, title in [(ax_acc, "Accuracy test (%)", "Accuracy test"),
                            (ax_f1, "F1 macro (%)", "F1 macro")]:
        ax.set_xticks(KS)
        ax.set_xlabel("nombre de classes K")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("Performance multi-classes selon K (config réelle, test set fixe)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, "multiclass_perf.png"), dpi=110)
    plt.close(fig)
    print(f"-> {OUT}/multiclass_perf.png")


if __name__ == "__main__":
    main()
