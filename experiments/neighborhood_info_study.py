"""
Synthèse S11 — quand l'information du VOISINAGE aide-t-elle ?

Trois scénarios de chevauchement, où l'indice discriminant local diffère :

1. « Density »  : bleu très dense / rouge très clairsemé (datasets.make_density).
   La densité est l'indice… mais elle est COUPLÉE à la position (la zone dense
   est aussi la zone de gauche) → l'info du voisinage est redondante.
2. « Densité découplée » : micro-amas bleus denses éparpillés PARTOUT + rouge
   uniforme clairsemé (généré ici). La position ne dit rien, la densité dit tout.
3. « Structure » : bleu en colonnes / rouge en lignes (datasets.make_structure).
   L'indice est l'ARRANGEMENT local (vertical vs horizontal) des voisins.

Panel de modèles (mêmes réglages que l'app : 2 couches de graphe × 8, tête [4],
k = 6, Adam lr 0.01, 100 époques, transductif 60/20/20, 3 seeds d'init) :
  - MLP (position seule) ;
  - GCN agrégation moyenne sur graphe k-NN (degré ~constant) ;
  - GCN agrégation moyenne sur graphe par RAYON (degré = densité locale) ;
  - GCN agrégation max (GraphSAGE) sur k-NN (sensible à l'étalement directionnel) ;
  - GCN agrégation SOMME (GIN) sur graphe par RAYON, BatchNorm forcée : la seule
    agrégation qui « compte » les voisins, donc qui peut lire le degré ;
  - MLP + features d'ARÊTE (longueur moyenne des k arêtes + verticalité du
    voisinage, cf. data.graphs.knn_edge_stats) : la recette GÉNÉRIQUE proposée
    par l'encadrant — les deux mêmes features partout, sans choisir selon le
    scénario ; la géométrie que la binarisation du graphe jette, consommée
    directement en entrée ;
  - MLP + « info voisinage » en 3e feature (degré du graphe rayon pour la
    densité, anisotropie |dy|-|dx| des voisins k-NN pour la structure) :
    borne « oracle » de ce que le voisinage PEUT apporter si on sait le lire
    ET qu'on sait laquelle donner.

Loi : le voisinage n'aide que si (a) il porte une info
ABSENTE des features (position) et (b) l'architecture sait la LIRE — l'agrégation
moyenne normalisée (Kipf) gomme le degré, et la moyenne compense les décalages
symétriques (arrangement). La barre « MLP + feature » (la borne) domine dès que
l'info existe ; la SOMME est la variante d'architecture qui remplit la condition
(b) pour la densité : sur le scénario découplé, elle rejoint la borne (~96 %) là
où moyenne et max restent à la baseline (~77 %).

NOTE protocole : contrairement aux autres études de ce dossier (test set fixe
de 1000 points), on est ici en TRANSDUCTIF (un seul graphe, split de nœuds
60/20/20), comme le mode graphe de l'app — c'est le cadre naturel d'un réseau
de graphe qui classe des nœuds.

Génère : figures/neighborhood_info.png
"""

import os
import sys

# Permet d'importer les modules du projet (models, data) quel que soit le
# répertoire depuis lequel on lance le script (même convention que _common.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from data.datasets import get_dataset
from data.graphs import build_knn, build_radius, knn_edge_stats
from models.gcn import GCN
from models.mlp import MLP

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

SEED = 42
K, EPOCHS, LR, NEUR = 6, 100, 0.01, 8
SEEDS_W = (0, 1, 2)

PALETTE = {
    "MLP (position)":       "#9C27B0",
    "GCN mean (k-NN)":      "#4CAF50",
    "GCN mean (rayon)":     "#FF9800",
    "GCN max (k-NN)":       "#FF5722",
    "GCN somme (rayon)":    "#E91E63",
    "MLP + feat. d'arête":  "#6A1B9A",
    "MLP + info voisinage": "#607D8B",
}


def make_decoupled(seed: int = SEED):
    """Micro-amas bleus denses éparpillés + rouge uniforme clairsemé (jouet S11)."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-1.8, 1.8, size=(12, 2))
    X0 = np.vstack([rng.normal(loc=c, scale=0.08, size=(13, 2)) for c in centers])
    X1 = rng.uniform(-2.0, 2.0, size=(50, 2))
    X = np.vstack([X0, X1]).astype(np.float32)
    y = np.array([0] * len(X0) + [1] * len(X1))
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def masks(n):
    """Split transductif 60/20/20 des nœuds (mêmes conventions que l'app)."""
    perm = np.random.default_rng(SEED).permutation(n)
    n_tr, n_val = int(0.6 * n), int(0.2 * n)
    return perm[:n_tr], perm[n_tr + n_val:]


def neighborhood_feature(X, kind, A_rad_bin):
    """L'info du voisinage sous forme de feature normalisée (borne « lisible »)."""
    if kind == "degre":
        f = A_rad_bin.sum(1)
    else:                                   # anisotropie des voisins k-NN
        _, A_bin = build_knn(X, k=K)
        f = np.zeros(len(X))
        for i in range(len(X)):
            nb = np.where(A_bin[i] > 0)[0]
            d = X[nb] - X[i]
            f[i] = np.abs(d[:, 1]).mean() - np.abs(d[:, 0]).mean()
    return (f - f.mean()) / (f.std() + 1e-9)


def run_scenario(X, y, r, feature_kind):
    """Entraîne le panel de modèles ; renvoie {nom: (acc_moy, acc_std)} (nœuds test)."""
    n = len(y)
    tr_idx, te_idx = masks(n)
    m_tr = torch.tensor(np.isin(np.arange(n), tr_idx))
    yt = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    crit = nn.BCEWithLogitsLoss()

    A_knn, _ = build_knn(X, k=K)
    A_rad, Ab_rad = build_radius(X, r=r)
    feat = neighborhood_feature(X, feature_kind, Ab_rad)
    X_feat = np.hstack([X, feat[:, None]]).astype(np.float32)
    # Route 3 : la géométrie des arêtes k-NN en features (longueur + verticalité)
    mlen, vert = knn_edge_stats(X, X, k=K)
    mlen_n = (mlen - mlen.mean()) / (mlen.std() + 1e-9)
    X_edge = np.hstack([X, mlen_n[:, None], vert[:, None]]).astype(np.float32)

    def gcn(Ah, agg, ws, bn=False):
        Xt = torch.tensor(X, dtype=torch.float32)
        torch.manual_seed(ws)
        gm = GCN(input_dim=2, gcn_layers=[NEUR] * 2, head_layers=[max(NEUR // 2, 4)],
                 output_dim=1, activation="relu", aggregation=agg, heads=4,
                 use_batchnorm=bn)
        opt = torch.optim.Adam(gm.parameters(), lr=LR)
        torch.manual_seed(ws)
        for _ in range(EPOCHS):
            gm.train(); opt.zero_grad()
            crit(gm(Xt, Ah)[m_tr], yt[m_tr]).backward(); opt.step()
        gm.eval()
        with torch.no_grad():
            return (gm(Xt, Ah).numpy()[:, 0] > 0).astype(int)

    def mlp(Xin, ws):
        Xt = torch.tensor(Xin, dtype=torch.float32)
        torch.manual_seed(ws)
        ml = MLP(Xt.shape[1], [NEUR, NEUR], 1, activation="relu")
        opt = torch.optim.Adam(ml.parameters(), lr=LR)
        torch.manual_seed(ws)
        for _ in range(EPOCHS):
            ml.train(); opt.zero_grad()
            crit(ml(Xt[tr_idx]), yt[tr_idx]).backward(); opt.step()
        ml.eval()
        with torch.no_grad():
            return (ml(Xt).numpy()[:, 0] > 0).astype(int)

    models = {
        "MLP (position)":       lambda ws: mlp(X, ws),
        "GCN mean (k-NN)":      lambda ws: gcn(A_knn, "mean", ws),
        "GCN mean (rayon)":     lambda ws: gcn(A_rad, "mean", ws),
        "GCN max (k-NN)":       lambda ws: gcn(A_knn, "max", ws),
        # Somme (GIN) : la seule agrégation qui « compte » les voisins — BN forcée
        # (sans elle : instable, ±9 pts entre seeds ; cf. l'article GIN).
        "GCN somme (rayon)":    lambda ws: gcn(A_rad, "sum", ws, bn=True),
        "MLP + feat. d'arête":  lambda ws: mlp(X_edge, ws),
        "MLP + info voisinage": lambda ws: mlp(X_feat, ws),
    }
    out = {}
    for name, fn in models.items():
        accs = [float((fn(ws)[te_idx] == y[te_idx]).mean()) for ws in SEEDS_W]
        out[name] = (float(np.mean(accs)), float(np.std(accs)))
    out["_baseline"] = float(max((y[te_idx] == 0).mean(), (y[te_idx] == 1).mean()))
    return out


def main():
    scenarios = [
        ("Density\n(densité couplée à la position)",
         *get_dataset("Density", n_samples=200, noise=0.2, seed=SEED), 0.40, "degre"),
        ("Densité découplée\n(amas partout + uniforme)",
         *make_decoupled(), 0.25, "degre"),
        ("Structure\n(colonnes vs lignes)",
         *get_dataset("Structure", n_samples=200, noise=0.2, seed=SEED), 0.40, "aniso"),
    ]

    results = []
    print("=== QUAND L'INFO DU VOISINAGE AIDE-T-ELLE ? (transductif, 2x8, 3 seeds) ===")
    for label, X, y, r, kind in scenarios:
        res = run_scenario(X, y, r, kind)
        results.append((label, res))
        print(f"-- {label.replace(chr(10), ' ')} (baseline majoritaire "
              f"{res['_baseline'] * 100:.0f} %) --")
        for name in PALETTE:
            m, s = res[name]
            print(f"   {name:22s} : {m * 100:5.1f}% (±{s * 100:.1f})")

    # ─── Figure : barres groupées par scénario ───
    names = list(PALETTE)
    n_models = len(names)
    width = 0.12
    fig, ax = plt.subplots(figsize=(13, 5.5))
    xs = np.arange(len(results))
    for j, name in enumerate(names):
        vals = [res[name][0] * 100 for _, res in results]
        errs = [res[name][1] * 100 for _, res in results]
        ax.bar(xs + (j - (n_models - 1) / 2) * width, vals, width, yerr=errs,
               capsize=3, color=PALETTE[name], label=name)
    for i, (_, res) in enumerate(results):
        ax.hlines(res["_baseline"] * 100, xs[i] - 0.42, xs[i] + 0.42,
                  color="gray", ls="--", lw=1)
    ax.hlines([], [], [], color="gray", ls="--", lw=1, label="baseline majoritaire")
    ax.set_xticks(xs)
    ax.set_xticklabels([label for label, _ in results], fontsize=9)
    ax.set_ylabel("accuracy test (%)"); ax.set_ylim(0, 100)
    ax.set_title("Le voisinage n'aide que si son info est ABSENTE des features "
                 "et LISIBLE par l'agrégation (3 seeds)")
    ax.legend(fontsize=8, ncol=3, loc="lower right")
    fig.tight_layout()
    path = os.path.join(OUT, "neighborhood_info.png")
    fig.savefig(path, dpi=130)
    print(f"figure : {path}")


if __name__ == "__main__":
    main()
