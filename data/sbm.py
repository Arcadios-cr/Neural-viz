"""
Stochastic Block Model (SBM) : un dataset où c'est le GRAPHE qui porte
l'information, pas les coordonnées.

Motivation. Sur nos datasets habituels, le graphe est construit à partir des
coordonnées (k-NN) → il ne peut rien dire de plus que les features, donc le
réseau de graphe ≈ MLP. Pour voir un GNN VRAIMENT gagner, il faut le cas
« features faibles + structure forte » de la littérature :

  - le graphe est FOURNI par un SBM : beaucoup d'arêtes À L'INTÉRIEUR d'une
    communauté (= une classe), peu ENTRE communautés (graphe homophile) ;
  - les features 2D sont VOLONTAIREMENT faibles (communautés très chevauchantes)
    → un MLP qui ne voit que les features est presque au niveau du hasard.

Le réseau de graphe exploite la structure (les communautés) → il classe ;
le MLP ne peut pas. On reste en 2D donc tout est visualisable.
"""

import numpy as np

P_IN, P_OUT = 0.15, 0.01     # densité d'arêtes intra / inter communauté (homophile)
SEP = 0.35                   # séparation des centres de features (petite = faible)


def make_sbm(n_samples: int = 300, n_classes: int = 2, noise: float = 0.5,
             seed: int = 42):
    """
    Génère un SBM à `n_classes` communautés.

    Retourne (X, y, A_bin) :
      - X (n, 2)  : features 2D FAIBLES (centres par classe à peine séparés + bruit),
                    écart-type piloté par `noise` (plus de bruit = features plus inutiles) ;
      - y (n,)    : la communauté de chaque nœud (= sa classe) ;
      - A_bin (n, n) : adjacence binaire symétrique du graphe SBM (sans self-loop).
    """
    rng = np.random.default_rng(seed)
    n_per = max(n_samples // n_classes, 2)
    n = n_per * n_classes
    y = np.repeat(np.arange(n_classes), n_per)

    # ─── Graphe SBM : P[i,j] = P_IN si même communauté, P_OUT sinon ───
    same = (y[:, None] == y[None, :])
    P = np.where(same, P_IN, P_OUT)
    A = (rng.random((n, n)) < P).astype(np.float32)
    A = np.triu(A, 1)                       # garde le triangle supérieur…
    A = A + A.T                             # …puis symétrise (graphe non orienté)

    # ─── Features 2D faibles : centres par classe sur un cercle (rayon SEP) + bruit ───
    angles = 2 * np.pi * np.arange(n_classes) / n_classes
    centers = np.stack([np.cos(angles), np.sin(angles)], axis=1) * SEP
    scale = 1.0 + 0.6 * noise               # plus de bruit → features encore plus faibles
    X = (centers[y] + rng.normal(scale=scale, size=(n, 2))).astype(np.float32)

    # mélange (les communautés ne sont pas rangées dans l'ordre)
    perm = rng.permutation(n)
    return X[perm], y[perm].astype(np.int64), A[perm][:, perm]
