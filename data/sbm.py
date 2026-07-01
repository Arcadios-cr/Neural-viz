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

AVG_DEG = 12.0               # degré moyen visé (constant quelle que soit l'homophilie)
SEP = 0.35                   # séparation des centres de features (petite = faible)


def make_sbm(n_samples: int = 300, n_classes: int = 2, noise: float = 0.5,
             seed: int = 42, homophily: float = 0.9):
    """
    Génère un SBM à `n_classes` communautés.

    `homophily` ∈ [0.5, 1.0] = fraction des arêtes qui restent DANS une communauté :
      - 0.5 → graphe (quasi) aléatoire : autant d'arêtes intra qu'inter → le graphe
        ne porte AUCUNE info de communauté → le réseau de graphe retombe au niveau
        du MLP ;
      - → 1.0 → communautés de plus en plus pures → le graphe porte l'info → le
        réseau de graphe gagne.
    Le degré moyen est maintenu ~constant (AVG_DEG) pour ne pas confondre densité et
    homophilie.

    Retourne (X, y, A_bin) :
      - X (n, 2)  : features 2D FAIBLES (centres par classe à peine séparés + bruit) ;
      - y (n,)    : la communauté de chaque nœud (= sa classe) ;
      - A_bin (n, n) : adjacence binaire symétrique du graphe SBM (sans self-loop).
    """
    rng = np.random.default_rng(seed)
    n_per = max(n_samples // n_classes, 2)
    n = n_per * n_classes
    y = np.repeat(np.arange(n_classes), n_per)

    # ─── Probas d'arête intra (p_in) / inter (p_out) telles que le ratio =
    #     homophily/(1-homophily) et le degré moyen ≈ AVG_DEG ───
    n_intra = n_per - 1                     # voisins potentiels de même communauté
    n_inter = (n_classes - 1) * n_per       # voisins potentiels d'autres communautés
    t = AVG_DEG / (homophily * n_intra + (1 - homophily) * n_inter + 1e-9)
    p_in = min(homophily * t, 1.0)
    p_out = min((1 - homophily) * t, 1.0)

    same = (y[:, None] == y[None, :])
    P = np.where(same, p_in, p_out)
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
