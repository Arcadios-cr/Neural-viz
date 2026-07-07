"""
Construction d'un graphe à partir d'un nuage de points 2D, pour les réseaux
convolutifs de graphe (GCN).

Idée : nos datasets ne sont pas sur une grille (contrairement à une image), donc
on définit le « voisinage » par un graphe des k plus proches voisins (k-NN). La
convolution de graphe agrège ensuite l'information le long de ce graphe.
"""

import numpy as np
import torch
from sklearn.neighbors import kneighbors_graph


def build_knn(X: np.ndarray, k: int = 6):
    """
    Construit le graphe k-NN d'un nuage de points et renvoie :
      - A_hat : adjacence normalisée à la Kipf & Welling (torch, n×n)
                Â = D^{-1/2} (A + I) D^{-1/2}, avec self-loops ;
      - A_bin : adjacence binaire symétrique (numpy, n×n), pour dessiner les arêtes.

    Symétrisation : si i est dans les k voisins de j (ou l'inverse), on met une
    arête — un graphe non orienté est plus naturel pour la convolution.
    """
    n = X.shape[0]
    k = min(k, n - 1)                      # garde-fou si très peu de points

    # Graphe k-NN binaire (connectivité 0/1), sans self-loop
    A = kneighbors_graph(X, n_neighbors=k, mode="connectivity", include_self=False).toarray()
    A = np.maximum(A, A.T)                 # symétrise : i~j si l'un est voisin de l'autre
    A_bin = A.copy()

    # Â = D^{-1/2} (A + I) D^{-1/2}  (ajout des self-loops puis normalisation)
    A = A + np.eye(n)
    deg = A.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    A_hat = d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]

    return torch.tensor(A_hat, dtype=torch.float32), A_bin


def build_radius(X: np.ndarray, r: float = 0.4):
    """
    Construit le graphe par RAYON (epsilon-ball) : deux points sont reliés si
    leur distance est inférieure à r. Renvoie (A_hat, A_bin) comme ``build_knn``.

    Différence fondamentale avec le k-NN : le nombre de voisins n'est PAS fixé.
    Dans une zone dense, un point a beaucoup de voisins ; dans une zone
    clairsemée, peu (voire aucun : le nœud reste isolé, la self-loop de la
    normalisation garde le calcul valide). Le DEGRÉ reflète donc la densité
    locale, que le k-NN efface (k voisins pour tout le monde).
    """
    from sklearn.neighbors import radius_neighbors_graph
    n = X.shape[0]
    A = radius_neighbors_graph(X, radius=r, mode="connectivity", include_self=False).toarray()
    A_bin = np.maximum(A, A.T)             # déjà symétrique en théorie ; on garantit

    A = A_bin + np.eye(n)
    deg = A.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    A_hat = d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]

    return torch.tensor(A_hat, dtype=torch.float32), A_bin


def knn_edges(A_bin: np.ndarray):
    """Liste des arêtes (i, j) avec i < j, pour le tracé du graphe."""
    iu, ju = np.where(np.triu(A_bin, k=1) > 0)
    return list(zip(iu.tolist(), ju.tolist()))


def normalize_adj(A_bin: np.ndarray) -> torch.Tensor:
    """
    Normalisation de Kipf : Â = D^{-1/2} (A + I) D^{-1/2}, à partir d'une
    adjacence binaire quelconque (k-NN ou graphe fourni type SBM). Ajoute les
    self-loops puis normalise par le degré → torch (n, n).
    """
    n = A_bin.shape[0]
    A = A_bin.astype(np.float32) + np.eye(n, dtype=np.float32)
    d_inv_sqrt = 1.0 / np.sqrt(A.sum(axis=1))
    A_hat = d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]
    return torch.tensor(A_hat, dtype=torch.float32)
