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


def knn_edges(A_bin: np.ndarray):
    """Liste des arêtes (i, j) avec i < j, pour le tracé du graphe."""
    iu, ju = np.where(np.triu(A_bin, k=1) > 0)
    return list(zip(iu.tolist(), ju.tolist()))
