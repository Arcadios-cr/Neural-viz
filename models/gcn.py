"""
Réseau convolutif de graphe (GCN), implémenté à la main (Kipf & Welling, 2017).

Principe : sur une image, un CNN convolue sur une grille (chaque pixel regarde
ses voisins). Sur un nuage de points, il n'y a pas de grille → on définit le
voisinage par un graphe k-NN (cf. data/graphs.py). La « convolution de graphe »
met à jour les features d'un nœud en AGRÉGEANT celles de ses voisins :

    H' = σ( Â · H · W )

où Â est l'adjacence normalisée (avec self-loops). Empiler des couches = laisser
l'information se propager sur un voisinage de plus en plus large (comme la
profondeur d'un CNN agrandit le champ réceptif). On termine par une petite tête
MLP (un « perceptron » qui agit nœud par nœud) pour classifier.
"""

import torch
import torch.nn as nn

_ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}


class GCNLayer(nn.Module):
    """Une couche de convolution de graphe : H' = Â · (H · W + b)."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features)

    def forward(self, H: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        # 1) transformation linéaire des features ; 2) agrégation des voisins
        return A_hat @ self.lin(H)


class GCN(nn.Module):
    """
    Quelques couches de convolution de graphe, puis une tête MLP par nœud.

    Paramètres
    ----------
    input_dim   : dimension des features d'entrée (2 = les coordonnées x₁, x₂).
    gcn_layers  : tailles des couches de graphe (ex. [16, 16]).
    head_layers : tailles des couches de la tête MLP (ex. [16]).
    output_dim  : 1 (binaire) ou K (multi-classes).

    forward(X, A_hat) renvoie les logits par nœud, shape (n, output_dim).
    """

    def __init__(self, input_dim=2, gcn_layers=(16, 16), head_layers=(16,),
                 output_dim=1, activation="relu"):
        super().__init__()
        act = _ACTIVATIONS[activation]

        # ─── Couches de convolution de graphe ───
        self.gcns = nn.ModuleList()
        d = input_dim
        for h in gcn_layers:
            self.gcns.append(GCNLayer(d, h))
            d = h
        self.act = act()

        # ─── Tête MLP (agit indépendamment sur chaque nœud) ───
        head = []
        for h in head_layers:
            head += [nn.Linear(d, h), act()]
            d = h
        head.append(nn.Linear(d, output_dim))
        self.head = nn.Sequential(*head)

    def forward(self, X: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        H = self.encode(X, A_hat)
        return self.head(H)

    def encode(self, X: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        """Représentation des nœuds après les couches de graphe (avant la tête)."""
        H = X
        for g in self.gcns:
            H = self.act(g(H, A_hat))
        return H
