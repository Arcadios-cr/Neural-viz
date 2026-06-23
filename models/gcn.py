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

L'agrégation des voisins se fait par MOYENNE (Kipf) ou par MAX (esprit du
*pooling* de GraphSAGE, Hamilton 2017) — cf. ``GCNLayer``.
"""

import torch
import torch.nn as nn

_ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}


class GCNLayer(nn.Module):
    """
    Une couche de convolution de graphe : transformation linéaire des features
    puis AGRÉGATION sur le voisinage. Deux agrégateurs au choix :

    - ``"mean"`` (Kipf & Welling, défaut) : moyenne pondérée des voisins,
      H' = Â · (H · W + b), où Â est l'adjacence normalisée (self-loops inclus) ;
    - ``"max"`` (esprit du *pooling* de GraphSAGE, Hamilton 2017) : pour chaque
      nœud, MAX dimension par dimension des features transformées de ses voisins
      (self-loop inclus), max({ W·h_u + b , ∀u ∈ N(v)∪{v} }).

    Le voisinage est lu sur le motif non-nul de la matrice fournie (Â en mode
    moyenne, ou n'importe quelle adjacence en mode max — seule la connectivité
    compte, le max ignore les poids). L'activation est appliquée APRÈS, dans
    ``GCN.encode`` : à structure égale, seul l'agrégateur (moyenne vs max) change.
    """

    def __init__(self, in_features: int, out_features: int, aggregation: str = "mean"):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features)
        self.aggregation = aggregation

    def forward(self, H: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        Z = self.lin(H)                              # transformation linéaire des features
        if self.aggregation == "mean":
            return A @ Z                             # moyenne pondérée (Â inclut les self-loops)

        # ─── max-pool : max sur les voisins (motif non-nul de A = voisinage) ───
        rows, cols = (A != 0).nonzero(as_tuple=True)  # arête i←j : j voisin de i
        index = rows.unsqueeze(1).expand(-1, Z.shape[1])
        out = Z.new_full((H.shape[0], Z.shape[1]), float("-inf"))
        # out[i] = max sur les voisins j de Z[j] (les self-loops garantissent qu'aucun
        # nœud ne reste à -inf). scatter_reduce est différentiable (autograd sur amax).
        return out.scatter_reduce(0, index, Z[cols], reduce="amax", include_self=True)


class GCN(nn.Module):
    """
    Quelques couches de convolution de graphe, puis une tête MLP par nœud.

    Paramètres
    ----------
    input_dim   : dimension des features d'entrée (2 = les coordonnées x₁, x₂).
    gcn_layers  : tailles des couches de graphe (ex. [16, 16]).
    head_layers : tailles des couches de la tête MLP (ex. [16]).
    output_dim  : 1 (binaire) ou K (multi-classes).
    aggregation : ``"mean"`` (Kipf, défaut) ou ``"max"`` (pooling GraphSAGE).
    use_batchnorm : si True, une BatchNorm est insérée après chaque couche de
        graphe (avant l'activation). En transductif (un seul graphe), elle
        normalise chaque feature sur l'ensemble des nœuds → entraînement plus
        stable, échelles homogènes entre couches.

    forward(X, A_hat) renvoie les logits par nœud, shape (n, output_dim).
    """

    def __init__(self, input_dim=2, gcn_layers=(16, 16), head_layers=(16,),
                 output_dim=1, activation="relu", aggregation="mean",
                 use_batchnorm=False):
        super().__init__()
        act = _ACTIVATIONS[activation]

        # ─── Couches de convolution de graphe (+ BatchNorm optionnelle) ───
        # Une BatchNorm (ou Identity si désactivée) par couche, appliquée entre
        # l'agrégation et l'activation — cf. encode().
        self.gcns = nn.ModuleList()
        self.bns = nn.ModuleList()
        d = input_dim
        for h in gcn_layers:
            self.gcns.append(GCNLayer(d, h, aggregation=aggregation))
            self.bns.append(nn.BatchNorm1d(h) if use_batchnorm else nn.Identity())
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
        for g, bn in zip(self.gcns, self.bns):
            # convolution de graphe → BatchNorm (sur tous les nœuds) → activation
            H = self.act(bn(g(H, A_hat)))
        return H
