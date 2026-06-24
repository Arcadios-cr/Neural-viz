"""
Réseau de graphe configurable, implémenté à la main.

Toutes les architectures de GNN suivent le même schéma de « passage de messages »
(message passing) : chaque nœud met à jour ses features en AGRÉGEANT celles de ses
voisins. Elles ne diffèrent que par la MANIÈRE d'agréger. On expose donc une seule
couche de graphe (``GCNLayer``) avec trois agrégations au choix, qui reproduisent
chacune une architecture classique :

    H' = σ( AGG_{j ∈ voisins(i)} ( W·h_j ) )

  - ``"mean"``  → moyenne normalisée à poids FIXES  → **GCN** (Kipf & Welling, 2017)
                  H' = σ( Â · H · W ), Â = adjacence normalisée (self-loops inclus).
  - ``"max"``   → max par dimension (agrégateur apprenable) → **GraphSAGE** (Hamilton, 2017).
  - ``"attention"`` → somme pondérée par une attention APPRISE multi-têtes → **GAT**
                  (Veličković et al., 2018) : le nœud apprend combien chaque voisin compte.

Empiler des couches élargit le champ réceptif (comme la profondeur d'un CNN). On
termine par une petite tête MLP (un « perceptron » par nœud) pour classifier.

Choix de conception : les têtes d'attention sont MOYENNÉES (pas concaténées), si
bien que la dimension de sortie d'une couche est la même quelle que soit
l'agrégation → « à architecture égale, seule l'agrégation change » (comparaison
GCN / GraphSAGE / GAT parfaitement équitable). C'est la variante d'agrégation
multi-têtes utilisée par la couche de sortie de l'article GAT.
"""

import torch
import torch.nn as nn

_ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}


class GCNLayer(nn.Module):
    """
    Une couche de graphe : transformation linéaire des features puis AGRÉGATION
    sur le voisinage, selon ``aggregation`` :

    - ``"mean"`` (GCN, Kipf) : moyenne pondérée des voisins, H' = Â · (H·W + b),
      Â = adjacence normalisée (self-loops inclus) ;
    - ``"max"`` (GraphSAGE) : pour chaque nœud, MAX dimension par dimension des
      features transformées de ses voisins (self-loop inclus) ;
    - ``"attention"`` (GAT) : somme pondérée par une attention apprise multi-têtes
      e_ij = LeakyReLU(aᵀ[W·h_i ‖ W·h_j]), α = softmax sur les voisins, les K têtes
      étant MOYENNÉES → sortie de dimension ``out_features`` (comme mean/max).

    L'activation est appliquée APRÈS, dans ``GCN.encode``. Le voisinage est lu sur
    le motif non-nul de la matrice fournie (les poids ne comptent que pour ``mean``).
    """

    def __init__(self, in_features: int, out_features: int,
                 aggregation: str = "mean", heads: int = 1):
        super().__init__()
        self.aggregation = aggregation
        self.out_features = out_features
        if aggregation == "attention":
            # une projection par tête (empilées) + un vecteur d'attention par tête
            self.heads = heads
            self.lin = nn.Linear(in_features, heads * out_features, bias=False)
            self.a_src = nn.Parameter(torch.empty(heads, out_features))
            self.a_dst = nn.Parameter(torch.empty(heads, out_features))
            self.leaky = nn.LeakyReLU(0.2)            # pente négative 0.2 (article GAT)
            nn.init.xavier_uniform_(self.lin.weight, gain=1.414)
            nn.init.xavier_uniform_(self.a_src, gain=1.414)
            nn.init.xavier_uniform_(self.a_dst, gain=1.414)
        else:
            self.lin = nn.Linear(in_features, out_features)

    def forward(self, H: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        if self.aggregation == "attention":
            return self._attention(H, A)

        Z = self.lin(H)                              # transformation linéaire des features
        if self.aggregation == "mean":
            return A @ Z                             # moyenne pondérée (Â inclut les self-loops)

        # ─── max-pool (GraphSAGE) : max sur les voisins (motif non-nul de A) ───
        rows, cols = (A != 0).nonzero(as_tuple=True)  # arête i←j : j voisin de i
        index = rows.unsqueeze(1).expand(-1, Z.shape[1])
        out = Z.new_full((H.shape[0], Z.shape[1]), float("-inf"))
        # out[i] = max sur les voisins j de Z[j] (les self-loops garantissent qu'aucun
        # nœud ne reste à -inf). scatter_reduce est différentiable (autograd sur amax).
        return out.scatter_reduce(0, index, Z[cols], reduce="amax", include_self=True)

    def _attention(self, H: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        n = H.shape[0]
        Wh = self.lin(H).view(n, self.heads, self.out_features)    # (n, heads, out)
        # e_ij = LeakyReLU( a_src·Wh_i + a_dst·Wh_j )  (décomposition additive)
        s_src = (Wh * self.a_src).sum(-1)                          # (n, heads)
        s_dst = (Wh * self.a_dst).sum(-1)                          # (n, heads)
        e = self.leaky(s_src.unsqueeze(1) + s_dst.unsqueeze(0))    # (n, n, heads)
        # non-voisins → -inf → poids nul après softmax (self-loops garantissent ≥1 voisin)
        e = e.masked_fill(~(A != 0).unsqueeze(-1), float("-inf"))
        alpha = torch.softmax(e, dim=1)                           # normalisé sur les voisins j
        out = torch.einsum("ijh,jhf->ihf", alpha, Wh)             # (n, heads, out)
        return out.mean(dim=1)                                    # MOYENNE des têtes → (n, out)

    @torch.no_grad()
    def attention_weights(self, H: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """Poids d'attention α (n, n, heads) — pour la visualisation (mode attention)."""
        n = H.shape[0]
        Wh = self.lin(H).view(n, self.heads, self.out_features)
        s_src = (Wh * self.a_src).sum(-1)
        s_dst = (Wh * self.a_dst).sum(-1)
        e = self.leaky(s_src.unsqueeze(1) + s_dst.unsqueeze(0))
        e = e.masked_fill(~(A != 0).unsqueeze(-1), float("-inf"))
        return torch.softmax(e, dim=1)


class GCN(nn.Module):
    """
    Réseau de graphe configurable : quelques couches de graphe (agrégation au choix),
    puis une tête MLP par nœud. Selon ``aggregation``, se comporte comme un GCN
    (mean), un GraphSAGE (max) ou un GAT (attention).

    Paramètres
    ----------
    input_dim   : dimension des features d'entrée (2 = les coordonnées x₁, x₂).
    gcn_layers  : tailles des couches de graphe (ex. [16, 16]).
    head_layers : tailles des couches de la tête MLP (ex. [16]).
    output_dim  : 1 (binaire) ou K (multi-classes).
    aggregation : ``"mean"`` (GCN) · ``"max"`` (GraphSAGE) · ``"attention"`` (GAT).
    heads       : nombre de têtes d'attention (utilisé seulement si attention ;
                  les têtes sont moyennées → dimension inchangée).
    use_batchnorm : BatchNorm après chaque couche de graphe (avant l'activation).
        En transductif (un seul graphe), elle normalise chaque feature sur
        l'ensemble des nœuds → entraînement plus stable.

    forward(X, A_hat) renvoie les logits par nœud, shape (n, output_dim).
    """

    def __init__(self, input_dim=2, gcn_layers=(16, 16), head_layers=(16,),
                 output_dim=1, activation="relu", aggregation="mean",
                 heads=1, use_batchnorm=False):
        super().__init__()
        act = _ACTIVATIONS[activation]

        # ─── Couches de graphe (+ BatchNorm optionnelle) ───
        # Têtes d'attention moyennées → la sortie d'une couche fait toujours `h`,
        # quelle que soit l'agrégation (dimensions stables d'une couche à l'autre).
        self.gcns = nn.ModuleList()
        self.bns = nn.ModuleList()
        d = input_dim
        for h in gcn_layers:
            self.gcns.append(GCNLayer(d, h, aggregation=aggregation, heads=heads))
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
            # agrégation de graphe → BatchNorm (sur tous les nœuds) → activation
            H = self.act(bn(g(H, A_hat)))
        return H
