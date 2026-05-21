"""
MLP avec un goulot d'étranglement (bottleneck) à dimension fixe.

L'idée : forcer le réseau à compresser ses données dans un espace
de très petite dimension (typiquement 2D) au milieu, pour pouvoir
visualiser graphiquement la représentation interne qu'il a apprise.

Architecture :
    Input (2D)
       ↓
    ─── Encoder ───────────────────────
       Couches cachées (Linear + Act)
       ↓
    BOTTLENECK : Linear vers 2D (sans activation)  ← l'espace latent
       ↓
    ─── Décodeur / "tête" ─────────────
       Couches cachées (Linear + Act)
       ↓
    Output (1D : logit de classification)

L'encodeur "déplie" les données complexes (par exemple des spirales
entrelacées) en une représentation simple dans le bottleneck.
Le décodeur fait ensuite une classification triviale (souvent quasi
linéaire) sur cette représentation propre.
"""

import torch
import torch.nn as nn

from models.mlp import ACTIVATIONS


class MLPBottleneck(nn.Module):
    """
    MLP avec un goulot d'étranglement de dimension fixe au milieu.

    Paramètres
    ----------
    input_dim : int
        Dimension de l'entrée (2 pour R²).
    encoder_layers : list[int]
        Liste des nombres de neurones des couches cachées AVANT le bottleneck.
        Exemple : [16, 16] = 2 couches cachées de 16 neurones.
    bottleneck_dim : int
        Dimension du bottleneck. 2 pour permettre la visualisation graphique
        de l'espace latent.
    decoder_layers : list[int]
        Liste des nombres de neurones des couches cachées APRÈS le bottleneck.
    output_dim : int
        Dimension de la sortie (1 pour classification binaire).
    activation : str
        "relu", "tanh" ou "sigmoid" — appliquée aux couches encoder/decoder.
        Le bottleneck lui n'a PAS d'activation (espace latent brut).
    use_batchnorm : bool
        Si True, insère un BatchNorm après chaque Linear cachée.
    """

    def __init__(
        self,
        input_dim: int = 2,
        encoder_layers: list = [16, 16],
        bottleneck_dim: int = 2,
        decoder_layers: list = [16, 16],
        output_dim: int = 1,
        activation: str = "relu",
        use_batchnorm: bool = False,
    ):
        super().__init__()

        if activation not in ACTIVATIONS:
            raise ValueError(
                f"Activation '{activation}' non supportée. "
                f"Choisir parmi : {list(ACTIVATIONS.keys())}"
            )

        self.input_dim = input_dim
        self.encoder_layers = encoder_layers
        self.bottleneck_dim = bottleneck_dim
        self.decoder_layers = decoder_layers
        self.output_dim = output_dim
        self.activation_name = activation
        self.use_batchnorm = use_batchnorm

        act_fn = ACTIVATIONS[activation]

        # ─── Encoder : empile Linear → [BN] → Act ─────────────
        encoder_modules = []
        prev = input_dim
        for h in encoder_layers:
            encoder_modules.append(nn.Linear(prev, h))
            if use_batchnorm:
                encoder_modules.append(nn.BatchNorm1d(h))
            encoder_modules.append(act_fn())
            prev = h
        self.encoder = nn.Sequential(*encoder_modules)

        # ─── Bottleneck : Linear vers `bottleneck_dim`, SANS activation ───
        # Pas d'activation = espace latent brut, peut prendre n'importe
        # quelle valeur réelle. Plus expressif et plus lisible pour la viz.
        self.bottleneck = nn.Linear(prev, bottleneck_dim)

        # ─── Decoder : empile Linear → [BN] → Act, puis Linear de sortie ───
        decoder_modules = []
        prev = bottleneck_dim
        for h in decoder_layers:
            decoder_modules.append(nn.Linear(prev, h))
            if use_batchnorm:
                decoder_modules.append(nn.BatchNorm1d(h))
            decoder_modules.append(act_fn())
            prev = h
        decoder_modules.append(nn.Linear(prev, output_dim))  # sortie : logits
        self.decoder = nn.Sequential(*decoder_modules)

    # ─────────────────────────────────────────────
    # API spéciale pour la visualisation
    # ─────────────────────────────────────────────
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcule la représentation latente d'une entrée.

        Renvoie un tenseur de shape (n_samples, bottleneck_dim).
        C'est ÇA qu'on visualise : la position de chaque point d'entrée
        dans l'espace latent appris par le réseau.
        """
        return self.bottleneck(self.encoder(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass complet : encode → décode → sortie (logits)."""
        z = self.encode(x)
        return self.decoder(z)

    # ─────────────────────────────────────────────
    def __repr__(self) -> str:
        bn_tag = ", batchnorm=True" if self.use_batchnorm else ""
        return (
            f"MLPBottleneck(input={self.input_dim}, "
            f"encoder={self.encoder_layers}, "
            f"bottleneck={self.bottleneck_dim}, "
            f"decoder={self.decoder_layers}, "
            f"output={self.output_dim}, "
            f"activation={self.activation_name}"
            f"{bn_tag})"
        )
