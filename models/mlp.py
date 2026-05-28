import torch
import torch.nn as nn


ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


class MLP(nn.Module):
    """
    Réseau de neurones MLP (Multi-Layer Perceptron) configurable.

    Paramètres
    ----------
    input_dim : int
        Nombre de neurones d'entrée (2 pour R²)
    hidden_layers : list[int]
        Liste du nombre de neurones par couche cachée (ex: [8, 8])
    output_dim : int
        Nombre de neurones de sortie (1 pour R²→R)
    activation : str
        Fonction d'activation : "relu", "tanh" ou "sigmoid"
    use_batchnorm : bool
        Si True, ajoute une couche de Batch Normalization après chaque couche linéaire cachée.
        Ordre : Linear -> BatchNorm -> Activation
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_layers: list = [8],
        output_dim: int = 1,
        activation: str = "relu",
        use_batchnorm: bool = False,
        dropout_rate: float = 0.0,
    ):
        super(MLP, self).__init__()

        if activation not in ACTIVATIONS:
            raise ValueError(f"Activation '{activation}' non supportée. Choisir parmi : {list(ACTIVATIONS.keys())}")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(f"dropout_rate doit être dans [0, 1[, reçu {dropout_rate}.")

        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.activation_name = activation
        self.use_batchnorm = use_batchnorm
        self.dropout_rate = dropout_rate

        act_fn = ACTIVATIONS[activation]

        # Construction des couches
        # Ordre : Linear -> [BatchNorm] -> Activation -> [Dropout]
        # Le dropout est placé APRES l'activation (convention la plus courante).
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act_fn())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def __repr__(self):
        bn_tag = ", batchnorm=True" if self.use_batchnorm else ""
        dropout_tag = f", dropout={self.dropout_rate:.2f}" if self.dropout_rate > 0 else ""
        return (
            f"MLP(input={self.input_dim}, "
            f"hidden={self.hidden_layers}, "
            f"output={self.output_dim}, "
            f"activation={self.activation_name}"
            f"{bn_tag}{dropout_tag})"
        )
