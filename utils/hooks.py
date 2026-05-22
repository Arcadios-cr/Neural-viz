"""
Module de capture des activations intermédiaires d'un réseau de neurones
via les forward hooks PyTorch.

Les forward hooks sont des fonctions attachées à un module PyTorch qui sont
appelées automatiquement à chaque forward pass. Elles permettent de "voler"
la sortie d'une couche sans modifier l'architecture du réseau.

Ce module est l'amorce de la visualisation des activations couche par couche
qui sera implémentée la semaine 4.
"""

import torch
import torch.nn as nn


# Types de modules considérés comme des "couches d'activation"
# (sortie post-non-linéarité, c'est ce qu'on veut visualiser)
ACTIVATION_TYPES = (nn.ReLU, nn.Tanh, nn.Sigmoid)


class ActivationCapture:
    """
    Capture les activations des couches cachées d'un MLP via forward hooks.

    Usage typique
    -------------
    >>> model = MLP(input_dim=2, hidden_layers=[8, 16], output_dim=1, activation="relu")
    >>> capture = ActivationCapture(model)
    >>> x = torch.randn(100, 2)
    >>> activations = capture.get_activations(x)
    >>> # activations est un dict : {"layer_0": tensor de shape (100, 8),
    >>> #                            "layer_1": tensor de shape (100, 16)}
    >>> capture.remove()  # libère les hooks

    Notes
    -----
    - Les hooks sont attachés aux modules d'activation (ReLU/Tanh/Sigmoid),
      donc on capture les valeurs APRÈS la non-linéarité — c'est ce qui est
      pédagogiquement intéressant : on voit quels neurones "s'allument" dans
      quelles régions du plan.
    - Les activations sont stockées dans un dictionnaire, par nom de couche.
    - Le mode eval() est temporairement activé pendant la capture pour que
      BatchNorm utilise ses statistiques accumulées (et non celles du tenseur
      de visualisation).
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.activations: dict[str, torch.Tensor] = {}
        self.handles: list = []
        self._register_hooks()

    # ─────────────────────────────────────────────
    # Gestion des hooks
    # ─────────────────────────────────────────────
    def _make_hook(self, name: str):
        """Crée une fonction hook qui stocke la sortie d'un module sous `name`."""
        def hook(module, input, output):
            # detach() : on coupe le lien avec autograd (on n'a pas besoin
            #            de gradients pour la visualisation).
            # clone() : on garde une copie indépendante, sinon les futures
            #           modifications du tenseur écraseraient ce qu'on a stocké.
            self.activations[name] = output.detach().clone()
        return hook

    def _register_hooks(self) -> None:
        """
        Parcourt le réseau (récursivement via `named_modules()`) et attache
        un hook à chaque module d'activation rencontré.

        Compatible avec les architectures variées :
        - MLP (un seul `nn.Sequential` nommé `network`)
        - MLPBottleneck (trois blocs : `encoder`, `bottleneck`, `decoder`)
        - Toute autre architecture combinant des couches d'activation.
        """
        layer_idx = 0
        for module in self.model.modules():
            if isinstance(module, ACTIVATION_TYPES):
                name = f"layer_{layer_idx}"
                handle = module.register_forward_hook(self._make_hook(name))
                self.handles.append(handle)
                layer_idx += 1

    def remove(self) -> None:
        """
        Supprime tous les hooks enregistrés.

        Important pour libérer la mémoire et éviter que les hooks restent
        actifs sur le modèle même quand l'objet ActivationCapture est détruit.
        """
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.activations = {}

    # ─────────────────────────────────────────────
    # Capture des activations
    # ─────────────────────────────────────────────
    def get_activations(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Effectue un forward pass et renvoie les activations capturées.

        Paramètres
        ----------
        x : torch.Tensor
            Entrée de shape (n_samples, input_dim)

        Retourne
        --------
        dict[str, torch.Tensor]
            Dictionnaire {nom_de_couche: tenseur d'activations}.
            Chaque tenseur a la shape (n_samples, n_neurones_de_la_couche).
        """
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            _ = self.model(x)
        if was_training:
            self.model.train()
        return self.activations

    # ─────────────────────────────────────────────
    # Context manager (optionnel mais propre)
    # ─────────────────────────────────────────────
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()


# ─────────────────────────────────────────────
# Démo / test rapide (lancé si on exécute le fichier directement)
# ─────────────────────────────────────────────
def _demo() -> None:
    """Petit test : crée un MLP, capture les activations, affiche les shapes."""
    from models.mlp import MLP

    print("Demo de ActivationCapture\n" + "-" * 40)

    model = MLP(
        input_dim=2,
        hidden_layers=[8, 16, 4],
        output_dim=1,
        activation="relu",
    )
    print(f"Modèle : {model}\n")

    # Avec un context manager — les hooks seront automatiquement libérés
    with ActivationCapture(model) as capture:
        x = torch.randn(5, 2)
        activations = capture.get_activations(x)

        print(f"Nombre de couches capturees : {len(activations)}")
        for name, tensor in activations.items():
            print(f"  {name} -> shape {tuple(tensor.shape)}")

    print("\nHooks liberes automatiquement a la sortie du with.")


if __name__ == "__main__":
    _demo()
