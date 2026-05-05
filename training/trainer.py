import copy
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    """
    Encapsule la boucle d'entraînement d'un modèle PyTorch.

    Le Trainer fournit un système de callbacks permettant à l'appelant
    (app Streamlit, script CLI, notebook...) d'injecter du code à chaque
    fin d'époque sans coupler le Trainer à un environnement particulier.

    Il garde aussi en mémoire :
    - l'historique des loss (`history["loss"]`)
    - un snapshot du modèle à chaque époque (`snapshots`) pour pouvoir
      revenir à n'importe quel état pendant la visualisation a posteriori.

    Paramètres
    ----------
    model : nn.Module
        Le modèle à entraîner.
    optimizer : torch.optim.Optimizer
        L'optimiseur (par exemple Adam).
    criterion : nn.Module
        La loss function (par exemple BCEWithLogitsLoss).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        # Historique et snapshots remplis pendant l'entraînement
        self.history: dict[str, list[float]] = {"loss": []}
        self.snapshots: list[dict] = []

    # ─────────────────────────────────────────────
    # Boucle interne — une époque
    # ─────────────────────────────────────────────
    def _train_one_epoch(self, loader: DataLoader) -> float:
        """Entraîne le modèle sur une époque complète et renvoie la loss moyenne."""
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            self.optimizer.zero_grad()
            preds = self.model(X_batch)
            loss = self.criterion(preds, y_batch)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(loader)

    # ─────────────────────────────────────────────
    # Boucle externe — n époques + callbacks
    # ─────────────────────────────────────────────
    def train(
        self,
        loader: DataLoader,
        n_epochs: int,
        on_epoch_end: Optional[Callable[[int, float, nn.Module], None]] = None,
        save_snapshots: bool = True,
    ) -> list[float]:
        """
        Entraîne le modèle sur n_epochs époques.

        Paramètres
        ----------
        loader : DataLoader
            Le DataLoader d'entraînement.
        n_epochs : int
            Nombre d'époques.
        on_epoch_end : callable, optionnel
            Fonction appelée après chaque époque avec la signature
            (epoch_index, epoch_loss, model). Permet à l'appelant
            d'injecter une mise à jour de visualisation, un logging,
            un early stopping, etc.
        save_snapshots : bool
            Si True (défaut), sauvegarde un deepcopy du state_dict du modèle
            à chaque fin d'époque. Permet la navigation a posteriori dans
            l'historique d'entraînement.

        Retourne
        --------
        list[float]
            La liste des loss moyennes par époque.
        """
        self.model.train()  # important pour BatchNorm / Dropout

        for epoch in range(n_epochs):
            epoch_loss = self._train_one_epoch(loader)
            self.history["loss"].append(epoch_loss)

            if save_snapshots:
                # deepcopy : on conserve un état figé indépendant des futures mises à jour
                self.snapshots.append(copy.deepcopy(self.model.state_dict()))

            if on_epoch_end is not None:
                on_epoch_end(epoch, epoch_loss, self.model)

        return self.history["loss"]

    # ─────────────────────────────────────────────
    # Restauration d'un snapshot
    # ─────────────────────────────────────────────
    def load_snapshot(self, epoch: int) -> None:
        """
        Restaure l'état du modèle tel qu'il était à la fin de l'époque donnée.

        Utile pour visualiser a posteriori l'état du réseau à un moment précis
        de l'entraînement (par exemple via un slider d'époque dans l'UI).
        """
        if not self.snapshots:
            raise RuntimeError("Aucun snapshot disponible. Lance d'abord train(save_snapshots=True).")
        if not (0 <= epoch < len(self.snapshots)):
            raise IndexError(
                f"Époque {epoch} hors plage. Snapshots disponibles : 0 à {len(self.snapshots) - 1}."
            )
        self.model.load_state_dict(self.snapshots[epoch])
