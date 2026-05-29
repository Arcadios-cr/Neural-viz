import copy
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────
# Early stopping — classe utilitaire
# ─────────────────────────────────────────────
class EarlyStopping:
    """
    Arrête l'entraînement quand la validation loss ne s'améliore plus
    pendant `patience` époques consécutives.

    Paramètres
    ----------
    patience : int
        Nombre d'époques sans amélioration avant de déclencher l'arrêt.
    min_delta : float
        Amélioration minimale considérée comme significative. Permet
        d'éviter que des micro-variations de la loss (bruit de l'optim)
        ne resettent inutilement le compteur.

    Usage
    -----
    >>> es = EarlyStopping(patience=10, min_delta=1e-4)
    >>> for epoch in range(n_epochs):
    ...     val_loss = compute_val_loss(...)
    ...     if es(val_loss):
    ...         break
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        if patience < 1:
            raise ValueError("patience doit être >= 1.")
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        """Renvoie True si l'entraînement doit s'arrêter."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

    def reset(self) -> None:
        """Remet à zéro l'état (utile pour réutiliser l'objet)."""
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False


# ─────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────
class Trainer:
    """
    Encapsule la boucle d'entraînement d'un modèle PyTorch avec support
    optionnel de la validation, du best-model checkpoint et de l'early stopping.

    Le Trainer fournit un système de callbacks permettant à l'appelant
    (app Streamlit, script CLI, notebook...) d'injecter du code à chaque
    fin d'époque sans coupler le Trainer à un environnement particulier.

    État conservé pendant et après l'entraînement
    ---------------------------------------------
    - `history["train_loss"]`  : loss moyenne sur le train à chaque époque
    - `history["val_loss"]`    : loss moyenne sur la validation (si val_loader fourni)
    - `snapshots`              : state_dict du modèle à chaque époque
    - `best_state_dict`        : meilleur state_dict (lowest val loss)
    - `best_epoch`             : époque correspondant au meilleur état
    - `best_val_loss`          : meilleure val loss observée
    - `stopped_early`          : True si l'early stopping a déclenché
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

        # Historique
        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        # Historique des normes de gradient par couche Linear (si activé).
        # Clé = label lisible de la couche, valeur = liste d'une norme par époque.
        self.grad_history: dict[str, list[float]] = {}

        # Snapshots à chaque époque (pour le slider d'exploration)
        self.snapshots: list[dict] = []

        # Best model checkpoint
        self.best_state_dict: Optional[dict] = None
        self.best_val_loss: float = float("inf")
        self.best_epoch: int = -1

        # Statut d'arrêt
        self.stopped_early: bool = False

    # ─────────────────────────────────────────────
    # Couches Linear — labels lisibles (pour le suivi des gradients)
    # ─────────────────────────────────────────────
    def _linear_layers(self) -> list[tuple[str, nn.Module]]:
        """
        Renvoie la liste des couches Linear du modèle, dans l'ordre du réseau,
        avec un label lisible de la forme "L0 (2→8)".

        L'ordre suit `model.modules()` : pour un nn.Sequential, c'est l'ordre
        de l'entrée vers la sortie. La couche L0 est donc la plus proche de
        l'entrée, la dernière la couche de sortie.
        """
        layers = []
        idx = 0
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                label = f"L{idx} ({module.in_features}→{module.out_features})"
                layers.append((label, module))
                idx += 1
        return layers

    # ─────────────────────────────────────────────
    # Boucles internes
    # ─────────────────────────────────────────────
    def _train_one_epoch(self, loader: DataLoader, track_gradients: bool = False) -> float:
        """
        Entraîne le modèle sur une époque complète. Renvoie la loss moyenne.

        Si `track_gradients` est True, on calcule la norme L2 du gradient des
        poids de chaque couche Linear après chaque backward, on moyenne sur
        tous les batches de l'époque, et on stocke le résultat dans
        `self.grad_history`. La capture se fait entre backward() et step(),
        donc sur les gradients réellement utilisés pour la mise à jour.
        """
        self.model.train()
        epoch_loss = 0.0

        # Accumulateur des normes de gradient par couche, pour cette époque.
        grad_accum: dict[str, float] = {}
        n_batches = 0

        for X_batch, y_batch in loader:
            self.optimizer.zero_grad()
            preds = self.model(X_batch)
            loss = self.criterion(preds, y_batch)
            loss.backward()

            if track_gradients:
                for label, module in self._linear_layers():
                    if module.weight.grad is not None:
                        grad_accum[label] = (
                            grad_accum.get(label, 0.0)
                            + module.weight.grad.norm(2).item()
                        )

            self.optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        # Moyenne des normes sur les batches → une valeur par couche par époque.
        if track_gradients and n_batches > 0:
            for label, total in grad_accum.items():
                self.grad_history.setdefault(label, []).append(total / n_batches)

        return epoch_loss / len(loader)

    def _validate(self, loader: DataLoader) -> float:
        """Évalue la loss sur le DataLoader donné, sans modifier les poids."""
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                preds = self.model(X_batch)
                loss = self.criterion(preds, y_batch)
                total_loss += loss.item()
        return total_loss / len(loader)

    # ─────────────────────────────────────────────
    # Boucle externe — n époques + callbacks
    # ─────────────────────────────────────────────
    def train(
        self,
        train_loader: DataLoader,
        n_epochs: int,
        val_loader: Optional[DataLoader] = None,
        early_stopping: Optional[EarlyStopping] = None,
        on_epoch_end: Optional[Callable[[int, float, Optional[float], nn.Module], None]] = None,
        save_snapshots: bool = True,
        restore_best: bool = True,
        track_gradients: bool = False,
    ) -> dict[str, list[float]]:
        """
        Entraîne le modèle sur n_epochs époques.

        Paramètres
        ----------
        train_loader : DataLoader
            DataLoader d'entraînement.
        n_epochs : int
            Nombre maximum d'époques.
        val_loader : DataLoader, optionnel
            Si fourni, on calcule la loss de validation à chaque époque,
            on suit la meilleure (best checkpoint) et l'early stopping
            peut être utilisé.
        early_stopping : EarlyStopping, optionnel
            Si fourni (et val_loader aussi), stoppe l'entraînement quand
            la val loss ne s'améliore plus pendant `patience` époques.
        on_epoch_end : callable, optionnel
            Fonction appelée après chaque époque avec la signature
            (epoch_index, train_loss, val_loss_or_None, model).
            Permet à l'appelant d'injecter une mise à jour de visualisation.
        save_snapshots : bool
            Si True (défaut), sauvegarde le state_dict du modèle à chaque
            époque pour permettre la navigation a posteriori.
        restore_best : bool
            Si True (défaut), à la fin de l'entraînement le modèle est
            restauré à l'état où il avait la meilleure val_loss.
            Sans effet si val_loader n'est pas fourni.
        track_gradients : bool
            Si True, capture la norme L2 du gradient de chaque couche Linear
            à chaque époque (moyennée sur les batches) dans `self.grad_history`.
            Permet de diagnostiquer les vanishing/exploding gradients.

        Retourne
        --------
        dict[str, list[float]]
            L'historique des loss : {"train_loss": [...], "val_loss": [...]}
        """
        if early_stopping is not None and val_loader is None:
            raise ValueError(
                "early_stopping nécessite un val_loader pour calculer la val loss."
            )

        for epoch in range(n_epochs):
            # ─── Phase entraînement ───
            train_loss = self._train_one_epoch(train_loader, track_gradients=track_gradients)
            self.history["train_loss"].append(train_loss)

            # ─── Phase validation ───
            val_loss: Optional[float] = None
            if val_loader is not None:
                val_loss = self._validate(val_loader)
                self.history["val_loss"].append(val_loss)

                # Sauvegarde du best checkpoint si nouvelle meilleure val loss
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    self.best_state_dict = copy.deepcopy(self.model.state_dict())

            # ─── Snapshot pour le slider d'exploration ───
            if save_snapshots:
                self.snapshots.append(copy.deepcopy(self.model.state_dict()))

            # ─── Callback utilisateur ───
            if on_epoch_end is not None:
                on_epoch_end(epoch, train_loss, val_loss, self.model)

            # ─── Early stopping ───
            if early_stopping is not None and val_loss is not None:
                if early_stopping(val_loss):
                    self.stopped_early = True
                    break

        # ─── Restaurer le meilleur modèle ───
        if restore_best and self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

        return self.history

    # ─────────────────────────────────────────────
    # Restauration d'un snapshot précis
    # ─────────────────────────────────────────────
    def load_snapshot(self, epoch: int) -> None:
        """
        Restaure l'état du modèle tel qu'il était à la fin de l'époque donnée.

        Utile pour visualiser a posteriori l'état du réseau à un moment précis
        de l'entraînement (par exemple via un slider d'époque dans l'UI).
        """
        if not self.snapshots:
            raise RuntimeError(
                "Aucun snapshot disponible. Lance d'abord train(save_snapshots=True)."
            )
        if not (0 <= epoch < len(self.snapshots)):
            raise IndexError(
                f"Époque {epoch} hors plage. Snapshots disponibles : 0 à {len(self.snapshots) - 1}."
            )
        self.model.load_state_dict(self.snapshots[epoch])
