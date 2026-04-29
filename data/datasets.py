import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def make_gaussians(
    n_samples: int = 200,
    std: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Génère deux nuages de points gaussiens dans R².

    Classe 0 : centré en (-1, -1)
    Classe 1 : centré en ( 1,  1)

    Paramètres
    ----------
    n_samples : int
        Nombre total de points (divisé équitablement entre les deux classes)
    std : float
        Écart-type des gaussiennes (plus grand = nuages plus étalés)
    seed : int
        Graine aléatoire pour la reproductibilité

    Retourne
    --------
    X : np.ndarray de shape (n_samples, 2)
    y : np.ndarray de shape (n_samples,) avec valeurs 0 ou 1
    """
    rng = np.random.default_rng(seed)
    n_per_class = n_samples // 2

    X0 = rng.normal(loc=[-1.0, -1.0], scale=std, size=(n_per_class, 2))
    X1 = rng.normal(loc=[1.0, 1.0], scale=std, size=(n_per_class, 2))

    X = np.vstack([X0, X1]).astype(np.float32)
    y = np.array([0] * n_per_class + [1] * n_per_class, dtype=np.float32)

    # Mélange aléatoire
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def to_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    """
    Convertit des arrays numpy en DataLoader PyTorch.

    Paramètres
    ----------
    X : np.ndarray de shape (n, 2)
    y : np.ndarray de shape (n,)
    batch_size : int
    shuffle : bool

    Retourne
    --------
    DataLoader PyTorch prêt à l'emploi
    """
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y).unsqueeze(1)  # shape (n, 1) pour BCEWithLogitsLoss

    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
