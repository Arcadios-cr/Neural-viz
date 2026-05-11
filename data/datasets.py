import numpy as np
import torch
from sklearn.datasets import make_moons as sk_make_moons, make_circles as sk_make_circles
from torch.utils.data import TensorDataset, DataLoader


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _shuffle(X: np.ndarray, y: np.ndarray, rng: np.random.Generator):
    """Mélange aléatoirement X et y avec la même permutation."""
    idx = rng.permutation(len(y))
    return X[idx].astype(np.float32), y[idx].astype(np.float32)


# ─────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────
def make_gaussians(
    n_samples: int = 200,
    noise: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Deux gaussiennes en (-1,-1) et (1,1). Linéairement séparable.
    """
    rng = np.random.default_rng(seed)
    n_per_class = n_samples // 2
    X0 = rng.normal(loc=[-1.0, -1.0], scale=noise, size=(n_per_class, 2))
    X1 = rng.normal(loc=[1.0, 1.0], scale=noise, size=(n_per_class, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return _shuffle(X, y, rng)


def make_moons(
    n_samples: int = 200,
    noise: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Deux croissants entrelacés (sklearn make_moons).

    """
    X, y = sk_make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    rng = np.random.default_rng(seed)
    return _shuffle(X, y, rng)


def make_circles(
    n_samples: int = 200,
    noise: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Deux cercles concentriques (sklearn make_circles).

    """
    X, y = sk_make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=seed)
    rng = np.random.default_rng(seed)
    return _shuffle(X, y, rng)


def make_xor(
    n_samples: int = 200,
    noise: float = 0.3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Quatre nuages gaussiens en motif XOR.

    Classe 0 : (-1,-1) et (1,1)   (diagonale principale)
    Classe 1 : (-1, 1) et (1,-1)  (anti-diagonale)

    """
    rng = np.random.default_rng(seed)
    n_per_cluster = n_samples // 4
    centers = [(-1, -1), (1, 1), (-1, 1), (1, -1)]
    labels = [0, 0, 1, 1]

    parts_X, parts_y = [], []
    for center, label in zip(centers, labels):
        parts_X.append(rng.normal(loc=center, scale=noise, size=(n_per_cluster, 2)))
        parts_y.append(np.full(n_per_cluster, label))

    X = np.vstack(parts_X)
    y = np.concatenate(parts_y)
    return _shuffle(X, y, rng)


def make_sinusoidal(
    n_samples: int = 200,
    noise: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Frontière sinusoïdale : la classe dépend du signe de y - sin(2x).

    Points tirés uniformément dans [-2, 2]², classés selon la position
    relativement à la courbe y = sin(2x). Un peu de bruit est ajouté
    sur la coordonnée y pour éviter une frontière parfaite.

    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(low=-2.0, high=2.0, size=(n_samples, 2))
    # Bruit appliqué à la décision (pas aux points eux-mêmes)
    decision_noise = rng.normal(loc=0.0, scale=noise, size=n_samples)
    y = ((X[:, 1] + decision_noise) > np.sin(2 * X[:, 0])).astype(int)
    return _shuffle(X, y, rng)


def make_islands(
    n_samples: int = 200,
    noise: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Plusieurs petits îlots gaussiens mélangés entre les deux classes.
    6 îlots placés de façon non séparable linéairement :
    classe 0 et classe 1 alternent en damier irrégulier.

    """
    rng = np.random.default_rng(seed)
    centers = [
        ((-1.5, -1.5), 0),
        (( 0.0, -1.0), 1),
        (( 1.5, -1.5), 0),
        ((-1.0,  1.0), 1),
        (( 1.0,  1.0), 0),
        (( 0.0,  1.5), 1),
    ]
    n_per_island = n_samples // len(centers)

    parts_X, parts_y = [], []
    for (cx, cy), label in centers:
        parts_X.append(rng.normal(loc=[cx, cy], scale=noise, size=(n_per_island, 2)))
        parts_y.append(np.full(n_per_island, label))

    X = np.vstack(parts_X)
    y = np.concatenate(parts_y)
    return _shuffle(X, y, rng)


def make_spirals(
    n_samples: int = 200,
    noise: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Deux spirales entrelacées (boss final).

    """
    rng = np.random.default_rng(seed)
    n_per_class = n_samples // 2
    t = np.linspace(0.0, 1.0, n_per_class)
    angle = 4 * np.pi * t  # 2 tours

    # Spirale classe 0
    r0 = t
    X0 = np.stack([r0 * np.cos(angle), r0 * np.sin(angle)], axis=1)
    # Spirale classe 1 (décalée de π)
    X1 = np.stack([r0 * np.cos(angle + np.pi), r0 * np.sin(angle + np.pi)], axis=1)

    # Bruit gaussien sur les coordonnées
    X0 = X0 + rng.normal(scale=noise * 0.3, size=X0.shape)
    X1 = X1 + rng.normal(scale=noise * 0.3, size=X1.shape)

    # Mise à l'échelle pour rester dans une plage comparable aux autres datasets
    X = np.vstack([X0, X1]) * 2.0
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return _shuffle(X, y, rng)


# ─────────────────────────────────────────────
# Dispatcher : permet d'appeler un dataset par son nom
# ─────────────────────────────────────────────
DATASETS = {
    "Two Gaussians":   make_gaussians,
    "Moons":           make_moons,
    "Circles":         make_circles,
    "XOR":             make_xor,
    "Sinusoidal":      make_sinusoidal,
    "Islands":         make_islands,
    "Spirals":         make_spirals,
}


def get_dataset(
    name: str,
    n_samples: int = 200,
    noise: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Génère le dataset demandé par son nom.

    Paramètres
    ----------
    name : str
        Nom du dataset (clé dans DATASETS)
    n_samples : int
        Nombre total de points
    noise : float
        Niveau de bruit (interprétation spécifique à chaque dataset)
    seed : int
        Graine aléatoire pour la reproductibilité
    """
    if name not in DATASETS:
        raise ValueError(f"Dataset inconnu : '{name}'. Choisir parmi : {list(DATASETS.keys())}")
    return DATASETS[name](n_samples=n_samples, noise=noise, seed=seed)


# ─────────────────────────────────────────────
# Conversion en DataLoader PyTorch
# ─────────────────────────────────────────────
def to_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    """
    Convertit des arrays numpy en DataLoader PyTorch.
    """
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y).unsqueeze(1)  # shape (n, 1) pour BCEWithLogitsLoss

    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
