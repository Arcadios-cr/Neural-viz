import numpy as np
import torch
from sklearn.datasets import (
    make_moons as sk_make_moons,
    make_circles as sk_make_circles,
    make_blobs as sk_make_blobs,
    make_classification as sk_make_classification,
    make_gaussian_quantiles as sk_make_gaussian_quantiles,
)
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


def make_overlap(
    n_samples: int = 200,
    noise: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Deux gaussiennes volontairement CHEVAUCHANTES (classes emmêlées).

    Les deux centres sont proches (-0.7, 0) et (0.7, 0), et l'écart-type
    augmente avec `noise`. Au-delà d'un certain bruit, les deux nuages se
    recouvrent : il n'existe alors AUCUNE frontière parfaite (l'erreur
    minimale possible est > 0).

    C'est le dataset idéal pour observer :
      - la "dentelle" : sans régularisation, le réseau crée des frontières
        tarabiscotées pour capturer les points isolés de la classe adverse
        (overfitting) ;
      - l'effet du dropout : avec régularisation, la frontière redevient lisse ;
      - la généralisation : l'écart entre la performance sur le train et sur
        le test grandit quand le réseau fait trop de dentelle.
    """
    rng = np.random.default_rng(seed)
    n_per_class = n_samples // 2
    # Plus de bruit => écart-type plus grand => chevauchement plus fort.
    scale = 0.4 + noise
    X0 = rng.normal(loc=[-0.7, 0.0], scale=scale, size=(n_per_class, 2))
    X1 = rng.normal(loc=[0.7, 0.0],  scale=scale, size=(n_per_class, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return _shuffle(X, y, rng)


def make_blobs(
    n_samples: int = 200,
    noise: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Deux blobs gaussiens (sklearn make_blobs), chevauchement piloté par `noise`.

    `cluster_std` augmente avec le bruit → les deux nuages se recouvrent de plus
    en plus. Avantage : la même mécanique se généralise directement à K classes
    (il suffira d'ajouter des centres) pour le futur module multi-classes.
    """
    centers = [[-1.5, 0.0], [1.5, 0.0]]
    cluster_std = 0.5 + 1.2 * noise
    X, y = sk_make_blobs(
        n_samples=n_samples, centers=centers,
        cluster_std=cluster_std, random_state=seed,
    )
    rng = np.random.default_rng(seed)
    return _shuffle(X, y, rng)


def make_classif(
    n_samples: int = 200,
    noise: float = 0.3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Dataset de classification standard (sklearn make_classification).

    `noise` pilote DEUX sources de difficulté à la fois :
      - `class_sep` (séparation des classes) qui DIMINUE avec le bruit
        → les classes se rapprochent et se chevauchent ;
      - `flip_y` (bruit d'étiquette) qui AUGMENTE avec le bruit
        → certains points sont volontairement mal étiquetés (points aberrants).
    Idéal pour observer si le réseau généralise ou s'il sur-apprend les aberrants.
    """
    X, y = sk_make_classification(
        n_samples=n_samples,
        n_features=2, n_informative=2, n_redundant=0, n_repeated=0,
        n_clusters_per_class=1,
        class_sep=max(0.2, 2.0 - 1.8 * noise),
        flip_y=0.10 * noise,
        random_state=seed,
    )
    rng = np.random.default_rng(seed)
    return _shuffle(X, y, rng)


def make_quantiles(
    n_samples: int = 200,
    noise: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Classes en anneaux concentriques gaussiens (sklearn make_gaussian_quantiles).

    Le cœur du nuage est une classe, la couronne extérieure l'autre — séparation
    par un cercle, mais avec un recouvrement gaussien naturel. `noise` ajoute du
    bruit sur les coordonnées pour accentuer le chevauchement. Se généralise
    aussi à K classes (anneaux successifs).
    """
    X, y = sk_make_gaussian_quantiles(
        n_samples=n_samples, n_features=2, n_classes=2, random_state=seed,
    )
    rng = np.random.default_rng(seed)
    X = X + rng.normal(scale=noise * 0.3, size=X.shape)
    return _shuffle(X, y, rng)


def make_checkerboard(
    n_samples: int = 200,
    noise: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Damier (échiquier) : la classe alterne comme les cases d'un échiquier.

    Sur le plan [-2, 2]², chaque case unité a une classe selon la parité
    (⌊x⌋ + ⌊y⌋) % 2. Cela crée de nombreuses régions alternées, donc le réseau
    doit apprendre BEAUCOUP de frontières linéaires (= beaucoup de neurones en
    première couche). `noise` déplace légèrement les points. Excellent cas pour
    étudier la capacité du réseau et le contrôle de la première couche.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2.0, 2.0, size=(n_samples, 2))
    gx = np.floor(X[:, 0])
    gy = np.floor(X[:, 1])
    y = ((gx + gy) % 2).astype(int)
    X = X + rng.normal(scale=noise * 0.2, size=X.shape)
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
    Quatre îles (classe 0) entourées par leur océan respectif (classe 1).

    Géométrie inspirée d'une vraie île tropicale :
        - 4 coins du plan : (-1.5, 1.5), (1.5, 1.5), (-1.5, -1.5), (1.5, -1.5)
        - Pour chaque coin :
            • ÎLE (classe 0, bleue) = disque dense au centre du coin
              (gaussienne resserrée + rejet des points hors d'un rayon r_island_max)
            • PLAGE (zone vide) = anneau intermédiaire sans aucun point
              entre l'île et l'océan, pour garantir une séparation visuelle nette
            • OCÉAN (classe 1, rouge) = couronne autour de l'île
              (anneau de r_water_in à r_water_out, densité uniforme)

    Aucun mélange entre les classes grâce au rejet strict des points
    qui sortiraient de leur zone.

    Pour bien classifier, le réseau doit apprendre à isoler 4 disques fermés
    (les îles) à l'intérieur d'un fond de la classe opposée. Demande
    de la profondeur (≥ 2 couches cachées) pour fonctionner correctement.
    """
    rng = np.random.default_rng(seed)

    corners = [(-1.5, 1.5), (1.5, 1.5), (-1.5, -1.5), (1.5, -1.5)]

    # Répartition : 35% pour les îles (compactes), 65% pour les océans (étalés)
    n_per_corner = n_samples // 4
    n_island = int(n_per_corner * 0.35)
    n_water = n_per_corner - n_island

    # ─── Paramètres géométriques (le bruit module légèrement, sans casser la structure) ───
    # ÎLE : disque dense
    island_std = 0.12 + 0.03 * noise       # écart-type de la gaussienne
    r_island_max = 0.35                    # rayon de rejet strict de l'île

    # PLAGE : zone vide entre 0.35 et 0.60 (0.25 d'espace minimum, garanti)
    # OCÉAN : anneau
    r_water_in = 0.60                      # début de l'océan (au-delà de la plage)
    r_water_out = 1.35                     # fin de l'océan
    water_jitter = 0.03 + 0.02 * noise     # très léger bruit pour pas casser l'anneau

    parts_X, parts_y = [], []
    for (cx, cy) in corners:
        # ─── ÎLE (classe 0) ───────────────────────────────────────
        # On sur-échantillonne puis on rejette les points hors du rayon max
        # pour garantir un disque bien fermé, sans débordement vers la mer.
        oversample = max(n_island * 3, 30)
        candidates = rng.normal(loc=[cx, cy], scale=island_std, size=(oversample, 2))
        distances = np.linalg.norm(candidates - np.array([cx, cy]), axis=1)
        valid = candidates[distances < r_island_max]
        # On garde les n_island premiers points valides (toujours assez grâce à l'oversample)
        island = valid[:n_island]
        parts_X.append(island)
        parts_y.append(np.zeros(len(island), dtype=int))

        # ─── OCÉAN (classe 1) ──────────────────────────────────────
        # Anneau à densité uniforme : on tire r² uniformément puis racine carrée.
        r2 = rng.uniform(r_water_in ** 2, r_water_out ** 2, size=n_water)
        r = np.sqrt(r2)
        theta = rng.uniform(0.0, 2 * np.pi, size=n_water)
        water = np.stack(
            [cx + r * np.cos(theta), cy + r * np.sin(theta)],
            axis=1,
        )
        # Très léger bruit gaussien pour naturaliser, MAIS on rejette tout point
        # qui retomberait dans la plage (distance < r_water_in - marge)
        water = water + rng.normal(scale=water_jitter, size=water.shape)
        distances_water = np.linalg.norm(water - np.array([cx, cy]), axis=1)
        keep = distances_water >= (r_water_in - 0.02)   # marge de sécurité
        water = water[keep]

        parts_X.append(water)
        parts_y.append(np.ones(len(water), dtype=int))

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
    "Two Gaussians":      make_gaussians,
    "Overlap":            make_overlap,
    "Blobs":              make_blobs,
    "Classification":     make_classif,
    "Gaussian Quantiles": make_quantiles,
    "Checkerboard":       make_checkerboard,
    "Moons":              make_moons,
    "Circles":            make_circles,
    "XOR":                make_xor,
    "Sinusoidal":         make_sinusoidal,
    "Islands":            make_islands,
    "Spirals":            make_spirals,
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


# ─────────────────────────────────────────────
# Split train / validation / test
# ─────────────────────────────────────────────
def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    """
    Découpe (X, y) en trois sous-ensembles disjoints : train, validation, test.

    Les ratios doivent sommer à <= 1. Le test reçoit le reste : `1 - train_ratio - val_ratio`.

    Paramètres
    ----------
    X : np.ndarray (n, 2)
    y : np.ndarray (n,)
    train_ratio : float
        Fraction du dataset allouée à l'entraînement (par défaut 0.6).
    val_ratio : float
        Fraction allouée à la validation (par défaut 0.2).
        Le test reçoit `1 - train_ratio - val_ratio`.
    seed : int
        Graine pour la reproductibilité du split.

    Retourne
    --------
    ((X_train, y_train), (X_val, y_val), (X_test, y_test))
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio doit être dans ]0, 1[.")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio doit être dans [0, 1[.")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio doit être strictement inférieur à 1 (sinon pas de test).")

    rng = np.random.default_rng(seed)
    n = len(y)
    indices = rng.permutation(n)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # n_test reçoit le reste pour éviter les arrondis perdus

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return (
        (X[train_idx], y[train_idx]),
        (X[val_idx],   y[val_idx]),
        (X[test_idx],  y[test_idx]),
    )
