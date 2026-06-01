import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from models.mlp import MLP
from models.mlp_bottleneck import MLPBottleneck
from data.datasets import DATASETS, get_dataset, split_dataset, to_dataloader
from training.trainer import Trainer, EarlyStopping
from training.metrics import evaluate
from utils.hooks import ActivationCapture


# ─────────────────────────────────────────────
# Configuration de la page
# ─────────────────────────────────────────────
st.set_page_config(page_title="Neural-Viz", layout="wide")
st.title("Visualisation d'un réseau de neurones MLP")


# ─────────────────────────────────────────────
# Sidebar — paramètres
# ─────────────────────────────────────────────
st.sidebar.header("Architecture du réseau")

n_hidden_layers = st.sidebar.slider(
    "Nombre de couches cachées", 1, 8, 1,
    help=(
        "Jusqu'à 8 couches. Augmenter la profondeur permet d'observer "
        "l'aggravation exponentielle du vanishing gradient (visible via "
        "« Suivre les gradients par couche »)."
    ),
)
neurons_per_layer = st.sidebar.slider("Neurones par couche cachée", 2, 64, 8)
activation = st.sidebar.selectbox("Fonction d'activation", ["relu", "tanh", "sigmoid"])
use_batchnorm = st.sidebar.checkbox("Utiliser BatchNorm", value=False)
dropout_rate = st.sidebar.slider(
    "Dropout (régularisation)",
    0.0, 0.7, 0.0, step=0.05,
    help=(
        "Probabilité d'éteindre aléatoirement chaque neurone à chaque pas "
        "d'entraînement. Réduit l'overfitting. 0 = désactivé. "
        "Valeurs typiques : 0.1 à 0.3."
    ),
)
use_bottleneck = st.sidebar.checkbox(
    "Espace latent (bottleneck)",
    value=False,
    help=(
        "Insère un goulot d'étranglement au milieu du réseau. "
        "Permet de visualiser comment le réseau 'redessine' les données."
    ),
)
bottleneck_dim = st.sidebar.radio(
    "Dimension de l'espace latent",
    [2, 3],
    index=0,
    horizontal=True,
    help=(
        "2D : scatter plot statique (matplotlib).\n"
        "3D : scatter plot interactif rotatif (Plotly) — permet de tourner "
        "autour de l'espace latent à la souris pour voir tous les angles."
    ),
    disabled=not use_bottleneck,
)
funnel_encoder = st.sidebar.checkbox(
    "Encoder en entonnoir",
    value=False,
    help=(
        "Au lieu d'avoir toutes les couches encoder de la même taille, "
        "les tailles décroissent : [n, n/2, n/4, ...]."
    ),
    disabled=not use_bottleneck,
)
head_layers = st.sidebar.slider(
    "Tête (décodeur) — nombre de couches",
    1, 4, 2,
    help=(
        "Nombre de couches cachées après le bottleneck. Par défaut la tête "
        "est plus petite que l'encoder, comme un vrai décodeur."
    ),
    disabled=not use_bottleneck,
)
head_neurons = st.sidebar.slider(
    "Tête — neurones par couche",
    2, 64, 16,
    help="Nombre de neurones dans chaque couche du décodeur.",
    disabled=not use_bottleneck,
)

st.sidebar.header("Visualisation")
viz_mode = st.sidebar.radio(
    "Affichage de la sortie du réseau",
    ["Logits (sortie brute)", "Probabilités (sigmoid)"],
    index=0,
)
live_training = st.sidebar.checkbox("Entraînement en temps réel", value=True)
track_gradients = st.sidebar.checkbox(
    "Suivre les gradients par couche",
    value=False,
    help=(
        "Capture la norme du gradient de chaque couche à chaque époque. "
        "Permet de diagnostiquer les vanishing gradients (gradients qui "
        "s'écrasent vers les couches d'entrée) et les exploding gradients. "
        "Très parlant en comparant ReLU vs sigmoid/tanh sur un réseau profond."
    ),
)
boundary_refresh = st.sidebar.slider(
    "Rafraîchir la frontière toutes les N époques",
    1, 50, 10,
    help="Plus petit = plus fluide mais plus lent. Plus grand = plus rapide.",
)

st.sidebar.header("Entraînement")
n_epochs = st.sidebar.slider("Nombre d'époques (max)", 10, 500, 100)
optimizer_name = st.sidebar.selectbox(
    "Optimiseur",
    ["Adam", "SGD"],
    index=0,
    help=(
        "Adam : adaptatif, converge vite, mais masque les vanishing gradients "
        "(il normalise les gradients par leur historique).\n"
        "SGD : descente de gradient simple, le pas est directement proportionnel "
        "à la norme du gradient → idéal pour observer les vanishing gradients."
    ),
)
learning_rate = st.sidebar.select_slider(
    "Taux d'apprentissage",
    options=[0.001, 0.005, 0.01, 0.05, 0.1],
    value=0.01,
)
batch_size = st.sidebar.slider("Batch size", 8, 64, 32)

st.sidebar.subheader("Split train / validation / test")
train_ratio = st.sidebar.slider("Ratio train", 0.4, 0.8, 0.6, step=0.05)
val_ratio = st.sidebar.slider("Ratio validation", 0.1, 0.4, 0.2, step=0.05)
test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)
st.sidebar.caption(f"Ratio test (calculé) : **{test_ratio:.2f}**")

st.sidebar.subheader("Early stopping")
use_early_stopping = st.sidebar.checkbox("Activer l'early stopping", value=True)
patience = st.sidebar.slider(
    "Patience (époques sans amélioration)", 1, 50, 15,
    disabled=not use_early_stopping,
    help="L'entraînement s'arrête si la val loss ne s'améliore plus pendant N époques.",
)
min_delta = st.sidebar.select_slider(
    "Min delta (amélioration minimale)",
    options=[0.0, 1e-5, 1e-4, 1e-3, 1e-2],
    value=1e-4,
    disabled=not use_early_stopping,
)

st.sidebar.header("Dataset")
dataset_name = st.sidebar.selectbox(
    "Type de dataset",
    list(DATASETS.keys()),
    index=0,
    help=(
        "⭐ Two Gaussians — trivial (linéairement séparable)\n"
        "🔀 Chevauchants (Overlap / Blobs / Classification / Gaussian Quantiles) "
        "— classes qui se recouvrent, pas de frontière parfaite : idéal pour "
        "observer l'overfitting / la dentelle et la généralisation\n"
        "🏁 Checkerboard — damier : demande beaucoup de frontières (de plans)\n"
        "⭐⭐ Moons / Circles / XOR — non linéairement séparables, simples\n"
        "⭐⭐⭐ Sinusoidal / Islands — frontières complexes\n"
        "⭐⭐⭐⭐ Spirals — le boss final"
    ),
)
n_samples = st.sidebar.slider("Nombre de points", 100, 500, 200)
noise = st.sidebar.slider("Niveau de bruit", 0.0, 1.0, 0.2, step=0.05)
seed = st.sidebar.number_input("Seed (données)", value=42, step=1)
weight_seed = st.sidebar.number_input(
    "Seed (initialisation des poids)",
    value=0, step=1,
    help=(
        "Fixe l'initialisation aléatoire des poids du réseau ET l'ordre de "
        "mélange des batches. À seed identique, deux entraînements donnent "
        "exactement le même résultat (reproductibilité). Changer ce seed en "
        "gardant le reste identique permet d'observer la variance due à "
        "l'initialisation."
    ),
)


# ─────────────────────────────────────────────
# Génération du dataset + split train / val / test
# ─────────────────────────────────────────────
X, y = get_dataset(dataset_name, n_samples=n_samples, noise=noise, seed=int(seed))
(X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(
    X, y,
    train_ratio=train_ratio,
    val_ratio=val_ratio,
    seed=int(seed),
)
train_loader = to_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
val_loader   = to_dataloader(X_val,   y_val,   batch_size=batch_size, shuffle=False)
# test_loader sera utilisé pour les métriques finales (mercredi)
test_loader  = to_dataloader(X_test,  y_test,  batch_size=batch_size, shuffle=False)


# ─────────────────────────────────────────────
# Construction du modèle
# ─────────────────────────────────────────────
# On fixe le générateur aléatoire de PyTorch juste avant de créer le modèle :
# l'initialisation des poids des couches Linear devient ainsi reproductible.
torch.manual_seed(int(weight_seed))

if use_bottleneck:
    # ─── Encoder : symétrique OU en entonnoir ───
    if funnel_encoder:
        # Sizes décroissantes : [n, max(n/2, 4), max(n/4, 4), ...]
        encoder_layers = []
        size = neurons_per_layer
        for _ in range(n_hidden_layers):
            encoder_layers.append(max(size, 4))   # min 4 neurones pour garder du sens
            size = size // 2
    else:
        encoder_layers = [neurons_per_layer] * n_hidden_layers

    # ─── Décodeur (la "tête") : indépendamment configuré ───
    decoder_layers = [head_neurons] * head_layers

    model = MLPBottleneck(
        input_dim=2,
        encoder_layers=encoder_layers,
        bottleneck_dim=bottleneck_dim,
        decoder_layers=decoder_layers,
        output_dim=1,
        activation=activation,
        use_batchnorm=use_batchnorm,
        dropout_rate=dropout_rate,
    )
else:
    hidden_layers = [neurons_per_layer] * n_hidden_layers
    model = MLP(
        input_dim=2,
        hidden_layers=hidden_layers,
        output_dim=1,
        activation=activation,
        use_batchnorm=use_batchnorm,
        dropout_rate=dropout_rate,
    )

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Modèle :** `{model}`")


# ─────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────
def plot_decision_boundary(model, X, y, mode: str = "logits"):
    """
    Affiche le scatter plot des données et la frontière de décision du réseau.

    Paramètres
    ----------
    mode : str
        "logits" : affiche la sortie brute du réseau (la fonction R²→R réellement apprise)
        "probas" : applique la sigmoid pour afficher des probabilités entre 0 et 1
    """
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # Grille de points dans R²
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

    # Prédiction sur la grille
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(grid)).numpy().reshape(xx.shape)

    if mode == "probas":
        Z = 1 / (1 + np.exp(-logits))   # sigmoid
        boundary_level = 0.5
        cbar_label = "P(classe 1)"
        # Échelle fixe [0, 1]
        vmin, vmax = 0.0, 1.0
    else:
        Z = logits
        boundary_level = 0.0
        cbar_label = "Logits (sortie brute)"
        # Échelle symétrique autour de 0 pour bien visualiser
        amax = max(abs(Z.min()), abs(Z.max()))
        vmin, vmax = -amax, amax

    # Plot
    fig, ax = plt.subplots(figsize=(7, 6))

    contour = ax.contourf(xx, yy, Z, levels=50, cmap="RdYlBu_r", alpha=0.8, vmin=vmin, vmax=vmax)
    plt.colorbar(contour, ax=ax, label=cbar_label)

    # Frontière de décision
    ax.contour(xx, yy, Z, levels=[boundary_level], colors="black", linewidths=1.5)

    # Points d'entraînement
    colors = ["#2196F3", "#F44336"]
    for cls in [0, 1]:
        mask = y == cls
        ax.scatter(
            X[mask, 0], X[mask, 1],
            c=colors[cls],
            edgecolors="white",
            linewidths=0.5,
            s=40,
            label=f"Classe {cls}",
            zorder=3,
        )

    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_title(f"Frontière de décision ({'logits' if mode == 'logits' else 'probabilités'})")
    ax.legend()
    return fig


def plot_latent_space(model: MLPBottleneck, X: np.ndarray, y: np.ndarray):
    """
    Affiche la représentation 2D de X dans l'espace latent (bottleneck)
    appris par le réseau.

    Chaque point d'entrée est encodé en (z1, z2) et affiché dans cet
    espace, coloré par sa vraie classe. Permet de voir comment le réseau
    a "redessiné" les données pour les rendre faciles à classifier.
    """
    model.eval()
    with torch.no_grad():
        Z = model.encode(torch.tensor(X, dtype=torch.float32)).numpy()

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#2196F3", "#F44336"]
    for cls in [0, 1]:
        mask = y == cls
        ax.scatter(
            Z[mask, 0], Z[mask, 1],
            c=colors[cls],
            edgecolors="white",
            linewidths=0.5,
            s=40,
            label=f"Classe {cls}",
        )

    # Ligne pointillée à z1=0 et z2=0 pour repérer l'origine
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    ax.set_xlabel("z₁ (1er neurone du bottleneck)")
    ax.set_ylabel("z₂ (2ème neurone du bottleneck)")
    ax.set_title("Espace latent — représentation 2D apprise par l'encoder")
    ax.legend()
    ax.set_aspect("equal", adjustable="datalim")
    return fig


def plot_latent_space_3d(model: MLPBottleneck, X: np.ndarray, y: np.ndarray):
    """
    Comme plot_latent_space() mais pour un bottleneck à 3 dimensions.

    Renvoie une figure Plotly 3D interactive. L'utilisateur peut tourner autour de l'espace latent pour
    découvrir les angles où les classes apparaissent comme séparées.

    """
    import plotly.graph_objects as go

    model.eval()
    with torch.no_grad():
        Z = model.encode(torch.tensor(X, dtype=torch.float32)).numpy()
    # Z a shape (n, 3)

    colors_pts = ["#2196F3", "#F44336"]
    fig = go.Figure()
    for cls in [0, 1]:
        mask = y == cls
        fig.add_trace(go.Scatter3d(
            x=Z[mask, 0],
            y=Z[mask, 1],
            z=Z[mask, 2],
            mode="markers",
            name=f"Classe {cls}",
            marker=dict(
                size=4,
                color=colors_pts[cls],
                line=dict(color="white", width=0.5),
                opacity=0.85,
            ),
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title="z₁",
            yaxis_title="z₂",
            zaxis_title="z₃",
            aspectmode="data",
        ),
        title="Espace latent 3D",
        margin=dict(l=0, r=0, t=40, b=0),
        height=550,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_activations_grid(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    grid_resolution: int = 100,
    max_cols: int = 8,
):
    """
    Pour chaque couche cachée du modèle, affiche une heatmap par neurone
    qui montre dans quelle région du plan ce neurone s'active.

    On utilise les forward hooks PyTorch via ActivationCapture pour
    intercepter les sorties de chaque couche d'activation (ReLU/Tanh/Sigmoid)
    quand on passe une grille 2D dans le réseau.

    Paramètres
    ----------
    model : nn.Module
        Le modèle entraîné (MLP ou MLPBottleneck).
    X : np.ndarray, shape (n, 2)
        Points d'entraînement (juste pour calibrer les bornes du plan
        et superposer les données sur les heatmaps).
    y : np.ndarray, shape (n,)
        Labels associés.
    grid_resolution : int
        Résolution de la grille (la grille fera grid_resolution x grid_resolution
        points). 100 par défaut — assez fin sans être trop lourd à calculer.
    max_cols : int
        Nombre maximum de neurones affichés par ligne (8 par défaut).

    Retourne
    --------
    list[tuple[str, matplotlib.figure.Figure]]
        Une liste de (nom_de_couche, figure) — une figure par couche.
    """
    # ─── Préparation de la grille 2D ───
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution),
    )
    grid_np = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    grid_t = torch.tensor(grid_np)

    # ─── Capture des activations via forward hooks ───
    # ActivationCapture attache automatiquement un hook sur chaque
    # ReLU/Tanh/Sigmoid du réseau et bascule en eval() pendant la capture
    with ActivationCapture(model) as cap:
        activations = cap.get_activations(grid_t)

    figures = []
    colors_pts = ["#2196F3", "#F44336"]

    for layer_name, act_tensor in activations.items():
        # act_tensor : shape (grid_resolution², n_neurones_de_la_couche)
        n_neurons = act_tensor.shape[1]
        n_cols = min(max_cols, n_neurons)
        n_rows = (n_neurons + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(2.2 * n_cols, 2.2 * n_rows),
            squeeze=False,
        )

        # Pour avoir une échelle de couleur cohérente entre tous les neurones
        # d'une même couche, on utilise les min/max globaux de la couche.
        layer_min = act_tensor.min().item()
        layer_max = act_tensor.max().item()
        amax = max(abs(layer_min), abs(layer_max))
        if amax == 0:
            amax = 1.0   # évite division par zéro pour les couches mortes

        for neuron_idx in range(n_neurons):
            r, c = divmod(neuron_idx, n_cols)
            ax = axes[r][c]
            Z = act_tensor[:, neuron_idx].numpy().reshape(xx.shape)
            ax.imshow(
                Z,
                extent=(x_min, x_max, y_min, y_max),
                origin="lower",
                cmap="RdYlBu_r",
                vmin=-amax, vmax=amax,
                aspect="auto",
            )
            # Superposition discrète des points d'entraînement
            for cls in [0, 1]:
                m = y == cls
                ax.scatter(X[m, 0], X[m, 1], c=colors_pts[cls],
                           s=4, alpha=0.7, edgecolors="none")
            ax.set_title(f"#{neuron_idx}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

        # Désactive les axes inutilisés (si le dernier rang n'est pas plein)
        for empty_idx in range(n_neurons, n_rows * n_cols):
            r, c = divmod(empty_idx, n_cols)
            axes[r][c].axis("off")

        fig.suptitle(f"Couche cachée : {layer_name}", fontsize=12)
        fig.tight_layout()
        figures.append((layer_name, fig))

    return figures


def plot_weight_heatmaps(model: nn.Module, max_cols: int = 6):
    """
    Pour chaque couche Linear du modèle, affiche une heatmap de la matrice
    de poids (et un mini-graphique des biais à côté).

    Lecture :
        - Chaque ligne de la heatmap  = un neurone de SORTIE de la couche
        - Chaque colonne              = un neurone d'ENTRÉE (ou feature d'entrée)
        - Couleur :  rouge intense = poids positif fort
                     bleu intense  = poids négatif fort
                     blanc         = poids proche de zéro (connexion faible)
        - Échelle symétrique autour de 0 pour bien voir les signes.

    Retourne
    --------
    list[tuple[str, matplotlib.figure.Figure]]
        Une liste de (nom_de_couche, figure) — une figure par couche Linear.
    """
    figures = []
    linear_idx = 0

    for module in model.modules():
        if not isinstance(module, nn.Linear):
            continue

        W = module.weight.detach().cpu().numpy()        # shape (out, in)
        b = module.bias.detach().cpu().numpy() if module.bias is not None else None
        n_out, n_in = W.shape

        # Échelle symétrique autour de 0 pour bien visualiser les signes
        amax = max(abs(W.min()), abs(W.max()), 1e-8)

        # Figure avec 2 sous-graphes : la matrice de poids + les biais en colonne
        fig, (ax_w, ax_b) = plt.subplots(
            1, 2,
            figsize=(max(5, 0.4 * n_in + 3), max(3, 0.4 * n_out + 1.5)),
            gridspec_kw={"width_ratios": [n_in, 1]},
        )

        # ─── Matrice de poids ───
        im = ax_w.imshow(W, cmap="RdBu_r", vmin=-amax, vmax=amax, aspect="auto")
        ax_w.set_xlabel(f"Neurone d'entrée (0 → {n_in - 1})")
        ax_w.set_ylabel(f"Neurone de sortie (0 → {n_out - 1})")
        ax_w.set_xticks(range(n_in))
        ax_w.set_yticks(range(n_out))

        # Annoter chaque case si la matrice n'est pas trop grande
        if n_in * n_out <= 256:
            for i in range(n_out):
                for j in range(n_in):
                    val = W[i, j]
                    color = "white" if abs(val) > 0.6 * amax else "black"
                    ax_w.text(j, i, f"{val:.2f}", ha="center", va="center",
                              color=color, fontsize=7)

        plt.colorbar(im, ax=ax_w, label="valeur du poids", fraction=0.03, pad=0.02)

        # ─── Biais (colonne) ───
        if b is not None:
            b_max = max(abs(b.min()), abs(b.max()), 1e-8)
            ax_b.imshow(b.reshape(-1, 1), cmap="RdBu_r",
                        vmin=-b_max, vmax=b_max, aspect="auto")
            ax_b.set_title("biais", fontsize=9)
            ax_b.set_xticks([])
            ax_b.set_yticks(range(n_out))
            for i in range(n_out):
                color = "white" if abs(b[i]) > 0.6 * b_max else "black"
                ax_b.text(0, i, f"{b[i]:.2f}", ha="center", va="center",
                          color=color, fontsize=7)
        else:
            ax_b.axis("off")

        fig.suptitle(
            f"Couche Linear #{linear_idx}  —  poids ({n_out} × {n_in}) + biais",
            fontsize=11,
        )
        fig.tight_layout()
        figures.append((f"linear_{linear_idx}", fig))
        linear_idx += 1

    return figures


def plot_first_layer_lines(model: nn.Module, X: np.ndarray, y: np.ndarray):
    """
    Visualise les "droites de séparation" apprises par chaque neurone de la
    première couche Linear, superposées au scatter plot des données.

    Pour la première couche, chaque neurone calcule :
        z = w₁ · x₁ + w₂ · x₂ + b
    puis applique l'activation. La frontière où z = 0 est l'équation d'une
    droite dans le plan d'entrée. Cette fonction trace cette droite pour
    chaque neurone, ce qui révèle géométriquement ce que CHAQUE neurone
    de la première couche "regarde".

    Renvoie None si le modèle n'a pas de première couche Linear avec
    input_dim = 2 (pas applicable).
    """
    # Trouver la première couche Linear
    first_linear = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            first_linear = module
            break

    if first_linear is None or first_linear.in_features != 2:
        return None

    W = first_linear.weight.detach().cpu().numpy()       # (n_out, 2)
    b = first_linear.bias.detach().cpu().numpy() if first_linear.bias is not None \
        else np.zeros(W.shape[0])
    n_neurons = W.shape[0]

    fig, ax = plt.subplots(figsize=(7, 6))

    # ─── Scatter des données en arrière-plan ───
    colors_pts = ["#2196F3", "#F44336"]
    for cls in [0, 1]:
        m = y == cls
        ax.scatter(X[m, 0], X[m, 1], c=colors_pts[cls],
                   edgecolors="white", linewidths=0.4, s=30,
                   alpha=0.5, label=f"Classe {cls}", zorder=1)

    # ─── Une droite par neurone : w₁x + w₂y + b = 0 ───
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    x_range = np.linspace(x_min, x_max, 100)

    # Palette de couleurs distinctes pour les droites
    cmap = plt.cm.tab20 if n_neurons > 10 else plt.cm.tab10
    line_colors = cmap(np.linspace(0, 1, n_neurons))

    for i, (w, bi) in enumerate(zip(W, b)):
        w1, w2 = w
        # w₁ x + w₂ y + b = 0  →  y = -(w₁ x + b) / w₂
        if abs(w2) > 1e-6:
            y_line = -(w1 * x_range + bi) / w2
            # On ne dessine que la portion qui passe dans la zone visible
            mask = (y_line >= y_min) & (y_line <= y_max)
            if mask.any():
                ax.plot(x_range[mask], y_line[mask],
                        color=line_colors[i], linewidth=1.5,
                        alpha=0.85, label=f"neurone {i}", zorder=2)
        else:
            # Ligne quasi-verticale : w₁ x + b = 0  →  x = -b / w₁
            if abs(w1) > 1e-6:
                x_vert = -bi / w1
                if x_min <= x_vert <= x_max:
                    ax.axvline(x_vert, color=line_colors[i], linewidth=1.5,
                               alpha=0.85, label=f"neurone {i}", zorder=2)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_title(
        f"Droites apprises par la 1ère couche ({n_neurons} neurones)\n"
        "Chaque droite = la frontière où un neurone passe de inactif à actif"
    )
    # Légende compacte si beaucoup de neurones
    if n_neurons <= 8:
        ax.legend(fontsize=8, loc="best")
    else:
        ax.legend(fontsize=6, loc="best", ncol=2)
    return fig


def plot_gradient_flow(grad_history: dict):
    """
    Visualise le flux des gradients pour diagnostiquer les vanishing /
    exploding gradients.

    Produit deux sous-graphiques :
      1. Évolution temporelle : norme L2 du gradient de chaque couche au fil
         des époques (échelle log en y). Permet de voir si les gradients
         s'effondrent ou explosent pendant l'entraînement.
      2. Profil par profondeur : norme moyenne du gradient par couche, de
         l'entrée (L0) vers la sortie. Un vanishing gradient se voit comme
         une décroissance forte vers les couches d'entrée.

    Paramètres
    ----------
    grad_history : dict[str, list[float]]
        Clé = label de couche ("L0 (2→8)"), valeur = norme par époque.
        Les couches sont supposées ordonnées de l'entrée vers la sortie
        (ordre d'insertion, garanti par le Trainer).

    Retourne
    --------
    matplotlib.figure.Figure
    """
    labels = list(grad_history.keys())
    n_epochs = len(next(iter(grad_history.values())))
    epochs = np.arange(1, n_epochs + 1)

    fig, (ax_time, ax_depth) = plt.subplots(1, 2, figsize=(13, 5))

    # ─── 1. Évolution temporelle (log y) ───
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(labels)))
    for color, label in zip(cmap, labels):
        ax_time.plot(epochs, grad_history[label], label=label,
                     color=color, linewidth=1.5)
    ax_time.set_yscale("log")
    ax_time.set_xlabel("Époque")
    ax_time.set_ylabel("Norme L2 du gradient (échelle log)")
    ax_time.set_title("Évolution des gradients par couche")
    ax_time.legend(fontsize=8, loc="best")
    ax_time.grid(True, which="both", alpha=0.3)

    # ─── 2. Profil par profondeur (moyenne sur les dernières époques) ───
    # On moyenne sur le dernier 20% des époques pour lisser le bruit de fin
    # d'entraînement (au moins 1 époque).
    tail = max(1, n_epochs // 5)
    mean_norms = [float(np.mean(grad_history[label][-tail:])) for label in labels]

    bar_colors = plt.cm.viridis(np.linspace(0, 0.9, len(labels)))
    ax_depth.bar(range(len(labels)), mean_norms, color=bar_colors)
    ax_depth.set_yscale("log")
    ax_depth.set_xticks(range(len(labels)))
    ax_depth.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax_depth.set_ylabel("Norme L2 moyenne (échelle log)")
    ax_depth.set_title(f"Profil par profondeur (moyenne des {tail} dernières époques)")
    ax_depth.grid(True, which="both", axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


def diagnose_gradients(grad_history: dict) -> tuple[str, str]:
    """
    Produit un diagnostic textuel automatique à partir de l'historique des
    gradients (moyenne sur les dernières époques).

    Retourne (niveau, message) où niveau ∈ {"success", "warning", "error"}
    pour piloter l'affichage Streamlit (st.success / st.warning / st.error).
    """
    labels = list(grad_history.keys())
    n_epochs = len(next(iter(grad_history.values())))
    tail = max(1, n_epochs // 5)
    mean_norms = [float(np.mean(grad_history[label][-tail:])) for label in labels]

    input_norm = mean_norms[0]    # couche la plus proche de l'entrée (L0)
    output_norm = mean_norms[-1]  # couche de sortie
    max_norm = max(mean_norms)

    # ─── Exploding : norme très élevée quelque part ───
    if max_norm > 1e2:
        return (
            "error",
            f"⚠️ Exploding gradients possible : norme max = {max_norm:.1e}. "
            "Les gradients sont très grands — envisager un learning rate plus "
            "faible, du gradient clipping ou de la BatchNorm.",
        )

    # ─── Vanishing : le gradient s'atténue en remontant vers l'entrée ───
    # Indicateur direct : combien de fois le gradient s'écrase de la sortie
    # vers la couche d'entrée. C'est la signature même du vanishing gradient.
    depth_ratio = output_norm / input_norm if input_norm > 0 else float("inf")

    if depth_ratio > 50:
        return (
            "warning",
            f"⚠️ Vanishing gradient : le gradient s'écrase d'un facteur "
            f"≈ {depth_ratio:.0f} entre la sortie ({output_norm:.1e}) et "
            f"l'entrée ({input_norm:.1e}). Les premières couches apprennent "
            "très lentement. Typique des activations saturantes (sigmoid/tanh) "
            "sur un réseau profond — essayer ReLU ou la BatchNorm.",
        )

    return (
        "success",
        f"✅ Flux de gradient sain : le gradient reste du même ordre de "
        f"grandeur entre l'entrée ({input_norm:.1e}) et la sortie "
        f"({output_norm:.1e}), facteur d'atténuation ≈ {depth_ratio:.1f}. "
        "Pas de vanishing/exploding marqué.",
    )


# ─────────────────────────────────────────────
# Initialisation de st.session_state
# ─────────────────────────────────────────────
# st.session_state persiste entre les reruns de Streamlit (à chaque interaction
# avec un widget, le script entier est ré-exécuté). On y stocke les artefacts
# d'entraînement pour pouvoir les explorer après coup via le slider d'époque.
if "trainer" not in st.session_state:
    st.session_state.trainer = None
    st.session_state.trained_X = None
    st.session_state.trained_y = None
    st.session_state.trained_mode = None
    st.session_state.test_loader = None
    st.session_state.test_report = None


# ─────────────────────────────────────────────
# Layout principal
# ─────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Données d'entraînement")
    fig_data, ax_data = plt.subplots(figsize=(7, 6))
    colors = ["#2196F3", "#F44336"]
    # Train : ronds pleins, Val : triangles, Test : croix
    for cls in [0, 1]:
        m_train = y_train == cls
        m_val   = y_val   == cls
        m_test  = y_test  == cls
        ax_data.scatter(X_train[m_train, 0], X_train[m_train, 1],
                        c=colors[cls], edgecolors="white", linewidths=0.4,
                        s=40, marker="o", label=f"Train classe {cls}")
        ax_data.scatter(X_val[m_val, 0], X_val[m_val, 1],
                        c=colors[cls], edgecolors="black", linewidths=0.8,
                        s=50, marker="^", label=f"Val classe {cls}")
        ax_data.scatter(X_test[m_test, 0], X_test[m_test, 1],
                        c=colors[cls], edgecolors="black", linewidths=0.8,
                        s=55, marker="x", label=f"Test classe {cls}")
    ax_data.set_xlabel("x₁")
    ax_data.set_ylabel("x₂")
    ax_data.set_title(f"Dataset — {dataset_name}  (train={len(y_train)}, val={len(y_val)}, test={len(y_test)})")
    ax_data.legend(fontsize=7, loc="best")
    st.pyplot(fig_data)
    plt.close(fig_data)

with col2:
    st.subheader("Frontière de décision")

    boundary_placeholder = st.empty()
    status_placeholder = st.empty()

    if st.button("Entraîner le réseau", type="primary"):
        if optimizer_name == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion)

        # Early stopping (optionnel)
        early_stopping = (
            EarlyStopping(patience=patience, min_delta=min_delta)
            if use_early_stopping else None
        )

        mode = "logits" if viz_mode.startswith("Logits") else "probas"

        # Mise en place de la zone "courbe d'apprentissage" en bas
        st.subheader("Courbes train / validation")
        loss_placeholder = st.empty()

        def update_callback(epoch: int, train_loss: float, val_loss, current_model):
            """Callback appelé à chaque fin d'époque pendant l'entraînement."""
            msg = f"Époque {epoch + 1} / {n_epochs} — train = {train_loss:.4f}"
            if val_loss is not None:
                msg += f" | val = {val_loss:.4f}"
            status_placeholder.info(msg)

            # Frontière de décision : seulement toutes les N époques (lourd)
            if (epoch + 1) % boundary_refresh == 0 or epoch == n_epochs - 1:
                current_model.eval()
                fig = plot_decision_boundary(current_model, X_train, y_train, mode=mode)
                boundary_placeholder.pyplot(fig)
                plt.close(fig)
                current_model.train()

            # Courbe de loss : à chaque époque (léger)
            chart_data = {"train": trainer.history["train_loss"]}
            if trainer.history["val_loss"]:
                chart_data["validation"] = trainer.history["val_loss"]
            loss_placeholder.line_chart(chart_data)

        callback = update_callback if live_training else None
        # Re-fixe le générateur juste avant l'entraînement : rend déterministes
        # le mélange des batches (shuffle) et le dropout. Combiné au seed de
        # l'init des poids, l'entraînement entier est alors reproductible.
        torch.manual_seed(int(weight_seed))
        with st.spinner("Entraînement en cours..."):
            history = trainer.train(
                train_loader=train_loader,
                n_epochs=n_epochs,
                val_loader=val_loader,
                early_stopping=early_stopping,
                on_epoch_end=callback,
                restore_best=True,
                track_gradients=track_gradients,
            )

        # Rendu final (le modèle a été restauré au best epoch grâce à restore_best=True)
        model.eval()
        fig_boundary = plot_decision_boundary(model, X_train, y_train, mode=mode)
        boundary_placeholder.pyplot(fig_boundary)
        plt.close(fig_boundary)

        chart_data = {"train": history["train_loss"]}
        if history["val_loss"]:
            chart_data["validation"] = history["val_loss"]
        loss_placeholder.line_chart(chart_data)

        # Message de statut final
        msg_parts = [f"Entraînement terminé en {len(history['train_loss'])} époques"]
        if trainer.best_epoch >= 0:
            msg_parts.append(
                f"meilleure val_loss = {trainer.best_val_loss:.4f} "
                f"à l'époque {trainer.best_epoch + 1}"
            )
        if trainer.stopped_early:
            msg_parts.append("⚠️ early stopping déclenché")
        status_placeholder.success(" — ".join(msg_parts))

        # Évaluation finale sur le test set (jamais vu pendant l'entraînement)
        # On utilise le modèle restauré au best epoch (grâce à restore_best=True).
        test_report = evaluate(model, test_loader)

        # Persistance pour le slider d'exploration et l'affichage des métriques
        st.session_state.trainer = trainer
        st.session_state.trained_X = X_train
        st.session_state.trained_y = y_train
        st.session_state.trained_mode = mode
        st.session_state.test_loader = test_loader
        st.session_state.test_report = test_report
    else:
        # S'il n'y a pas eu d'entraînement encore, on affiche le message d'invite
        if st.session_state.trainer is None:
            boundary_placeholder.info(
                "Configure les paramètres dans la sidebar, puis clique sur **Entraîner le réseau**."
            )


# ─────────────────────────────────────────────
# Visualisation de l'espace latent (si le modèle est un MLPBottleneck)
# ─────────────────────────────────────────────
if (
    st.session_state.trainer is not None
    and isinstance(st.session_state.trainer.model, MLPBottleneck)
):
    st.markdown("---")
    st.subheader("Espace latent — comment le réseau a 'redessiné' les données")
    st.caption(
        "Chaque point d'entrée est encodé en 2D par la partie 'encoder' du réseau. "
        "On voit ici la position de chaque point d'entraînement dans l'espace latent. "
        "Sur des datasets complexes (Spirals, Islands), on observe que le réseau "
        "**déplie** les données pour les rendre linéairement séparables avant la "
        "classification finale."
    )

    bm = st.session_state.trainer.model
    bX = st.session_state.trained_X
    by = st.session_state.trained_y

    # Dimension réelle du bottleneck du modèle entraîné
    latent_dim = bm.bottleneck_dim

    col_in, col_lat = st.columns(2)
    with col_in:
        st.markdown("**Espace d'entrée** (donné au réseau)")
        fig_in, ax_in = plt.subplots(figsize=(6, 5))
        colors = ["#2196F3", "#F44336"]
        for cls in [0, 1]:
            m = by == cls
            ax_in.scatter(bX[m, 0], bX[m, 1], c=colors[cls],
                          edgecolors="white", linewidths=0.4, s=40,
                          label=f"Classe {cls}")
        ax_in.set_xlabel("x₁")
        ax_in.set_ylabel("x₂")
        ax_in.set_title("Données originales")
        ax_in.legend()
        ax_in.set_aspect("equal", adjustable="datalim")
        st.pyplot(fig_in)
        plt.close(fig_in)

    with col_lat:
        if latent_dim == 2:
            st.markdown("**Espace latent** (sortie du bottleneck 2D)")
            fig_lat = plot_latent_space(bm, bX, by)
            st.pyplot(fig_lat)
            plt.close(fig_lat)
        elif latent_dim == 3:
            st.markdown(
                "**Espace latent 3D** (sortie du bottleneck — rotative à la souris)"
            )
            fig_lat3d = plot_latent_space_3d(bm, bX, by)
            st.plotly_chart(fig_lat3d, use_container_width=True)
        else:
            st.warning(
                f"Bottleneck en dimension {latent_dim} non visualisable directement "
                "(seulement 2D et 3D supportés)."
            )


# ─────────────────────────────────────────────
# Visualisation des activations couche par couche (heatmaps)
# ─────────────────────────────────────────────
if st.session_state.trainer is not None:
    st.markdown("---")
    st.subheader("Activations couche par couche")
    st.caption(
        "Pour chaque couche cachée et chaque neurone, on affiche une **heatmap** "
        "qui montre dans quelle région du plan ce neurone s'active. Rouge = "
        "activation positive forte, bleu = activation négative forte, blanc = "
        "neurone inactif. Cela permet de voir **ce que chaque neurone individuel "
        "détecte** dans les données."
    )

    show_activations = st.checkbox(
        "Afficher les heatmaps d'activations",
        value=False,
        help=(
            "Calcule les activations sur une grille 100x100 puis affiche une "
            "heatmap par neurone. Peut être lent si le réseau a beaucoup de "
            "neurones — décoche pour économiser des ressources."
        ),
    )

    if show_activations:
        bm = st.session_state.trainer.model
        bX = st.session_state.trained_X
        by = st.session_state.trained_y

        with st.spinner("Calcul des activations..."):
            layer_figures = plot_activations_grid(bm, bX, by)

        if not layer_figures:
            st.info(
                "Aucune couche d'activation détectée (le modèle ne contient "
                "que des Linear ?). Les hooks s'attachent aux ReLU/Tanh/Sigmoid."
            )

        for layer_name, fig in layer_figures:
            st.pyplot(fig)
            plt.close(fig)


# ─────────────────────────────────────────────
# Visualisation des poids (heatmaps des matrices Linear)
# ─────────────────────────────────────────────
if st.session_state.trainer is not None:
    st.markdown("---")
    st.subheader("Poids appris par chaque couche")
    st.caption(
        "Les heatmaps ci-dessous montrent les **valeurs des poids** appris par "
        "chaque couche linéaire du réseau. **Rouge** = poids positif fort, "
        "**bleu** = poids négatif fort, **blanc** = connexion faible ou nulle. "
        "Pour la première couche, on affiche aussi les **droites de séparation** "
        "apprises par chaque neurone, superposées aux données — c'est la vision "
        "géométrique directe de ce que chaque neurone 'regarde'."
    )

    show_weights = st.checkbox(
        "Afficher les poids appris",
        value=False,
        help=(
            "Calcule et affiche les heatmaps des matrices de poids de toutes "
            "les couches Linear du réseau, plus la visualisation des droites "
            "de la première couche."
        ),
    )

    if show_weights:
        bm = st.session_state.trainer.model
        bX = st.session_state.trained_X
        by = st.session_state.trained_y

        # ─── Droites de la première couche (vue géométrique) ───
        fig_lines = plot_first_layer_lines(bm, bX, by)
        if fig_lines is not None:
            st.markdown(
                "**Vue géométrique : droites apprises par la 1ère couche**"
            )
            st.caption(
                "Chaque droite correspond à un neurone : il s'active fortement "
                "d'un côté et est inactif (ReLU = 0) de l'autre. Les angles et "
                "positions des droites montrent où chaque neurone a appris à "
                "découper le plan."
            )
            st.pyplot(fig_lines)
            plt.close(fig_lines)

        # ─── Heatmaps des matrices de poids de toutes les couches ───
        st.markdown("**Heatmaps des matrices de poids (toutes les couches)**")
        with st.spinner("Calcul des heatmaps de poids..."):
            weight_figures = plot_weight_heatmaps(bm)
        for layer_name, fig in weight_figures:
            st.pyplot(fig)
            plt.close(fig)


# ─────────────────────────────────────────────
# Flux des gradients (vanishing / exploding)
# ─────────────────────────────────────────────
if (
    st.session_state.trainer is not None
    and st.session_state.trainer.grad_history
):
    st.markdown("---")
    st.subheader("Flux des gradients par couche")
    st.caption(
        "On affiche ici la norme L2 du gradient de "
        "chaque couche : son évolution au fil des époques (à gauche) et son "
        "profil par profondeur (à droite). Pour bien voir le phénomène, "
        "comparer **ReLU** et **sigmoid/tanh** sur un réseau profond (≥3 couches)."
    )

    grad_history = st.session_state.trainer.grad_history

    # Diagnostic automatique
    level, message = diagnose_gradients(grad_history)
    {"success": st.success, "warning": st.warning, "error": st.error}[level](message)

    # Graphiques
    fig_grad = plot_gradient_flow(grad_history)
    st.pyplot(fig_grad)
    plt.close(fig_grad)


# ─────────────────────────────────────────────
# Exploration de l'historique d'entraînement
# ─────────────────────────────────────────────
if st.session_state.trainer is not None:
    trainer = st.session_state.trainer
    n_snapshots = len(trainer.snapshots)

    st.markdown("---")
    st.subheader("Exploration de l'historique d'entraînement")
    st.caption(
        "Déplace le curseur pour visualiser l'état du réseau à n'importe quelle époque "
        "passée. Les snapshots sont sauvegardés à chaque époque pendant l'entraînement."
    )

    selected_epoch = st.slider(
        "Époque à visualiser",
        min_value=1,
        max_value=n_snapshots,
        value=n_snapshots,
        step=1,
    )

    # Restauration du snapshot demandé
    trainer.load_snapshot(selected_epoch - 1)
    trainer.model.eval()

    col_a, col_b = st.columns(2)
    with col_a:
        fig_hist = plot_decision_boundary(
            trainer.model,
            st.session_state.trained_X,
            st.session_state.trained_y,
            mode=st.session_state.trained_mode,
        )
        # Mise à jour du titre pour refléter l'époque sélectionnée
        fig_hist.axes[0].set_title(f"Frontière à l'époque {selected_epoch}")
        st.pyplot(fig_hist)
        plt.close(fig_hist)

    with col_b:
        st.metric(
            label="Train loss à cette époque",
            value=f"{trainer.history['train_loss'][selected_epoch - 1]:.4f}",
        )
        if trainer.history["val_loss"]:
            st.metric(
                label="Val loss à cette époque",
                value=f"{trainer.history['val_loss'][selected_epoch - 1]:.4f}",
            )
        st.metric(
            label="Nombre total d'époques",
            value=n_snapshots,
        )
        if trainer.best_epoch >= 0:
            st.metric(
                label="Meilleure époque (val_loss)",
                value=f"{trainer.best_epoch + 1}",
                delta=f"val_loss = {trainer.best_val_loss:.4f}",
            )
        if st.button("Réinitialiser l'historique"):
            st.session_state.trainer = None
            st.session_state.trained_X = None
            st.session_state.trained_y = None
            st.session_state.trained_mode = None
            st.session_state.test_loader = None
            st.session_state.test_report = None
            st.rerun()


# ─────────────────────────────────────────────
# Évaluation sur le test set
# ─────────────────────────────────────────────
if st.session_state.test_report is not None:
    st.markdown("---")
    st.subheader("Évaluation sur le test set")
    st.caption(
        "Métriques calculées sur le **test set** (données jamais vues "
        "pendant l'entraînement). Le modèle évalué est celui restauré "
        "à la meilleure époque (best val_loss)."
    )

    report = st.session_state.test_report

    # ─── Métriques principales ───
    cols = st.columns(4)
    with cols[0]:
        st.metric("Accuracy",  f"{report.accuracy  * 100:.2f} %")
    with cols[1]:
        st.metric("Precision", f"{report.precision * 100:.2f} %")
    with cols[2]:
        st.metric("Recall",    f"{report.recall    * 100:.2f} %")
    with cols[3]:
        st.metric("F1-score",  f"{report.f1        * 100:.2f} %")

    st.caption(f"Évalué sur **{report.n_samples} échantillons** du test set.")

    # ─── Matrice de confusion ───
    cm_col, info_col = st.columns([2, 1])

    with cm_col:
        st.markdown("**Matrice de confusion**")
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        cm = report.confusion
        im = ax_cm.imshow(cm, cmap="Blues", aspect="equal")
        plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

        # Annotations dans chaque case
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                # Texte blanc si la case est foncée (valeur élevée), sinon noir
                color = "white" if val > cm.max() / 2 else "black"
                ax_cm.text(j, i, str(val), ha="center", va="center",
                           color=color, fontsize=14, fontweight="bold")

        ax_cm.set_xticks([0, 1])
        ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(["Prédit 0", "Prédit 1"])
        ax_cm.set_yticklabels(["Vrai 0", "Vrai 1"])
        ax_cm.set_xlabel("Classe prédite")
        ax_cm.set_ylabel("Classe réelle")
        ax_cm.set_title("Matrice de confusion")
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    with info_col:
        st.markdown("**Lecture rapide**")
        tn, fp, fn, tp = report.confusion.ravel()
        st.markdown(
            f"- **Vrais négatifs (TN)** : {tn} → classe 0 bien prédite\n"
            f"- **Faux positifs (FP)** : {fp} → classe 0 prédite à tort en classe 1\n"
            f"- **Faux négatifs (FN)** : {fn} → classe 1 ratée\n"
            f"- **Vrais positifs (TP)** : {tp} → classe 1 bien prédite"
        )
