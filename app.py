import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from models.mlp import MLP
from models.mlp_bottleneck import MLPBottleneck
from models.gcn import GCN
from data.datasets import DATASETS, MULTICLASS_CAPABLE, get_dataset, split_dataset, to_dataloader
from data.graphs import build_knn, build_radius, knn_edges, normalize_adj
from data.sbm import make_sbm

# Dataset spécial : le graphe est FOURNI (communautés), les features 2D sont faibles.
SBM_NAME = "SBM (graphe communautaire)"
from training.trainer import Trainer, EarlyStopping
from training.metrics import evaluate
from utils.hooks import ActivationCapture


# Palette partagée : une classe = toujours la même couleur partout
# (scatter des données, frontière de décision, et plus tard l'espace latent).
# Le sélecteur K va de 2 à 5 classes → 5 couleurs suffisent.
CLASS_COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]


# ─────────────────────────────────────────────
# Configuration de la page
# ─────────────────────────────────────────────
st.set_page_config(page_title="Neural-Viz", layout="wide")
st.title("Visualisation d'un réseau de neurones")
with st.expander("À quoi sert cet outil ?"):
    st.markdown(
        "Cet outil montre **comment un réseau de neurones transforme des données** pour "
        "les classer, le tout en 2D pour qu'on puisse tout voir.\n\n"
        "- **MLP** (perceptron) : choisis un dataset, entraîne, et observe la frontière de "
        "décision, l'espace latent, les activations, les gradients…\n"
        "- **Réseau de graphe (GCN / GraphSAGE / GAT)** : on relie les points en un graphe "
        "et chaque nœud agrège ses voisins. Le fil rouge est **quand et pourquoi** un "
        "réseau de graphe bat (ou non) un MLP : il ne gagne que si le **graphe porte une "
        "information que les features n'ont pas** (dataset **SBM**), et empiler trop de "
        "couches finit par tout **lisser** (sur-lissage).\n\n"
        "Choisis le **type de modèle** et le **dataset** dans la barre latérale, puis "
        "entraîne. Les sections « Approfondir » sont repliées par défaut."
    )


# ─────────────────────────────────────────────
# Sidebar — paramètres
# ─────────────────────────────────────────────
st.sidebar.header("Architecture du réseau")

model_type = st.sidebar.radio(
    "Type de modèle",
    ["MLP (perceptron)", "Réseau de graphe (GCN / GraphSAGE / GAT)"],
    help=(
        "MLP : perceptron classique (chaque point traité indépendamment).\n"
        "Réseau de graphe : construit un graphe k-NN et agrège les voisins. Selon "
        "l'**agrégation** choisie ci-dessous, il devient un GCN (moyenne), un "
        "GraphSAGE (max) ou un GAT (attention)."
    ),
)
is_graph = model_type.startswith("Réseau")
knn_k = st.sidebar.slider(
    "Graphe — voisins k", 2, 15, 6,
    disabled=not is_graph,
    help="Nombre de voisins reliés à chaque point dans le graphe k-NN (taille du voisinage).",
)
gcn_layers_n = st.sidebar.slider(
    "Graphe — couches (profondeur)", 1, 4, 2,
    disabled=not is_graph,
    help=(
        "Nombre de couches de graphe (répétitions de l'agrégation). Avec 1 seule, le "
        "champ réceptif est d'un seul saut ; 2 suffisent en général. Trop de couches "
        "→ sur-lissage (oversmoothing)."
    ),
)
gcn_aggregation = st.sidebar.selectbox(
    "Agrégation des voisins", ["moyenne (GCN)", "max (GraphSAGE)", "attention (GAT)"],
    disabled=not is_graph,
    help=(
        "Comment un nœud combine ses voisins — c'est ce qui distingue les "
        "architectures. **Moyenne** (Kipf/GCN) : poids fixes, tous les voisins "
        "égaux. **Max** (GraphSAGE) : on garde la plus forte réponse (détecteur de "
        "motif). **Attention** (GAT) : le nœud APPREND combien chaque voisin compte "
        "(multi-têtes) — il « se concentre sur l'info pertinente »."
    ),
)
gcn_agg = {"moyenne": "mean", "max": "max", "attention": "attention"}[gcn_aggregation.split()[0]]
# Nom de l'architecture classique correspondante (pour les affichages)
agg_label = {"mean": "GCN", "max": "GraphSAGE", "attention": "GAT"}[gcn_agg]
gat_heads = st.sidebar.slider(
    "Attention — têtes", 1, 8, 4,
    disabled=not (is_graph and gcn_agg == "attention"),
    help=(
        "Nombre de mécanismes d'attention indépendants calculés en parallèle puis "
        "moyennés (plusieurs façons de pondérer les voisins → plus stable). Actif "
        "seulement avec l'agrégation « attention »."
    ),
)

n_hidden_layers = st.sidebar.slider(
    "Nombre de couches cachées", 1, 8, 1,
    help=(
        "Jusqu'à 8 couches. Augmenter la profondeur permet d'observer "
        "l'aggravation exponentielle du vanishing gradient (visible via "
        "« Suivre les gradients par couche »)."
    ),
)
neurons_per_layer = st.sidebar.slider("Neurones par couche cachée", 2, 64, 8)
control_first_layer = st.sidebar.checkbox(
    "Contrôler la 1ère couche séparément",
    value=False,
    help=(
        "Chaque neurone de la 1ère couche calcule une droite de séparation "
        "dans le plan d'entrée. Plus de neurones en 1ère couche = plus de "
        "« plans » pour découper des frontières complexes (ex : damier). "
        "Visible via « Afficher les poids appris » (droites de la 1ère couche)."
    ),
)
first_layer_neurons = st.sidebar.slider(
    "Neurones de la 1ère couche", 2, 128, 16,
    disabled=not control_first_layer,
    help="Nombre de droites de séparation apprises directement sur l'entrée.",
)
activation = st.sidebar.selectbox("Fonction d'activation", ["relu", "tanh", "sigmoid"])
use_batchnorm = st.sidebar.checkbox(
    "Utiliser BatchNorm", value=False,
    help=(
        "Normalise les activations couche par couche (entraînement plus stable, "
        "échelles homogènes). S'applique au MLP comme au GCN : en mode GCN, une "
        "BatchNorm est insérée après chaque couche de convolution de graphe."
    ),
)
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
    [1, 2, 3],
    index=1,
    horizontal=True,
    help=(
        "1D : le réseau doit tout résumer en UN seul nombre — test extrême "
        "(le problème est-il assez simple pour tenir sur une ligne ?).\n"
        "2D : scatter plot statique (matplotlib).\n"
        "3D : scatter plot interactif rotatif (Plotly)."
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
    ["Logits (sortie brute)", "Probabilités (sigmoid / softmax)"],
    index=0,
    help=(
        "Logits : sortie brute du réseau.\n"
        "Probabilités : sigmoid en binaire ; en multi-classes, softmax → "
        "l'opacité de la frontière indique la confiance du réseau (pâle = "
        "le réseau hésite, saturé = il est sûr)."
    ),
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
batch_size = st.sidebar.slider(
    "Batch size", 8, 128, 32,
    help=(
        "Taille des lots. Petits batch = gradient plus bruité → minima plus "
        "« plats » → souvent meilleure généralisation. Gros batch = plus rapide "
        "mais peut moins bien généraliser."
    ),
)
weight_decay = st.sidebar.select_slider(
    "Weight decay (régularisation L2)",
    options=[0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    value=0.0,
    help=(
        "Pénalise les poids trop grands → régularisation. Alternative ou "
        "complément au dropout pour réduire l'overfitting. 0 = désactivé."
    ),
)

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
    list(DATASETS.keys()) + [SBM_NAME],
    index=0,
    help=(
        "⭐ Two Gaussians — trivial (linéairement séparable)\n"
        "🔀 Chevauchants (Overlap / Blobs / Classification / Gaussian Quantiles) "
        "— classes qui se recouvrent, pas de frontière parfaite : idéal pour "
        "observer l'overfitting / la dentelle et la généralisation\n"
        "🌫️ Density — bleu très dense / rouge très clairsemé qui se chevauchent : "
        "dans la zone commune, l'indice est la DENSITÉ locale des points, pas la "
        "position (à tester en mode « Réseau de graphe » : le k-NN capte-t-il la "
        "densité ?)\n"
        "🏁 Checkerboard — damier : demande beaucoup de frontières (de plans)\n"
        "⭐⭐ Moons / Circles / XOR — non linéairement séparables, simples\n"
        "⭐⭐⭐ Sinusoidal / Islands — frontières complexes\n"
        "⭐⭐⭐⭐ Spirals — le boss final\n"
        "🕸️ SBM — graphe communautaire FOURNI + features 2D faibles : le cas où le "
        "réseau de graphe gagne et le MLP échoue (à utiliser en mode « Réseau de graphe »)."
    ),
)
is_sbm = dataset_name == SBM_NAME
n_classes = st.sidebar.slider(
    "Nombre de classes (K)", 2, 5, 2,
    disabled=not (dataset_name in MULTICLASS_CAPABLE or is_sbm),
    help=(
        "Multi-classes disponible pour Blobs, Gaussian Quantiles et SBM (= nombre de "
        "communautés). Les autres datasets restent binaires. K ≥ 3 active le mode "
        "multi-classes (softmax + CrossEntropy + frontière multi-couleurs)."
    ),
)
n_samples = st.sidebar.slider(
    "Nombre de points", 50, 1000, 200,
    help=(
        "Densité du dataset. Peu de points → le réseau mémorise facilement "
        "(overfitting, dentelle) ; beaucoup de points → il doit apprendre la "
        "vraie règle (meilleure généralisation)."
    ),
)
noise = st.sidebar.slider("Niveau de bruit", 0.0, 1.0, 0.2, step=0.05)
sbm_homophily = st.sidebar.slider(
    "SBM — homophilie du graphe", 0.5, 1.0, 0.9, step=0.05,
    disabled=not is_sbm,
    help=(
        "Force du graphe communautaire : fraction des arêtes qui restent DANS une "
        "communauté. **0.5** = graphe aléatoire (n'aide pas → réseau de graphe ≈ MLP) ; "
        "**→ 1.0** = communautés pures (le graphe porte l'info → le réseau de graphe "
        "gagne). Le degré moyen reste constant."
    ),
)
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
# Le SBM fournit AUSSI un graphe (A_sbm) ; les autres datasets n'ont pas de graphe
# propre (il sera construit par k-NN en mode réseau de graphe).
if is_sbm:
    X, y, A_sbm = make_sbm(n_samples=n_samples, n_classes=n_classes,
                           noise=noise, seed=int(seed), homophily=sbm_homophily)
else:
    X, y = get_dataset(dataset_name, n_samples=n_samples, noise=noise,
                       seed=int(seed), n_classes=n_classes)
    A_sbm = None

# Nombre de classes RÉEL du dataset (robuste : certains datasets restent binaires).
# K ≥ 3 → mode multi-classes (softmax + CrossEntropy) ; K = 2 → binaire (BCE).
n_classes_eff = int(len(np.unique(y)))
is_multiclass = n_classes_eff >= 3

(X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(
    X, y,
    train_ratio=train_ratio,
    val_ratio=val_ratio,
    seed=int(seed),
)
train_loader = to_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True, multiclass=is_multiclass)
val_loader   = to_dataloader(X_val,   y_val,   batch_size=batch_size, shuffle=False, multiclass=is_multiclass)
test_loader  = to_dataloader(X_test,  y_test,  batch_size=batch_size, shuffle=False, multiclass=is_multiclass)


# ─────────────────────────────────────────────
# Construction du modèle
# ─────────────────────────────────────────────
def build_model():
    """
    Construit un modèle NEUF à partir de la configuration actuelle de la
    sidebar. Factorisé pour être réutilisable : flux principal d'entraînement
    ET évaluation k-fold (qui reconstruit un modèle frais à chaque fold).
    """
    # Binaire → 1 sortie (logit) ; multi-classes → K sorties (logits bruts,
    # softmax appliqué par CrossEntropyLoss).
    out_dim = n_classes_eff if is_multiclass else 1

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

        # ─── Contrôle indépendant de la 1ère couche (nombre de "plans") ───
        if control_first_layer:
            encoder_layers[0] = first_layer_neurons

        # ─── Décodeur (la "tête") : indépendamment configuré ───
        decoder_layers = [head_neurons] * head_layers

        return MLPBottleneck(
            input_dim=2,
            encoder_layers=encoder_layers,
            bottleneck_dim=bottleneck_dim,
            decoder_layers=decoder_layers,
            output_dim=out_dim,
            activation=activation,
            use_batchnorm=use_batchnorm,
            dropout_rate=dropout_rate,
        )
    else:
        hidden_layers = [neurons_per_layer] * n_hidden_layers
        # ─── Contrôle indépendant de la 1ère couche (nombre de "plans") ───
        if control_first_layer:
            hidden_layers[0] = first_layer_neurons
        return MLP(
            input_dim=2,
            hidden_layers=hidden_layers,
            output_dim=out_dim,
            activation=activation,
            use_batchnorm=use_batchnorm,
            dropout_rate=dropout_rate,
        )


def make_optimizer(m):
    """Crée l'optimiseur choisi dans la sidebar pour le modèle donné."""
    if optimizer_name == "SGD":
        return torch.optim.SGD(m.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return torch.optim.Adam(m.parameters(), lr=learning_rate, weight_decay=weight_decay)


# On fixe le générateur aléatoire juste avant de créer le modèle :
# l'initialisation des poids devient reproductible.
torch.manual_seed(int(weight_seed))
model = build_model()   # MLP (toujours construit ; utilisé uniquement en mode MLP)

st.sidebar.markdown("---")
if is_graph:
    heads_txt = f", {gat_heads} têtes" if gcn_agg == "attention" else ""
    st.sidebar.markdown(
        f"**Modèle :** Réseau de graphe ({agg_label}) — {gcn_layers_n} couche(s) × "
        f"{neurons_per_layer}, agrégation {gcn_agg}{heads_txt}, k = {knn_k}"
    )
else:
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
        out = model(torch.tensor(grid)).numpy()   # (N, C) : C=1 binaire, C=K multi

    n_out = out.shape[1]
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = CLASS_COLORS

    if n_out == 1:
        # ───── Cas binaire (comportement historique) ─────
        logits = out.reshape(xx.shape)
        if mode == "probas":
            Z = 1 / (1 + np.exp(-logits))   # sigmoid
            boundary_level, cbar_label = 0.5, "P(classe 1)"
            vmin, vmax = 0.0, 1.0
        else:
            Z = logits
            boundary_level, cbar_label = 0.0, "Logits (sortie brute)"
            amax = max(abs(Z.min()), abs(Z.max()))
            vmin, vmax = -amax, amax
        contour = ax.contourf(xx, yy, Z, levels=50, cmap="RdYlBu_r", alpha=0.8, vmin=vmin, vmax=vmax)
        plt.colorbar(contour, ax=ax, label=cbar_label)
        ax.contour(xx, yy, Z, levels=[boundary_level], colors="black", linewidths=1.5)
        classes = [0, 1]
        title = f"Frontière de décision ({'logits' if mode == 'logits' else 'probabilités'})"
    else:
        # ───── Cas multi-classes ─────
        from matplotlib.colors import ListedColormap, to_rgb
        K = n_out
        pred_flat = out.argmax(axis=1)
        pred = pred_flat.reshape(xx.shape)

        if mode == "probas":
            # Confiance = max(softmax). Couleur = classe argmax, OPACITÉ = certitude.
            # Softmax stable numériquement (on retranche le max par ligne).
            shifted = out - out.max(axis=1, keepdims=True)
            soft = np.exp(shifted)
            soft /= soft.sum(axis=1, keepdims=True)
            conf = soft.max(axis=1)                       # ∈ [1/K, 1]
            # Normalise : 1/K (équiprobable, le réseau hésite) → 0 ; 1 (certain) → 1
            alpha = np.clip((conf - 1.0 / K) / (1.0 - 1.0 / K), 0.0, 1.0)
            rgba = np.zeros((pred_flat.size, 4))
            for k in range(K):
                rgba[pred_flat == k, :3] = to_rgb(colors[k % len(colors)])
            rgba[:, 3] = alpha * 0.85
            img = rgba.reshape(xx.shape[0], xx.shape[1], 4)
            # origin="lower" : yy croît de y_min (1ère ligne) vers y_max → cohérent
            ax.imshow(img, extent=(x_min, x_max, y_min, y_max),
                      origin="lower", aspect="auto", interpolation="nearest")
            title = f"Frontière — {K} classes (softmax : opacité = confiance)"
        else:
            # Régions argmax « pleines » (toutes de même intensité)
            cmap = ListedColormap(colors[:K])
            ax.contourf(xx, yy, pred, levels=np.arange(-0.5, K, 1.0), cmap=cmap, alpha=0.35)
            title = f"Frontière de décision — {K} classes (argmax)"

        # Frontières entre régions (aux demi-entiers) — dans les deux modes
        ax.contour(xx, yy, pred, levels=[i + 0.5 for i in range(K - 1)],
                   colors="black", linewidths=1.0)
        classes = list(range(K))

    # Points (colorés par classe réelle)
    for cls in classes:
        mask = y == cls
        ax.scatter(
            X[mask, 0], X[mask, 1],
            c=colors[cls % len(colors)],
            edgecolors="white", linewidths=0.5, s=40,
            label=f"Classe {cls}", zorder=3,
        )

    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_title(title)
    ax.legend()
    return fig


def plot_latent_space_1d(model: MLPBottleneck, X: np.ndarray, y: np.ndarray):
    """
    Affiche la représentation 1D de X dans l'espace latent (bottleneck à 1
    neurone) appris par le réseau.

    Chaque point d'entrée est résumé en UN seul nombre z₁. On affiche :
      - en haut : un strip plot (chaque point placé sur l'axe z₁, coloré par classe) ;
      - en bas : les histogrammes de z₁ par classe, posés sur les RÉGIONS DE
        DÉCISION de la tête (fond pâle = classe prédite le long de z₁), avec un
        score de SÉPARABILITÉ (quel % de points la tête classe bien à partir de z₁).

    Lecture : si les classes occupent des plages de z₁ **séparées** et tombent
    chacune sur le bon fond, le réseau a réussi à ranger le problème sur une
    seule dimension (séparabilité ≈ 100 %). Si elles se **chevauchent**, une
    seule dimension ne suffit pas — il faut un bottleneck plus large.
    """
    model.eval()
    with torch.no_grad():
        z = model.encode(torch.tensor(X, dtype=torch.float32)).numpy().ravel()

    colors = CLASS_COLORS
    classes = np.unique(y).astype(int)
    rng = np.random.default_rng(0)

    # ─── Décision de la TÊTE le long de l'axe z₁ ───
    # Le bottleneck est 1D : on balaie z₁ et on demande à la tête (décodeur) ce
    # qu'elle prédit. Cela donne (a) les « régions de décision » sur la ligne et
    # (b) la séparabilité RÉELLEMENT atteinte par le réseau sur cette seule
    # dimension = quel % de points la tête classe correctement à partir de z₁.
    pad = 0.05 * (z.max() - z.min() + 1e-9)
    grid = np.linspace(z.min() - pad, z.max() + pad, 400).astype(np.float32)
    with torch.no_grad():
        out_grid = model.decoder(torch.tensor(grid).view(-1, 1)).numpy()
        out_pts = model.decoder(torch.tensor(z, dtype=torch.float32).view(-1, 1)).numpy()
    if out_grid.shape[1] > 1:                       # multi-classes : argmax
        region = out_grid.argmax(1)
        pts_pred = out_pts.argmax(1)
    else:                                           # binaire : seuil à 0
        region = (out_grid[:, 0] > 0).astype(int)
        pts_pred = (out_pts[:, 0] > 0).astype(int)
    sep = (pts_pred == y).mean()

    fig, (ax_strip, ax_hist) = plt.subplots(
        2, 1, figsize=(8, 5.5), sharex=True,
        gridspec_kw={"height_ratios": [1, 2]},
    )

    # ─── Strip plot : points sur l'axe z₁ + jitter vertical ───
    for cls in classes:
        m = y == cls
        jitter = rng.uniform(-0.4, 0.4, size=int(m.sum()))
        ax_strip.scatter(z[m], cls + jitter, c=colors[cls % len(colors)], s=25,
                         edgecolors="white", linewidths=0.4, alpha=0.8)
    ax_strip.set_yticks(classes)
    ax_strip.set_yticklabels([f"Classe {c}" for c in classes])
    ax_strip.set_title(f"Espace latent 1D — séparabilité sur z₁ : {sep * 100:.0f} %")

    # ─── Régions de décision de la tête (fond coloré pâle) + histogrammes ───
    bounds = np.where(np.diff(region) != 0)[0]
    edges = np.concatenate(([grid[0]], (grid[bounds] + grid[bounds + 1]) / 2, [grid[-1]]))
    seg_cls = np.concatenate(([region[0]], region[bounds + 1]))
    for k in range(len(seg_cls)):
        ax_hist.axvspan(edges[k], edges[k + 1],
                        color=colors[int(seg_cls[k]) % len(colors)], alpha=0.12, zorder=0)
    for b in bounds:                                # frontières de décision sur z₁
        ax_hist.axvline((grid[b] + grid[b + 1]) / 2, color="black", ls="--", lw=1, alpha=0.6)

    bins = np.linspace(grid[0], grid[-1], 40)
    for cls in classes:
        m = y == cls
        ax_hist.hist(z[m], bins=bins, color=colors[cls % len(colors)], alpha=0.6,
                     label=f"Classe {cls}", zorder=2)
    ax_hist.set_xlabel("z₁ (unique neurone du bottleneck)")
    ax_hist.set_ylabel("Nombre de points")
    ax_hist.legend(title="vraie classe", fontsize=8)
    ax_hist.text(0.01, 0.97, "fond pâle = classe prédite par la tête le long de z₁",
                 transform=ax_hist.transAxes, fontsize=7, va="top", color="dimgray")
    fig.tight_layout()
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
    colors = CLASS_COLORS
    for cls in np.unique(y).astype(int):
        mask = y == cls
        ax.scatter(
            Z[mask, 0], Z[mask, 1],
            c=colors[cls % len(colors)],
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

    colors_pts = CLASS_COLORS
    fig = go.Figure()
    for cls in np.unique(y).astype(int):
        mask = y == cls
        fig.add_trace(go.Scatter3d(
            x=Z[mask, 0],
            y=Z[mask, 1],
            z=Z[mask, 2],
            mode="markers",
            name=f"Classe {cls}",
            marker=dict(
                size=4,
                color=colors_pts[cls % len(colors_pts)],
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
    colors_pts = CLASS_COLORS

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
            for cls in np.unique(y).astype(int):
                m = y == cls
                ax.scatter(X[m, 0], X[m, 1], c=colors_pts[cls % len(colors_pts)],
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
    colors_pts = CLASS_COLORS
    for cls in np.unique(y).astype(int):
        m = y == cls
        ax.scatter(X[m, 0], X[m, 1], c=colors_pts[cls % len(colors_pts)],
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


def plot_generalization(model, X_train, y_train, X_test, y_test, mode="logits"):
    """
    Frontière de décision avec les points de TEST superposés (jamais vus
    pendant l'entraînement). Les points de test MAL classés sont entourés
    de noir.

    Permet de VOIR la généralisation : si la frontière (apprise sur le train)
    place correctement des points nouveaux, le réseau généralise ; si beaucoup
    de points de test tombent du mauvais côté (surtout là où la frontière fait
    de la dentelle), c'est de l'overfitting.
    """
    # Bornes calculées sur train + test pour tout afficher
    X_all = np.vstack([X_train, X_test])
    x_min, x_max = X_all[:, 0].min() - 0.5, X_all[:, 0].max() + 0.5
    y_min, y_max = X_all[:, 1].min() - 0.5, X_all[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

    model.eval()
    with torch.no_grad():
        grid_out = model(torch.tensor(grid)).numpy()                       # (N, C)
        test_out = model(torch.tensor(X_test.astype(np.float32))).numpy()  # (M, C)

    y_test_int = y_test.astype(int)
    n_out = grid_out.shape[1]
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = CLASS_COLORS

    if n_out == 1:
        # ───── Binaire (historique) : carte de la sortie + frontière à 0 ─────
        logits = grid_out.reshape(xx.shape)
        test_pred = (test_out.squeeze(-1) > 0).astype(int)
        if mode == "probas":
            Z = 1 / (1 + np.exp(-logits))
            boundary_level, cbar_label = 0.5, "P(classe 1)"
            vmin, vmax = 0.0, 1.0
        else:
            Z = logits
            boundary_level, cbar_label = 0.0, "Logits (sortie brute)"
            amax = max(abs(Z.min()), abs(Z.max()))
            vmin, vmax = -amax, amax
        contour = ax.contourf(xx, yy, Z, levels=50, cmap="RdYlBu_r", alpha=0.8, vmin=vmin, vmax=vmax)
        plt.colorbar(contour, ax=ax, label=cbar_label)
        ax.contour(xx, yy, Z, levels=[boundary_level], colors="black", linewidths=1.5)
        classes = [0, 1]
    else:
        # ───── Multi-classes : régions argmax + points de test en K couleurs ─────
        from matplotlib.colors import ListedColormap
        K = n_out
        pred = grid_out.argmax(axis=1).reshape(xx.shape)
        test_pred = test_out.argmax(axis=1)
        ax.contourf(xx, yy, pred, levels=np.arange(-0.5, K, 1.0),
                    cmap=ListedColormap(colors[:K]), alpha=0.35)
        ax.contour(xx, yy, pred, levels=[i + 0.5 for i in range(K - 1)],
                   colors="black", linewidths=1.0)
        classes = list(range(K))

    misclassified = test_pred != y_test_int

    # Points de test colorés par vraie classe
    for cls in classes:
        m = y_test_int == cls
        ax.scatter(X_test[m, 0], X_test[m, 1], c=colors[cls % len(colors)],
                   edgecolors="white", linewidths=0.5, s=45,
                   label=f"Test classe {cls}", zorder=3)

    # Points de test MAL classés : entourés de noir
    if misclassified.any():
        ax.scatter(X_test[misclassified, 0], X_test[misclassified, 1],
                   s=160, facecolors="none", edgecolors="black",
                   linewidths=1.8, zorder=4,
                   label=f"Mal classé ({int(misclassified.sum())})")

    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    n_err = int(misclassified.sum())
    n_tot = len(y_test_int)
    ax.set_title(
        f"Généralisation : points de TEST sur la frontière\n"
        f"{n_tot - n_err}/{n_tot} bien classés "
        f"({100 * (n_tot - n_err) / n_tot:.1f}% test accuracy)"
    )
    ax.legend(fontsize=8, loc="best")
    return fig


def run_kfold(X, y, k, n_epochs, batch_size, progress=None):
    """
    Validation croisée k-fold stratifiée (équilibre des classes par fold).

    Découpe (X, y) en k folds. Pour chaque fold : reconstruit un modèle NEUF
    (config actuelle de la sidebar), l'entraîne sur les k-1 autres folds, puis
    l'évalue sur le fold restant. Renvoie le tableau des accuracies de test.

    Donne une estimation plus robuste de la généralisation qu'un seul split
    (accuracy moyenne ± écart-type) — utile surtout sur les petits datasets,
    où un découpage unique peut être chanceux ou malchanceux.
    """
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=int(weight_seed))
    accs = []
    for i, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]

        loader_tr = to_dataloader(X_tr, y_tr, batch_size=batch_size, shuffle=True,
                                  multiclass=is_multiclass)
        torch.manual_seed(int(weight_seed))          # init reproductible
        m = build_model()
        criterion_k = nn.CrossEntropyLoss() if is_multiclass else nn.BCEWithLogitsLoss()
        trainer_k = Trainer(m, make_optimizer(m), criterion_k)
        torch.manual_seed(int(weight_seed))          # shuffle + dropout
        trainer_k.train(
            loader_tr, n_epochs=n_epochs,
            save_snapshots=False, restore_best=False,
        )
        rep = evaluate(m, to_dataloader(X_te, y_te, batch_size=batch_size, shuffle=False,
                                        multiclass=is_multiclass))
        accs.append(rep.accuracy)
        if progress is not None:
            progress.progress((i + 1) / k, text=f"Fold {i + 1}/{k} — accuracy {rep.accuracy * 100:.1f}%")
    return np.array(accs)


def plot_kfold(accs):
    """
    Bar chart de l'accuracy de chaque fold, avec la ligne de moyenne et la
    bande ± écart-type. Rend la dispersion visible : des barres regroupées =
    performance stable ; des barres dispersées = sensible au découpage.
    """
    k = len(accs)
    mean, std = accs.mean(), accs.std()

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = plt.cm.viridis(np.linspace(0.25, 0.8, k))
    ax.bar(range(1, k + 1), accs * 100, color=colors, edgecolor="white", zorder=3)

    # Moyenne + bande ± écart-type
    ax.axhspan((mean - std) * 100, (mean + std) * 100, color="gray", alpha=0.15,
               zorder=1, label=f"± écart-type ({std * 100:.1f} pts)")
    ax.axhline(mean * 100, color="black", linestyle="--", linewidth=1.5,
               zorder=2, label=f"moyenne {mean * 100:.1f} %")

    ax.set_xticks(range(1, k + 1))
    ax.set_xticklabels([f"Fold {i}" for i in range(1, k + 1)])
    ax.set_ylabel("Accuracy test (%)")
    ax.set_ylim(max(0, accs.min() * 100 - 10), min(100, accs.max() * 100 + 10))
    ax.set_title(f"Accuracy par fold (validation croisée k={k})")
    ax.legend(loc="best", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def run_repeated_holdout(X, y, n_runs, n_epochs, batch_size, test_ratio, progress=None):
    """
    Repeated holdout (Monte-Carlo cross-validation) : re-tire N fois un découpage
    train/test **stratifié**, ré-entraîne un modèle neuf et mesure l'accuracy à
    chaque fois. L'init des poids est FIXE → la dispersion observée vient
    uniquement du DÉCOUPAGE (quels points tombent en train vs en test).

    Renvoie (accs_test, accs_train) : deux tableaux de N accuracies.
    """
    test_size = min(max(test_ratio, 0.1), 0.5)       # garde-fou sur la part de test
    sss = StratifiedShuffleSplit(n_splits=n_runs, test_size=test_size, random_state=0)
    accs_te, accs_tr = [], []
    for i, (tr_idx, te_idx) in enumerate(sss.split(X, y)):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]

        loader_tr = to_dataloader(X_tr, y_tr, batch_size=batch_size, shuffle=True,
                                  multiclass=is_multiclass)
        torch.manual_seed(int(weight_seed))          # init FIXE → isole la variance du découpage
        m = build_model()
        criterion = nn.CrossEntropyLoss() if is_multiclass else nn.BCEWithLogitsLoss()
        trainer_h = Trainer(m, make_optimizer(m), criterion)
        torch.manual_seed(int(weight_seed))
        trainer_h.train(loader_tr, n_epochs=n_epochs, save_snapshots=False, restore_best=False)

        accs_te.append(evaluate(m, to_dataloader(X_te, y_te, batch_size=batch_size,
                                                 shuffle=False, multiclass=is_multiclass)).accuracy)
        accs_tr.append(evaluate(m, to_dataloader(X_tr, y_tr, batch_size=batch_size,
                                                 shuffle=False, multiclass=is_multiclass)).accuracy)
        if progress is not None:
            progress.progress((i + 1) / n_runs, text=f"Découpage {i + 1}/{n_runs}")
    return np.array(accs_te), np.array(accs_tr)


def plot_holdout_distribution(accs):
    """Histogramme de l'accuracy test sur N découpages + moyenne et bande ± écart-type."""
    mean, std = accs.mean(), accs.std()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(accs * 100, bins=min(15, max(5, len(accs) // 2)),
            color="#2196F3", alpha=0.75, edgecolor="white", zorder=3)
    ax.axvspan((mean - std) * 100, (mean + std) * 100, color="gray", alpha=0.15,
               zorder=1, label=f"± écart-type ({std * 100:.1f} pts)")
    ax.axvline(mean * 100, color="black", linestyle="--", linewidth=1.5,
               zorder=2, label=f"moyenne {mean * 100:.1f} %")
    ax.set_xlabel("Accuracy test (%)")
    ax.set_ylabel("Nombre de découpages")
    ax.set_title(f"Distribution de l'accuracy test sur {len(accs)} découpages")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
# Métriques de classification — helpers partagés (MLP et GCN)
# ─────────────────────────────────────────────
def plot_confusion_matrix(cm, title="Matrice de confusion"):
    """Matrice de confusion K×K (lignes = vraie classe, colonnes = prédiction)."""
    K = cm.shape[0]
    fig, ax = plt.subplots(figsize=(0.7 * K + 3.5, 0.7 * K + 3))
    im = ax.imshow(cm, cmap="Blues", aspect="equal")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Annotations dans chaque case (texte blanc si case foncée, sinon noir)
    fs = 14 if K <= 3 else 11
    for i in range(K):
        for j in range(K):
            val = cm[i, j]
            ax.text(j, i, str(val), ha="center", va="center",
                    color="white" if val > cm.max() / 2 else "black",
                    fontsize=fs, fontweight="bold")

    rot = 45 if K > 3 else 0
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels([f"Prédit {k}" for k in range(K)],
                       rotation=rot, ha="right" if rot else "center")
    ax.set_yticklabels([f"Vrai {k}" for k in range(K)])
    ax.set_xlabel("Classe prédite"); ax.set_ylabel("Classe réelle")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def per_class_table(cm):
    """
    Précision / rappel / F1 / support PAR CLASSE, dérivés de la matrice K×K.
    Rappel = diag/somme(ligne), précision = diag/somme(colonne). La moyenne de
    chaque colonne redonne la valeur « macro ».
    """
    import pandas as pd
    K = cm.shape[0]
    support = cm.sum(axis=1)              # nb de points réellement dans la classe (ligne)
    col_sum = cm.sum(axis=0)              # nb de points prédits dans la classe (colonne)
    diag = np.diag(cm).astype(float)
    recall_k    = np.divide(diag, support, out=np.zeros(K), where=support > 0)
    precision_k = np.divide(diag, col_sum, out=np.zeros(K), where=col_sum > 0)
    denom = precision_k + recall_k
    f1_k = np.divide(2 * precision_k * recall_k, denom,
                     out=np.zeros(K), where=denom > 0)
    return pd.DataFrame(
        {
            "Précision": [f"{p * 100:.1f} %" for p in precision_k],
            "Rappel":    [f"{r * 100:.1f} %" for r in recall_k],
            "F1-score":  [f"{f * 100:.1f} %" for f in f1_k],
            "Support":   [int(s) for s in support],
        },
        index=[f"Classe {k}" for k in range(K)],
    )


# ─────────────────────────────────────────────
# Robustesse à la rotation (test de Barthe : « Pb de rotation de données ? »)
# ─────────────────────────────────────────────
def rotate_points(X: np.ndarray, deg: float) -> np.ndarray:
    """Fait tourner un nuage 2D de `deg` degrés autour de l'origine."""
    t = np.deg2rad(deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]], dtype=np.float32)
    return (X @ R.T).astype(np.float32)


def plot_rotation_robustness(model, X_test, y_test):
    """
    Accuracy du modèle entraîné quand on fait tourner le TEST set de 0 à 180°.
    Nos features = coordonnées absolues → le modèle n'est PAS invariant par
    rotation (sauf si le problème l'est déjà, ex. cercles concentriques).
    Renvoie (fig, accuracies).
    """
    model.eval()
    angles = list(range(0, 181, 15))
    accs = []
    for d in angles:
        with torch.no_grad():
            out = model(torch.tensor(rotate_points(X_test, d))).numpy()
        pred = out.argmax(1) if out.shape[1] > 1 else (out[:, 0] > 0).astype(int)
        accs.append(float((pred == y_test).mean()) * 100)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(angles, accs, "-o", color="#2196F3", zorder=3)
    ax.axhline(accs[0], color="gray", ls=":", lw=1, label=f"sans rotation ({accs[0]:.0f} %)")
    ax.set_xlabel("angle de rotation du test (°)")
    ax.set_ylabel("accuracy test (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(angles)
    ax.set_title("Robustesse à la rotation des données")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, accs


# ─────────────────────────────────────────────
# Réseau de graphe — modèle, masques et visualisations (GCN / GraphSAGE / GAT)
# ─────────────────────────────────────────────
def build_gcn_model():
    """Construit le réseau de graphe selon la sidebar (agrégation, têtes, profondeur)."""
    out_dim = n_classes_eff if is_multiclass else 1
    gcn_layers = [neurons_per_layer] * gcn_layers_n
    head_layers = [max(neurons_per_layer // 2, 4)]
    # SBM : on force la BatchNorm — sur des features faibles, un petit réseau peut
    # s'effondrer en multi-classes (prédire une seule classe) ; la BN stabilise.
    bn = use_batchnorm or is_sbm
    return GCN(input_dim=2, gcn_layers=gcn_layers, head_layers=head_layers,
               output_dim=out_dim, activation=activation, aggregation=gcn_agg,
               heads=gat_heads, use_batchnorm=bn)


def gcn_masks(n):
    """Indices train/val/test sur les n nœuds (mêmes ratios que la sidebar)."""
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    n_tr, n_val = int(train_ratio * n), int(val_ratio * n)
    return perm[:n_tr], perm[n_tr:n_tr + n_val], perm[n_tr + n_val:]


def plot_gcn_graph(X, y, edges, title=None):
    """Dessine le graphe de voisinage : arêtes grises + nœuds colorés par vraie classe."""
    from matplotlib.collections import LineCollection
    fig, ax = plt.subplots(figsize=(7, 6))
    segs = [[(X[i, 0], X[i, 1]), (X[j, 0], X[j, 1])] for (i, j) in edges]
    ax.add_collection(LineCollection(segs, colors="lightgray", linewidths=0.4, zorder=1))
    for c in range(n_classes_eff):
        m = y == c
        ax.scatter(X[m, 0], X[m, 1], c=CLASS_COLORS[c % len(CLASS_COLORS)], s=28,
                   edgecolors="white", linewidths=0.4, zorder=2, label=f"Classe {c}")
    ax.set_title(title or f"Graphe k-NN (k={knn_k}) — {len(edges)} arêtes")
    ax.set_xlabel("x₁"); ax.set_ylabel("x₂"); ax.legend(fontsize=8)
    return fig


def plot_gcn_pred(X, y, pred, te_idx):
    """Nœuds colorés par classe PRÉDITE ; points de test mal classés entourés de noir."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for c in range(n_classes_eff):
        m = pred == c
        ax.scatter(X[m, 0], X[m, 1], c=CLASS_COLORS[c % len(CLASS_COLORS)], s=30,
                   edgecolors="white", linewidths=0.4, zorder=2)
    wrong = np.zeros(len(y), dtype=bool)
    wrong[te_idx] = pred[te_idx] != y[te_idx]
    if wrong.any():
        ax.scatter(X[wrong, 0], X[wrong, 1], s=150, facecolors="none",
                   edgecolors="black", linewidths=1.6, zorder=3,
                   label=f"test mal classé ({int(wrong.sum())})")
        ax.legend(fontsize=8)
    ax.set_title("Prédictions (nœuds de test mal classés entourés)")
    ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
    return fig


def plot_gcn_latent(model, X, y, A_hat):
    """
    Espace latent du réseau de graphe : les embeddings de nœuds renvoyés par
    ``encode()`` (après les couches de graphe, AVANT la tête). Affichage côte à côte de
    l'espace d'entrée (coordonnées brutes) et de l'espace latent, ce dernier
    projeté en 2D par PCA si sa dimension dépasse 2. Un score de séparabilité
    (silhouette, de -1 à 1) résume à quel point les classes y sont distinctes.
    """
    from sklearn.metrics import silhouette_score
    model.eval()
    with torch.no_grad():
        Z = model.encode(torch.tensor(X, dtype=torch.float32), A_hat).numpy()

    if Z.shape[1] == 2:
        emb, xl, yl = Z, "z₁", "z₂"
    else:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=0)
        emb = pca.fit_transform(Z)
        v = pca.explained_variance_ratio_ * 100
        xl, yl = f"PC1 ({v[0]:.0f} %)", f"PC2 ({v[1]:.0f} %)"

    # Séparabilité mesurée sur la VRAIE dimension latente (pas la projection)
    sil = silhouette_score(Z, y) if len(np.unique(y)) > 1 else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, P, title, axx, axy in [
        (axes[0], X, "Espace d'entrée (coordonnées brutes)", "x₁", "x₂"),
        (axes[1], emb, f"Espace latent (dim {Z.shape[1]} → 2D) — "
                       f"séparabilité {sil:.2f}", xl, yl),
    ]:
        for c in range(n_classes_eff):
            m = y == c
            ax.scatter(P[m, 0], P[m, 1], c=CLASS_COLORS[c % len(CLASS_COLORS)],
                       s=26, edgecolors="white", linewidths=0.4, label=f"Classe {c}")
        ax.set_title(title); ax.set_xlabel(axx); ax.set_ylabel(axy)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return fig


def gcn_attention(model, X, A_hat):
    """Poids d'attention de la 1ère couche, moyennés sur les têtes → matrice (n, n)."""
    model.eval()
    with torch.no_grad():
        a = model.gcns[0].attention_weights(torch.tensor(X, dtype=torch.float32), A_hat)
    return a.mean(-1).numpy()          # α[i, j] = attention que i donne à j


def plot_attention_graph(X, y, edges, alpha, A_bin, focus=None):
    """
    Graphe avec arêtes colorées/épaissies par l'attention (moyenne des têtes).
    Si `focus` est donné, met en évidence ce nœud et l'attention qu'il DONNE à
    ses voisins (traits noirs, épaisseur ∝ α).
    """
    from matplotlib.collections import LineCollection
    fig, ax = plt.subplots(figsize=(7, 6))

    if edges:
        w = np.array([(alpha[i, j] + alpha[j, i]) / 2 for (i, j) in edges])
        segs = [[(X[i, 0], X[i, 1]), (X[j, 0], X[j, 1])] for (i, j) in edges]
        lc = LineCollection(segs, cmap="viridis", array=w, alpha=0.6, zorder=1,
                            linewidths=0.3 + 3.0 * (w / (w.max() + 1e-9)))
        ax.add_collection(lc)
        plt.colorbar(lc, ax=ax, fraction=0.046, pad=0.04, label="attention α (moyenne)")

    for c in range(n_classes_eff):
        m = y == c
        ax.scatter(X[m, 0], X[m, 1], c=CLASS_COLORS[c % len(CLASS_COLORS)], s=22,
                   edgecolors="white", linewidths=0.4, zorder=2)

    if focus is not None:
        nbrs = np.where(A_bin[focus] > 0)[0]
        if len(nbrs):
            aw = alpha[focus, nbrs]
            fsegs = [[(X[focus, 0], X[focus, 1]), (X[j, 0], X[j, 1])] for j in nbrs]
            ax.add_collection(LineCollection(
                fsegs, colors="black", zorder=3,
                linewidths=0.6 + 5.0 * (aw / (aw.max() + 1e-9))))
        ax.scatter([X[focus, 0]], [X[focus, 1]], s=240, facecolors="none",
                   edgecolors="black", linewidths=2.4, zorder=4)
        ax.set_title(f"Nœud {focus} en surbrillance — épaisseur des traits noirs = "
                     "attention qu'il donne à chaque voisin")
    else:
        ax.set_title("Attention apprise (couleur/épaisseur des arêtes = poids α)")
    ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
    return fig


def plot_node_attention_bars(focus, alpha, A_bin, y):
    """Voisins du nœud `focus` classés par attention reçue (barres colorées par classe)."""
    nbrs = np.where(A_bin[focus] > 0)[0]
    aw = alpha[focus, nbrs]
    order = np.argsort(aw)[::-1][:8]                    # top 8 voisins
    nbrs, aw = nbrs[order], aw[order]
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = [CLASS_COLORS[int(y[j]) % len(CLASS_COLORS)] for j in nbrs]
    ax.barh([f"nœud {j} (cl. {int(y[j])})" for j in nbrs][::-1], aw[::-1],
            color=colors[::-1])
    ax.set_xlabel("poids d'attention α")
    ax.set_title(f"Voisins du nœud {focus} (classe {int(y[focus])}), par attention")
    fig.tight_layout()
    return fig


def _dirichlet_energy(H, edges):
    """
    Énergie de Dirichlet (normalisée) : différence quadratique moyenne entre nœuds
    VOISINS, divisée par la norme moyenne → mesure « à quel point les voisins restent
    différents ». Tend vers 0 quand les embeddings s'effondrent (sur-lissage).
    """
    if not edges:
        return 0.0
    d = np.mean([np.sum((H[i] - H[j]) ** 2) for (i, j) in edges])
    return float(d / (np.mean(np.sum(H ** 2, axis=1)) + 1e-9))


def oversmoothing_study(X, y, A_hat, edges, depths, width=16, ep=120, n_seeds=3):
    """
    Entraîne des GCN (agrégation moyenne) de profondeur croissante, AVEC et SANS
    BatchNorm, MOYENNÉS sur plusieurs seeds (l'effondrement est un phénomène de
    seuil, erratique par seed → on moyenne pour une tendance propre). Renvoie
    l'accuracy moyenne (des deux), l'énergie de Dirichlet, et les embeddings latents
    à profondeur faible/forte du seed le plus effondré (pour visualiser).
    """
    from sklearn.metrics import silhouette_score
    n = len(y)
    tr_idx, _, te_idx = gcn_masks(n)
    m_tr = torch.tensor(np.isin(np.arange(n), tr_idx))
    Xt = torch.tensor(X, dtype=torch.float32)
    multi = is_multiclass
    od = n_classes_eff if multi else 1
    yt = (torch.tensor(y, dtype=torch.long) if multi
          else torch.tensor(y, dtype=torch.float32).view(-1, 1))
    crit = nn.CrossEntropyLoss() if multi else nn.BCEWithLogitsLoss()

    acc = {False: np.zeros(len(depths)), True: np.zeros(len(depths))}
    energy = np.zeros(len(depths))
    shallow_by_seed, deep_by_seed = {}, {}
    for seed in range(n_seeds):
        for di, L in enumerate(depths):
            for bn in (False, True):
                torch.manual_seed(seed)
                gm = GCN(2, [width] * L, [max(width // 2, 4)], output_dim=od,
                         aggregation="mean", use_batchnorm=bn)
                opt = torch.optim.Adam(gm.parameters(), lr=learning_rate)
                torch.manual_seed(seed)
                for _ in range(ep):
                    gm.train(); opt.zero_grad()
                    crit(gm(Xt, A_hat)[m_tr], yt[m_tr]).backward(); opt.step()
                gm.eval()
                with torch.no_grad():
                    out = gm(Xt, A_hat).numpy()
                    H = gm.encode(Xt, A_hat).numpy() if not bn else None
                pred = out.argmax(1) if multi else (out[:, 0] > 0).astype(int)
                acc[bn][di] += (pred[te_idx] == y[te_idx]).mean()
                if not bn:
                    energy[di] += _dirichlet_energy(H, edges)
                    if L == depths[0]:
                        shallow_by_seed[seed] = H
                    if L == depths[-1]:
                        deep_by_seed[seed] = H
    acc[False] /= n_seeds; acc[True] /= n_seeds; energy /= n_seeds

    # seed dont le latent PROFOND est le plus effondré (silhouette minimale)
    def _sil(H):
        return silhouette_score(H, y) if len(np.unique(y)) > 1 else 0.0
    best = min(deep_by_seed, key=lambda s: _sil(deep_by_seed[s]))
    return dict(depths=depths, acc_nobn=acc[False], acc_bn=acc[True], energy=energy,
                emb_shallow=shallow_by_seed[best], emb_deep=deep_by_seed[best])


def plot_oversmoothing(res, y):
    """Accuracy vs profondeur (sans/avec BN) + latent effondré (faible vs forte profondeur)."""
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    depths = res["depths"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ax = axes[0]
    ax.plot(depths, np.array(res["acc_nobn"]) * 100, "-o", color="#F44336", label="sans BatchNorm")
    ax.plot(depths, np.array(res["acc_bn"]) * 100, "-s", color="#2196F3", label="avec BatchNorm")
    chance = 100.0 / max(n_classes_eff, 2)
    ax.axhline(chance, color="gray", ls="--", lw=1, label=f"hasard ({chance:.0f} %)")
    ax.set_xlabel("profondeur (nb de couches de graphe)"); ax.set_ylabel("accuracy test (%)")
    ax.set_ylim(0, 100); ax.set_title("Accuracy vs profondeur"); ax.legend(fontsize=8)

    for ax, key, L in [(axes[1], "emb_shallow", depths[0]), (axes[2], "emb_deep", depths[-1])]:
        H = res[key]
        emb2 = PCA(2, random_state=0).fit_transform(H) if H.shape[1] > 2 else H
        sil = silhouette_score(H, y) if len(np.unique(y)) > 1 else float("nan")
        for c in range(n_classes_eff):
            mm = y == c
            ax.scatter(emb2[mm, 0], emb2[mm, 1], c=CLASS_COLORS[c % len(CLASS_COLORS)],
                       s=18, edgecolors="white", linewidths=0.3)
        tag = "effondré" if L == depths[-1] else "séparé"
        ax.set_title(f"Latent à {L} couche(s) — séparabilité {sil:.2f} ({tag})")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.tight_layout()
    return fig


def homophily_study(n_samples, n_classes, noise, seed, homophily_values, n_seeds=3, ep=150):
    """
    Balaye l'homophilie du graphe SBM : pour chaque valeur, régénère un SBM et
    compare l'accuracy d'un réseau de graphe (qui voit la structure) et d'un MLP
    (features seules). Moyenné sur quelques seeds. Renvoie (gcn_acc%, mlp_acc%).
    """
    multi = n_classes >= 3
    od = n_classes if multi else 1
    crit = nn.CrossEntropyLoss() if multi else nn.BCEWithLogitsLoss()
    gcn_acc = np.zeros(len(homophily_values))
    mlp_acc = np.zeros(len(homophily_values))
    for si in range(n_seeds):
        for hi, h in enumerate(homophily_values):
            X, y, A = make_sbm(n_samples, n_classes, noise, seed + si, homophily=h)
            A_hat = normalize_adj(A)
            n = len(y)
            perm = np.random.default_rng(seed + si).permutation(n)
            tr, te = perm[:int(.6 * n)], perm[int(.8 * n):]
            Xt = torch.tensor(X)
            yt = (torch.tensor(y) if multi else torch.tensor(y, dtype=torch.float32).view(-1, 1))
            m = torch.tensor(np.isin(np.arange(n), tr))
            # réseau de graphe (avec BatchNorm, comme le SBM en prod)
            torch.manual_seed(0)
            gm = GCN(2, [16, 16], [8], output_dim=od, aggregation="mean", use_batchnorm=True)
            opt = torch.optim.Adam(gm.parameters(), lr=learning_rate)
            for _ in range(ep):
                gm.train(); opt.zero_grad(); crit(gm(Xt, A_hat)[m], yt[m]).backward(); opt.step()
            gm.eval()
            with torch.no_grad():
                og = gm(Xt, A_hat).numpy()
            pg = og.argmax(1) if multi else (og[:, 0] > 0).astype(int)
            gcn_acc[hi] += (pg[te] == y[te]).mean()
            # MLP (features seules) — insensible à l'homophilie (features inchangées)
            torch.manual_seed(0)
            ml = MLP(2, [16, 16], od, activation="relu")
            opt = torch.optim.Adam(ml.parameters(), lr=learning_rate)
            for _ in range(ep):
                ml.train(); opt.zero_grad(); crit(ml(Xt[tr]), yt[tr]).backward(); opt.step()
            ml.eval()
            with torch.no_grad():
                om = ml(Xt).numpy()
            pm = om.argmax(1) if multi else (om[:, 0] > 0).astype(int)
            mlp_acc[hi] += (pm[te] == y[te]).mean()
    return gcn_acc / n_seeds * 100, mlp_acc / n_seeds * 100


def plot_homophily(homophily_values, gcn_acc, mlp_acc, n_classes):
    """Accuracy réseau de graphe vs MLP en fonction de l'homophilie (écart ombré)."""
    hv = np.array(homophily_values)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(hv, mlp_acc, gcn_acc, where=(gcn_acc >= mlp_acc),
                    color="#4CAF50", alpha=0.12, label="gain du graphe")
    ax.plot(hv, gcn_acc, "-o", color="#4CAF50", label="Réseau de graphe (structure)")
    ax.plot(hv, mlp_acc, "-s", color="#9C27B0", label="MLP (features seules)")
    chance = 100.0 / max(n_classes, 2)
    ax.axhline(chance, color="gray", ls="--", lw=1, label=f"hasard ({chance:.0f} %)")
    ax.set_xlabel("homophilie du graphe  (0.5 = aléatoire  →  1.0 = communautés pures)")
    ax.set_ylabel("accuracy test (%)"); ax.set_ylim(0, 100)
    ax.set_title("À partir de quand le graphe sert ? — écart GCN vs MLP selon l'homophilie")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def hop_distances(A_bin, source):
    """
    Distance en nombre de SAUTS de `source` à tous les nœuds (BFS sur le graphe).
    Renvoie un vecteur d'entiers ; les nœuds non atteignables ont une distance
    « infinie » (n+1). C'est le champ réceptif : à L couches, un nœud agrège tout
    ce qui est à distance ≤ L.
    """
    n = A_bin.shape[0]
    # listes d'adjacence (rapide pour le BFS)
    nbrs = [np.where(A_bin[u] > 0)[0] for u in range(n)]
    dist = np.full(n, n + 1)
    dist[source] = 0
    frontier, d = [source], 0
    while frontier:
        d += 1
        nxt = []
        for u in frontier:
            for v in nbrs[u]:
                if dist[v] > d:
                    dist[v] = d
                    nxt.append(v)
        frontier = nxt
    return dist


def plot_receptive_field(X, y, edges, source, dist, hops):
    """
    Gauche : le graphe, nœuds à ≤ `hops` sauts colorés par distance (le reste en
    gris = hors champ), source entourée. Droite : fraction du graphe atteinte en
    fonction du nombre de sauts (→ 100 % = le nœud « voit » tout le graphe).
    """
    from matplotlib.collections import LineCollection
    n = len(y)
    reachable_max = int(dist[dist <= n].max())            # diamètre atteint depuis source
    in_field = dist <= hops

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.4, 1]})
    segs = [[(X[i, 0], X[i, 1]), (X[j, 0], X[j, 1])] for (i, j) in edges]
    axL.add_collection(LineCollection(segs, colors="lightgray", linewidths=0.3, alpha=0.5, zorder=1))
    axL.scatter(X[~in_field, 0], X[~in_field, 1], c="#dddddd", s=16, zorder=2, label="hors champ")
    sc = axL.scatter(X[in_field, 0], X[in_field, 1], c=dist[in_field], cmap="viridis_r",
                     vmin=0, vmax=max(hops, 1), s=32, edgecolors="white", linewidths=0.3, zorder=3)
    plt.colorbar(sc, ax=axL, fraction=0.046, pad=0.04, label="distance (sauts)")
    axL.scatter([X[source, 0]], [X[source, 1]], s=260, facecolors="none",
                edgecolors="red", linewidths=2.5, zorder=4)
    axL.set_title(f"Champ réceptif du nœud {source} à {hops} saut(s) — "
                  f"{int(in_field.sum())} nœuds atteints")
    axL.set_xlabel("x₁"); axL.set_ylabel("x₂"); axL.legend(fontsize=8, loc="upper right")

    ls = list(range(0, reachable_max + 1))
    frac = [100.0 * np.mean(dist <= L) for L in ls]
    axR.plot(ls, frac, "-o", color="#2196F3")
    axR.axvline(hops, color="red", ls="--", lw=1.2, label=f"{hops} saut(s)")
    axR.set_xlabel("nombre de sauts (= couches)"); axR.set_ylabel("% du graphe atteint")
    axR.set_ylim(0, 105); axR.set_title("Le champ réceptif grandit avec la profondeur")
    axR.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _grid(X, res=60):
    """Grille régulière couvrant le nuage de points (pour les régions de décision)."""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, res), np.linspace(y_min, y_max, res))
    return xx, yy, np.c_[xx.ravel(), yy.ravel()].astype(np.float32)


def gcn_decision_regions(gcn_model, X, A_hat, k, res=60, radius=None):
    """
    Régions de décision d'un GCN. Comme un GCN classe des NŒUDS (pas des points
    arbitraires), on insère la grille dans le graphe : chaque point de la grille
    est relié à ses k plus proches voisins du dataset (ou, si ``radius`` est
    donné, aux points du dataset à distance < radius — au moins le plus proche),
    par des arêtes dirigées grille → dataset, donc les nœuds du dataset gardent
    EXACTEMENT leur représentation d'entraînement. On lit ensuite la prédiction
    du GCN sur les nœuds-grille.
    """
    from sklearn.neighbors import NearestNeighbors
    xx, yy, G = _grid(X, res)
    n, g = len(X), len(G)
    nn_fit = NearestNeighbors(n_neighbors=k).fit(X)
    N = n + g
    A = torch.zeros((N, N), dtype=torch.float32)
    A[:n, :n] = A_hat                                       # bloc dataset = Â d'entraînement
    if radius is None:
        idx = nn_fit.kneighbors(G, return_distance=False)
        w = 1.0 / (k + 1)
        rows = torch.tensor(np.repeat(np.arange(g), k) + n)
        cols = torch.tensor(idx.ravel())
        A[rows, cols] = w                                   # grille → voisins dataset
        diag = torch.arange(n, N)
        A[diag, diag] = w                                   # self-loops des nœuds-grille
    else:
        nbrs = nn_fit.radius_neighbors(G, radius=radius, return_distance=False)
        nn1 = nn_fit.kneighbors(G, n_neighbors=1, return_distance=False)
        for i, nb in enumerate(nbrs):
            nb = nb if len(nb) else nn1[i]                  # nœud-grille jamais orphelin
            w = 1.0 / (len(nb) + 1)
            A[n + i, torch.tensor(np.asarray(nb, dtype=np.int64))] = w
            A[n + i, n + i] = w
    Xaug = torch.tensor(np.vstack([X, G]), dtype=torch.float32)
    gcn_model.eval()
    with torch.no_grad():
        out = gcn_model(Xaug, A).numpy()[n:]
    pred = out.argmax(1) if out.shape[1] > 1 else (out[:, 0] > 0).astype(int)
    return xx, yy, pred.reshape(xx.shape)


def mlp_decision_regions(mlp_model, X, res=60):
    """Régions de décision d'un MLP (il classe chaque point de la grille directement)."""
    xx, yy, G = _grid(X, res)
    mlp_model.eval()
    with torch.no_grad():
        out = mlp_model(torch.tensor(G)).numpy()
    pred = out.argmax(1) if out.shape[1] > 1 else (out[:, 0] > 0).astype(int)
    return xx, yy, pred.reshape(xx.shape)


def plot_regions_compare(X, y, reg_gcn, reg_mlp, acc_gcn, acc_mlp, left_label="GCN"):
    """Régions de décision du réseau de graphe (gauche) vs MLP (droite)."""
    from matplotlib.colors import ListedColormap
    K = max(n_classes_eff, 2)
    cmap = ListedColormap(CLASS_COLORS[:K])
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (xx, yy, R), title, acc in [
        (axes[0], reg_gcn, left_label, acc_gcn),
        (axes[1], reg_mlp, "MLP", acc_mlp),
    ]:
        ax.contourf(xx, yy, R, levels=np.arange(-0.5, K, 1.0), cmap=cmap, alpha=0.35)
        if K > 1:
            ax.contour(xx, yy, R, levels=[i + 0.5 for i in range(K - 1)],
                       colors="black", linewidths=1.0)
        for c in range(n_classes_eff):
            m = y == c
            ax.scatter(X[m, 0], X[m, 1], c=CLASS_COLORS[c % len(CLASS_COLORS)], s=18,
                       edgecolors="white", linewidths=0.3, zorder=3)
        ax.set_title(f"{title} — test acc {acc * 100:.1f}%")
        ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
    return fig


def radius_density_study(r, n_seeds=3):
    """
    MLP vs GCN (k-NN) vs GCN (rayon) sur le dataset courant, même protocole que
    l'entraînement du mode graphe (transductif, full-batch), moyenné sur n_seeds.
    Renvoie {nom: (acc_moy, acc_std, rappel_classe1_moy)} sur les nœuds de test.
    """
    n = len(y)
    tr, _, te = gcn_masks(n)
    m_tr = torch.tensor(np.isin(np.arange(n), tr))
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = (torch.tensor(y, dtype=torch.long) if is_multiclass
          else torch.tensor(y, dtype=torch.float32).view(-1, 1))
    crit = nn.CrossEntropyLoss() if is_multiclass else nn.BCEWithLogitsLoss()
    od = n_classes_eff if is_multiclass else 1
    A_rad, Ab_rad = build_radius(X, r=r)
    graphs = {"GCN (k-NN)": build_knn(X, k=knn_k)[0],
              "GCN (rayon)": A_rad}
    # Degré du graphe par rayon (normalisé) donné en 3e feature : même info de
    # densité que le graphe, mais sous une forme que le réseau lit sans agrégation.
    deg_r = Ab_rad.sum(1)
    deg_feat = (deg_r - deg_r.mean()) / (deg_r.std() + 1e-9)
    X_deg = torch.tensor(np.hstack([X, deg_feat[:, None]]), dtype=torch.float32)
    res = {name: []
           for name in ["MLP (features)"] + list(graphs) + ["MLP + degré (rayon)"]}

    def _scores(p):
        rec1 = float((p[te][y[te] == 1] == 1).mean()) if (y[te] == 1).any() else float("nan")
        return float((p[te] == y[te]).mean()), rec1

    for ws in range(n_seeds):
        for name, Ah in graphs.items():
            torch.manual_seed(ws)
            gm = GCN(input_dim=2, gcn_layers=[neurons_per_layer] * gcn_layers_n,
                     head_layers=[max(neurons_per_layer // 2, 4)], output_dim=od,
                     activation=activation, aggregation=gcn_agg, heads=gat_heads,
                     use_batchnorm=use_batchnorm)
            opt = torch.optim.Adam(gm.parameters(), lr=learning_rate)
            torch.manual_seed(ws)
            for _ in range(n_epochs):
                gm.train(); opt.zero_grad()
                crit(gm(Xt, Ah)[m_tr], yt[m_tr]).backward(); opt.step()
            gm.eval()
            with torch.no_grad():
                o = gm(Xt, Ah).numpy()
            res[name].append(_scores(o.argmax(1) if is_multiclass
                                     else (o[:, 0] > 0).astype(int)))
        torch.manual_seed(ws)
        ml = MLP(2, [neurons_per_layer, neurons_per_layer], od, activation=activation)
        opt = torch.optim.Adam(ml.parameters(), lr=learning_rate)
        torch.manual_seed(ws)
        for _ in range(n_epochs):
            ml.train(); opt.zero_grad()
            crit(ml(Xt[tr]), yt[tr]).backward(); opt.step()
        ml.eval()
        with torch.no_grad():
            o = ml(Xt).numpy()
        res["MLP (features)"].append(_scores(o.argmax(1) if is_multiclass
                                             else (o[:, 0] > 0).astype(int)))
        torch.manual_seed(ws)
        md = MLP(3, [neurons_per_layer, neurons_per_layer], od, activation=activation)
        opt = torch.optim.Adam(md.parameters(), lr=learning_rate)
        torch.manual_seed(ws)
        for _ in range(n_epochs):
            md.train(); opt.zero_grad()
            crit(md(X_deg[tr]), yt[tr]).backward(); opt.step()
        md.eval()
        with torch.no_grad():
            o = md(X_deg).numpy()
        res["MLP + degré (rayon)"].append(_scores(o.argmax(1) if is_multiclass
                                                  else (o[:, 0] > 0).astype(int)))

    return {name: (float(np.mean([a for a, _ in v])), float(np.std([a for a, _ in v])),
                   float(np.mean([rc for _, rc in v])))
            for name, v in res.items()}


def plot_radius_density(res, r):
    """Barres : accuracy test (± std sur les seeds) et rappel classe 1, par modèle."""
    names = list(res)
    palette = {"MLP (features)": "#9C27B0", "GCN (k-NN)": "#4CAF50",
               "GCN (rayon)": "#FF9800", "MLP + degré (rayon)": "#607D8B"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, vi, err, title in [
        (axes[0], 0, [res[nm][1] * 100 for nm in names], "Accuracy test"),
        (axes[1], 2, None, "Rappel de la classe 1 (la clairsemée sur Density)"),
    ]:
        ax.bar(range(len(names)), [res[nm][vi] * 100 for nm in names],
               yerr=err, capsize=4, color=[palette.get(nm, "#607D8B") for nm in names])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([nm.replace(" (", "\n(") for nm in names], fontsize=8)
        ax.set_ylim(0, 100); ax.set_ylabel("%"); ax.set_title(title)
    fig.suptitle(f"Le graphe par rayon (r={r:g}) rend-il la densité au réseau de graphe ?")
    fig.tight_layout()
    return fig


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
    st.session_state.train_report = None      # métriques sur le train (pour l'écart train/test)
    st.session_state.test_X = None            # points de test (pour la viz de généralisation)
    st.session_state.test_y = None
    st.session_state.gcn = None               # résultat du dernier entraînement GCN


# ─────────────────────────────────────────────
# Layout principal
# ─────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Données d'entraînement")
    fig_data, ax_data = plt.subplots(figsize=(7, 6))
    from matplotlib.lines import Line2D
    # Couleur = classe (jusqu'à K=5) ; forme = échantillon.
    # Train : ronds pleins, Val : triangles, Test : croix.
    splits = [
        ("Train", X_train, y_train, "o", "white", 0.4, 40),
        ("Val",   X_val,   y_val,   "^", "black", 0.8, 50),
        # "x" est un marker non rempli → edgecolor=None (évite un warning mpl)
        ("Test",  X_test,  y_test,  "x", None,    0.8, 55),
    ]
    for _, Xs, ys, mk, edge, lw, s in splits:
        for cls in range(n_classes_eff):
            m = ys == cls
            ax_data.scatter(Xs[m, 0], Xs[m, 1],
                            c=CLASS_COLORS[cls % len(CLASS_COLORS)],
                            edgecolors=edge, linewidths=lw, s=s, marker=mk)
    ax_data.set_xlabel("x₁")
    ax_data.set_ylabel("x₂")
    ax_data.set_title(f"Dataset — {dataset_name}  (train={len(y_train)}, val={len(y_val)}, test={len(y_test)})")
    # Légende factorisée : un bloc couleurs (les classes) + un bloc formes (les
    # échantillons). Évite l'explosion en K×3 entrées illisibles quand K augmente.
    class_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=7,
               markerfacecolor=CLASS_COLORS[c % len(CLASS_COLORS)],
               markeredgecolor="white", label=f"Classe {c}")
        for c in range(n_classes_eff)
    ]
    split_handles = [
        Line2D([0], [0], marker=mk, linestyle="", markersize=8,
               color="dimgray", label=name)
        for name, _, _, mk, _, _, _ in splits
    ]
    leg_classes = ax_data.legend(handles=class_handles, title="Classe",
                                 fontsize=7, loc="upper left", framealpha=0.9)
    ax_data.add_artist(leg_classes)
    ax_data.legend(handles=split_handles, title="Échantillon",
                   fontsize=7, loc="upper right", framealpha=0.9)
    st.pyplot(fig_data)
    plt.close(fig_data)

with col2:
    st.subheader("Frontière de décision")

    boundary_placeholder = st.empty()
    status_placeholder = st.empty()

    if (not is_graph) and st.button("Entraîner le réseau", type="primary"):
        optimizer = make_optimizer(model)
        criterion = nn.CrossEntropyLoss() if is_multiclass else nn.BCEWithLogitsLoss()
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
        # On évalue AUSSI sur le train : l'écart train↔test mesure l'overfitting
        # (gros écart = le réseau mémorise au lieu de généraliser).
        train_report = evaluate(model, train_loader)

        # Persistance pour le slider d'exploration et l'affichage des métriques
        st.session_state.trainer = trainer
        st.session_state.trained_X = X_train
        st.session_state.trained_y = y_train
        st.session_state.trained_mode = mode
        st.session_state.test_loader = test_loader
        st.session_state.test_report = test_report
        st.session_state.train_report = train_report
        st.session_state.test_X = X_test
        st.session_state.test_y = y_test
    else:
        # S'il n'y a pas eu d'entraînement encore, on affiche le message d'invite
        if is_graph:
            boundary_placeholder.info(
                "**Mode réseau de graphe** : l'entraînement et la visualisation du "
                "graphe se font dans la section ci-dessous ⬇️"
            )
        elif st.session_state.trainer is None:
            boundary_placeholder.info(
                "Configure les paramètres dans la sidebar, puis clique sur **Entraîner le réseau**."
            )


# ─────────────────────────────────────────────
# Mode réseau de graphe (GCN / GraphSAGE / GAT) : graphe k-NN, transductif
# ─────────────────────────────────────────────
if is_graph:
    _agg_desc = {
        "mean": "chaque nœud agrège ses voisins par une **moyenne** à poids fixes (GCN)",
        "max": "chaque nœud agrège ses voisins par un **max** par dimension (GraphSAGE)",
        "attention": "chaque nœud pondère ses voisins par une **attention apprise** "
                     "multi-têtes (GAT)",
    }[gcn_agg]
    st.markdown("---")
    st.subheader(f"Réseau de graphe — agrégation {agg_label}")
    if is_sbm:
        st.caption(
            f"Dataset **SBM** : le graphe est **fourni** (communautés), et {_agg_desc}, "
            "puis une tête MLP classe chaque nœud. Les features 2D sont **faibles** "
            "(communautés chevauchantes) → un MLP qui ne voit que les features échoue, "
            "alors que le réseau de graphe exploite la **structure** des communautés. "
            "Entraînement **transductif** (le slider *k* est ignoré : le graphe n'est pas "
            "construit par k-NN). BatchNorm activée d'office (stabilise l'entraînement "
            "sur features faibles)."
        )
    else:
        st.caption(
            f"On construit un graphe de voisinage sur le nuage de points (k-NN ou rayon, "
            f"au choix ci-dessous) : {_agg_desc}, puis une "
            "tête MLP classe chaque nœud. Entraînement **transductif** : un seul graphe sur "
            "tous les points, perte calculée sur les nœuds d'entraînement (les nœuds de test "
            "participent à la propagation, mais leur label n'est pas vu)."
        )

    # SBM : graphe FOURNI (communautés) ; sinon graphe construit sur les coordonnées,
    # par k-NN (degré ~constant) ou par RAYON (le degré reflète la densité locale).
    use_radius, radius_r = False, 0.40
    if not is_sbm:
        bcol1, bcol2 = st.columns([1, 1])
        with bcol1:
            graph_build = st.radio(
                "Construction du graphe", ["k-NN (k voisins)", "rayon (r)"],
                horizontal=True,
                help=(
                    "**k-NN** : chaque point est relié à ses k plus proches voisins — tout "
                    "le monde reçoit ~k voisins, le degré n'encode PAS la densité locale. "
                    "**Rayon** : on relie tout ce qui est à distance < r — beaucoup de "
                    "voisins en zone dense, peu (voire zéro) en zone clairsemée : le degré "
                    "reflète la densité."
                ),
            )
            use_radius = graph_build.startswith("rayon")
        with bcol2:
            radius_r = st.slider(
                "Rayon r", 0.05, 2.0, 0.40, 0.05, disabled=not use_radius,
                help=(
                    "Distance de connexion du graphe par rayon. Trop petit → le graphe "
                    "s'émiette (nœuds isolés) ; trop grand → tout est relié à tout "
                    "(sur-lissage assuré)."
                ),
            )
    if is_sbm:
        A_hat, A_bin = normalize_adj(A_sbm), A_sbm
    elif use_radius:
        A_hat, A_bin = build_radius(X, r=radius_r)
    else:
        A_hat, A_bin = build_knn(X, k=knn_k)
    edges = knn_edges(A_bin)
    if not is_sbm:
        deg_g = A_bin.sum(1)
        per_class = "  ·  ".join(
            f"classe {c} : {deg_g[y == c].mean():.1f}" for c in range(n_classes_eff)
        )
        st.caption(
            f"Degré moyen : **{deg_g.mean():.1f}** ({per_class})  ·  nœuds isolés : "
            f"{int((deg_g == 0).sum())}. Le k-NN égalise les degrés entre zones denses et "
            "clairsemées ; le rayon les fait refléter la densité locale."
        )

    graph_title = ("Graphe communautaire (SBM)" if is_sbm else
                   f"Graphe par rayon (r={radius_r:g}) — {len(edges)} arêtes" if use_radius
                   else f"Graphe k-NN (k={knn_k}) — {len(edges)} arêtes")
    gcol1, gcol2 = st.columns(2)
    with gcol1:
        st.markdown("**Graphe communautaire (SBM)**" if is_sbm else "**Graphe des voisins**")
        st.pyplot(plot_gcn_graph(X, y, edges, title=graph_title))
        plt.close("all")

    with gcol2:
        st.markdown("**Entraînement & prédictions**")
        if st.button("Entraîner le réseau de graphe", type="primary"):
            n = len(y)
            tr_idx, val_idx, te_idx = gcn_masks(n)
            m_tr = torch.tensor(np.isin(np.arange(n), tr_idx))
            m_val = torch.tensor(np.isin(np.arange(n), val_idx))
            Xt = torch.tensor(X, dtype=torch.float32)
            yt = (torch.tensor(y, dtype=torch.long) if is_multiclass
                  else torch.tensor(y, dtype=torch.float32).view(-1, 1))

            torch.manual_seed(int(weight_seed))
            gcn = build_gcn_model()
            opt = torch.optim.Adam(gcn.parameters(), lr=learning_rate)
            crit = nn.CrossEntropyLoss() if is_multiclass else nn.BCEWithLogitsLoss()

            hist = {"train": [], "validation": []}
            prog = st.progress(0.0, text="Entraînement du réseau de graphe…")
            torch.manual_seed(int(weight_seed))
            for epoch in range(n_epochs):
                gcn.train()
                opt.zero_grad()
                out = gcn(Xt, A_hat)
                loss = crit(out[m_tr], yt[m_tr])
                loss.backward()
                opt.step()
                gcn.eval()
                with torch.no_grad():
                    vloss = crit(gcn(Xt, A_hat)[m_val], yt[m_val]).item()
                hist["train"].append(loss.item())
                hist["validation"].append(vloss)
                prog.progress((epoch + 1) / n_epochs)
            prog.empty()

            gcn.eval()
            with torch.no_grad():
                out = gcn(Xt, A_hat).numpy()
            pred = out.argmax(1) if is_multiclass else (out[:, 0] > 0).astype(int)
            st.session_state.gcn = dict(
                X=X, y=y, edges=edges, pred=pred, te_idx=te_idx, tr_idx=tr_idx,
                acc_test=float((pred[te_idx] == y[te_idx]).mean()),
                acc_train=float((pred[tr_idx] == y[tr_idx]).mean()),
                hist=hist, model=gcn, A_hat=A_hat, A_bin=A_bin, k=knn_k, agg=gcn_agg,
                build=("sbm" if is_sbm else "radius" if use_radius else "knn"), r=radius_r,
            )

        g = st.session_state.gcn
        if g is not None:
            m1, m2 = st.columns(2)
            m1.metric("Accuracy test", f"{g['acc_test'] * 100:.1f} %")
            m2.metric("Accuracy train", f"{g['acc_train'] * 100:.1f} %")
            st.pyplot(plot_gcn_pred(g["X"], g["y"], g["pred"], g["te_idx"]))
            plt.close("all")
            st.line_chart(g["hist"])
        else:
            st.info("Clique sur **Entraîner le réseau de graphe**.")

    # ─── Précision / rappel par classe (sur les nœuds de test) ───
    g = st.session_state.gcn
    if g is not None:
        with st.expander("Approfondir — précision / rappel par classe (nœuds de test)"):
            st.caption(
                "Mêmes métriques que pour le MLP, calculées sur les nœuds de **test** du "
                "graphe (jamais supervisés). Matrice de confusion (lignes = vraie classe, "
                "colonnes = prédiction) et précision / rappel / F1 par classe."
            )
            from sklearn.metrics import confusion_matrix
            yt_te, yp_te = g["y"][g["te_idx"]], g["pred"][g["te_idx"]]
            K_g = int(max(g["y"].max(), g["pred"].max())) + 1     # robuste au binaire (K=2)
            cm_g = confusion_matrix(yt_te, yp_te, labels=list(range(K_g)))

            cc1, cc2 = st.columns([1, 1])
            with cc1:
                st.pyplot(plot_confusion_matrix(cm_g, f"Confusion — nœuds de test ({agg_label})"))
                plt.close("all")
            with cc2:
                st.table(per_class_table(cm_g))
            st.caption(
                "**Précision** d'une classe : parmi les nœuds *prédits* dans cette classe, "
                "combien sont corrects. **Rappel** : parmi les nœuds *réellement* de cette "
                "classe, combien sont retrouvés. **Support** : nb de nœuds de test de la classe."
            )

    # ─── Espace latent du réseau de graphe (embeddings via encode()) ───
    g = st.session_state.gcn
    if g is not None and g.get("model") is not None:
        with st.expander(f"Approfondir — espace latent ({agg_label}) : ce que « voient » les couches de graphe"):
            st.caption(
                "À gauche, l'espace d'entrée (coordonnées brutes). À droite, la "
                "représentation des nœuds renvoyée par `encode()` (après les couches de "
                "graphe, avant la tête), projetée en 2D par PCA si sa dimension dépasse 2. "
                "Plus les classes y sont séparées, plus la tête a un travail facile — c'est "
                "ce que résume le **score de séparabilité** (silhouette, de -1 à 1)."
            )
            st.pyplot(plot_gcn_latent(g["model"], g["X"], g["y"], g["A_hat"]))
            plt.close("all")

    # ─── Attention : sur quels voisins le modèle se concentre (GAT) ───
    g = st.session_state.gcn
    if g is not None and g.get("model") is not None and g.get("agg") == "attention":
        with st.expander("Approfondir — attention : sur quels voisins le modèle se concentre"):
            st.caption(
                "Chaque nœud du GAT pondère ses voisins par une **attention apprise** α. "
                "Les arêtes sont colorées/épaissies selon cette attention (moyenne des "
                "têtes). Choisis un nœud pour voir sur quels voisins il « se concentre ». "
                "⚠️ L'attention indique *quels voisins pèsent*, mais elle n'est **pas** une "
                "explication complète du modèle (résultat connu : moyenner les poids "
                "d'attention ne suffit pas à interpréter un GAT)."
            )
            alpha = gcn_attention(g["model"], g["X"], g["A_hat"])
            focus = st.slider("Nœud à inspecter", 0, len(g["y"]) - 1, 0, key="att_focus")
            ac1, ac2 = st.columns([3, 2])
            with ac1:
                st.pyplot(plot_attention_graph(g["X"], g["y"], g["edges"], alpha,
                                               g["A_bin"], focus=focus))
                plt.close("all")
            with ac2:
                st.pyplot(plot_node_attention_bars(focus, alpha, g["A_bin"], g["y"]))
                plt.close("all")

            # Insight quantitatif : attention moyenne même classe vs classe différente,
            # comparée à l'uniforme (1/degré) → diagnostique si le GAT « sélectionne » vraiment.
            same = diff = 0.0
            ns = nd = 0
            for i, j in g["edges"]:
                a = (alpha[i, j] + alpha[j, i]) / 2
                if g["y"][i] == g["y"][j]:
                    same += a; ns += 1
                else:
                    diff += a; nd += 1
            if ns and nd:
                sm, dm = same / ns, diff / nd
                uni = 1.0 / (g["A_bin"].sum(1).mean() + 1)   # attention uniforme moyenne ≈ 1/(degré+1)
                if sm > 1.15 * dm:
                    verdict = "→ le GAT **privilégie** nettement les voisins de même classe."
                else:
                    verdict = (f"→ l'attention est **quasi uniforme** (≈ 1/degré = {uni:.3f}) : "
                               "sur ces données 2D, le GAT se comporte presque comme la "
                               "**moyenne (GCN)** — c'est pourquoi *GAT ≈ GCN* ici.")
                st.caption(
                    f"Attention moyenne par arête : **même classe** {sm:.3f} vs "
                    f"**classe différente** {dm:.3f}. {verdict}"
                )

    # ─── SBM : le résultat — réseau de graphe (structure) vs MLP (features seules) ───
    g = st.session_state.gcn
    if is_sbm and g is not None and g.get("model") is not None:
        st.markdown("---")
        st.markdown(f"**Le résultat — {agg_label} (structure) vs MLP (features seules)**")
        st.caption(
            "On entraîne un MLP sur les **mêmes features 2D** (sans le graphe), même "
            "découpage. Il ne voit que des features faibles → il échoue ; le réseau de "
            "graphe, lui, exploite la **structure** des communautés."
        )
        out_dim = n_classes_eff if is_multiclass else 1
        torch.manual_seed(int(weight_seed))
        mlp_cmp = MLP(2, [neurons_per_layer, neurons_per_layer], out_dim, activation=activation)
        opt = torch.optim.Adam(mlp_cmp.parameters(), lr=learning_rate)
        crit = nn.CrossEntropyLoss() if is_multiclass else nn.BCEWithLogitsLoss()
        Xtr = torch.tensor(g["X"][g["tr_idx"]], dtype=torch.float32)
        ytr = (torch.tensor(g["y"][g["tr_idx"]], dtype=torch.long) if is_multiclass
               else torch.tensor(g["y"][g["tr_idx"]], dtype=torch.float32).view(-1, 1))
        torch.manual_seed(int(weight_seed))
        for _ in range(n_epochs):
            mlp_cmp.train(); opt.zero_grad(); crit(mlp_cmp(Xtr), ytr).backward(); opt.step()
        mlp_cmp.eval()
        with torch.no_grad():
            mout = mlp_cmp(torch.tensor(g["X"], dtype=torch.float32)).numpy()
        mpred = mout.argmax(1) if is_multiclass else (mout[:, 0] > 0).astype(int)
        acc_mlp = float((mpred[g["te_idx"]] == g["y"][g["te_idx"]]).mean())

        cc1, cc2 = st.columns([1, 1])
        with cc1:
            mm1, mm2 = st.columns(2)
            mm1.metric(f"Réseau de graphe ({agg_label})", f"{g['acc_test'] * 100:.1f} %")
            mm2.metric("MLP (features seules)", f"{acc_mlp * 100:.1f} %",
                       delta=f"{(acc_mlp - g['acc_test']) * 100:.1f} pts")
        with cc2:
            fig_cmp, ax_cmp = plt.subplots(figsize=(5, 4))
            chance = 100.0 / max(n_classes_eff, 2)
            ax_cmp.bar(["MLP\n(features)", f"{agg_label}\n(graphe)"],
                       [acc_mlp * 100, g["acc_test"] * 100], color=["#9C27B0", "#4CAF50"])
            ax_cmp.axhline(chance, color="gray", ls="--", lw=1, label=f"hasard ({chance:.0f} %)")
            ax_cmp.set_ylim(0, 100); ax_cmp.set_ylabel("accuracy test (%)"); ax_cmp.legend(fontsize=8)
            fig_cmp.tight_layout(); st.pyplot(fig_cmp); plt.close(fig_cmp)
        st.success(
            f"Le réseau de graphe atteint **{g['acc_test'] * 100:.0f} %** là où le MLP "
            f"plafonne à **{acc_mlp * 100:.0f} %** (≈ hasard) : **c'est la structure du "
            "graphe qui porte l'information**, pas les features."
        )

    # ─── SBM : loi de l'homophilie — « à partir de quand le graphe sert ? » ───
    if is_sbm:
        st.markdown("---")
        st.markdown("**Loi de l'homophilie — à partir de quand le graphe sert ?**")
        st.caption(
            "On balaie l'**homophilie du graphe** (0.5 = aléatoire → 1.0 = communautés "
            "pures, à degré moyen constant) et on compare, à chaque valeur, un réseau de "
            "graphe (structure) à un MLP (features seules). Le MLP est **insensible** à "
            "l'homophilie ; le réseau de graphe **gagne proportionnellement** à l'homophilie."
        )
        if st.checkbox("Balayer l'homophilie (un peu lent)", key="homophily_sweep"):
            hvals = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
            with st.spinner("Entraînement sur toute la plage d'homophilie (3 seeds)…"):
                ga_h, ma_h = homophily_study(n_samples, n_classes, noise, int(seed), hvals)
            st.pyplot(plot_homophily(hvals, ga_h, ma_h, n_classes))
            plt.close("all")
            st.caption(
                f"À gauche (aléatoire) le réseau de graphe est au niveau du MLP "
                f"({ga_h[0]:.0f} % vs {ma_h[0]:.0f} %) ; à droite (communautés pures) il "
                f"**écrase** le MLP ({ga_h[-1]:.0f} % vs {ma_h[-1]:.0f} %). ➡️ **Un GNN aide "
                "proportionnellement à l'information portée par le graphe** — la réponse à "
                "« quand un réseau de graphe bat-il un MLP ? »."
            )

    # ─── Régions de décision : réseau de graphe vs MLP (datasets à coordonnées) ───
    g = st.session_state.gcn
    if (not is_sbm) and g is not None and g.get("model") is not None:
        st.markdown("---")
        st.markdown(f"**Régions de décision — {agg_label} vs MLP**")
        st.caption(
            "Pour colorer le plan, on insère une grille de points dans le graphe "
            "(chaque point relié à ses k plus proches voisins du dataset, sans modifier "
            "les nœuds existants). On compare à un MLP entraîné sur les mêmes points."
        )
        if st.checkbox("Calculer les régions de décision (un peu lent)"):
            with st.spinner("Calcul des régions…"):
                reg_gcn = gcn_decision_regions(
                    g["model"], g["X"], g["A_hat"], g["k"],
                    radius=(g["r"] if g.get("build") == "radius" else None))
                out_dim = n_classes_eff if is_multiclass else 1
                torch.manual_seed(int(weight_seed))
                mlp_cmp = MLP(2, [neurons_per_layer, neurons_per_layer], out_dim,
                              activation=activation)
                opt = torch.optim.Adam(mlp_cmp.parameters(), lr=learning_rate)
                crit = nn.CrossEntropyLoss() if is_multiclass else nn.BCEWithLogitsLoss()
                Xtr = torch.tensor(g["X"][g["tr_idx"]], dtype=torch.float32)
                ytr = (torch.tensor(g["y"][g["tr_idx"]], dtype=torch.long) if is_multiclass
                       else torch.tensor(g["y"][g["tr_idx"]], dtype=torch.float32).view(-1, 1))
                torch.manual_seed(int(weight_seed))
                for _ in range(n_epochs):
                    mlp_cmp.train()
                    opt.zero_grad()
                    crit(mlp_cmp(Xtr), ytr).backward()
                    opt.step()
                reg_mlp = mlp_decision_regions(mlp_cmp, g["X"])
                mlp_cmp.eval()
                with torch.no_grad():
                    mout = mlp_cmp(torch.tensor(g["X"], dtype=torch.float32)).numpy()
                mpred = mout.argmax(1) if is_multiclass else (mout[:, 0] > 0).astype(int)
                acc_mlp = float((mpred[g["te_idx"]] == g["y"][g["te_idx"]]).mean())
            st.pyplot(plot_regions_compare(g["X"], g["y"], reg_gcn, reg_mlp,
                                           g["acc_test"], acc_mlp, left_label=agg_label))
            plt.close("all")
            _gdesc = (f"rayon r = {g['r']:g}" if g.get("build") == "radius"
                      else f"k = {g['k']}")
            st.caption(
                f"Graphe à {_gdesc}. Change la **construction du graphe**, **k**/**r** ou "
                "l'**agrégation**, puis ré-entraîne pour voir l'effet sur la frontière."
            )

    # ─── Étude : le graphe par rayon rend-il la densité au réseau de graphe ? ───
    if not is_sbm:
        st.markdown("---")
        st.markdown("**Étude — densité : graphe par rayon vs k-NN**")
        st.caption(
            "Le k-NN donne ~k voisins à tout le monde : le **degré** n'encode pas la "
            "densité locale (sur Density : degré ≈ 7 pour les deux classes alors que la "
            "densité réelle diffère d'un facteur 4). Le graphe par **rayon**, lui, remet "
            "la densité dans le degré. Mais l'agrégation **moyenne normalisée** (Kipf) "
            "divise justement par le degré — elle est conçue pour gommer ces écarts... "
            "L'étude entraîne, sur le dataset courant, un MLP, un GCN sur graphe k-NN et "
            "un GCN sur graphe par rayon (mêmes réglages, 3 seeds d'initialisation)."
        )
        if st.checkbox("Lancer l'étude densité (MLP vs k-NN vs rayon vs degré-feature, 3 seeds)",
                       key="radius_study"):
            with st.spinner("Entraînement des quatre modèles (3 seeds)…"):
                res_rd = radius_density_study(radius_r)
            st.pyplot(plot_radius_density(res_rd, radius_r))
            plt.close("all")
            best = max(res_rd, key=lambda nm: res_rd[nm][0])
            st.caption(
                f"Meilleure accuracy test : **{best}** ({res_rd[best][0] * 100:.1f} %). "
                "Grille de lecture : si « GCN (rayon) » ne dépasse pas « GCN (k-NN) », le "
                "degré est bien revenu dans le graphe mais l'agrégation moyenne **ne sait "
                "pas le lire** — la normalisation de Kipf divise précisément par le degré. "
                "La barre « MLP + degré » reçoit la même info de densité en *feature* : si "
                "elle seule décolle, l'info était là et c'est l'agrégation qui la gomme. "
                "Si rien ne décolle, la densité est **redondante avec la position** sur ce "
                "dataset (cas Density : la zone dense est aussi la zone de gauche)."
            )

    # ─── Champ réceptif : jusqu'où l'info d'un nœud se propage ───
    with st.expander("Approfondir — champ réceptif : jusqu'où un nœud « voit » selon la profondeur"):
        st.caption(
            "Après L couches de graphe, un nœud a agrégé l'information de tous les nœuds à "
            "≤ L sauts de lui : c'est son champ réceptif. Choisis un nœud et un nombre de "
            "sauts, et regarde la zone atteinte grandir. C'est le mécanisme derrière le "
            "sur-lissage : quand le champ couvre tout le graphe, tous les nœuds voient la "
            "même chose, donc leurs représentations s'effondrent."
        )
        rf1, rf2 = st.columns(2)
        rf_src = rf1.slider("Nœud source", 0, len(y) - 1, 0, key="rf_node")
        rf_hops = rf2.slider("Nombre de sauts (= couches)", 1, 10, 2, key="rf_hops")
        rf_dist = hop_distances(A_bin, rf_src)
        st.pyplot(plot_receptive_field(X, y, edges, rf_src, rf_dist, rf_hops))
        plt.close("all")
        reached = 100.0 * np.mean(rf_dist <= rf_hops)
        full_at = int(rf_dist[rf_dist <= len(y)].max())
        st.caption(
            f"À {rf_hops} saut(s), le nœud {rf_src} atteint {reached:.0f} % du graphe. Son "
            f"champ réceptif est complet à {full_at} sauts (il « voit » alors "
            f"{100.0 * np.mean(rf_dist <= full_at):.0f} % du graphe). Au-delà, empiler des "
            "couches ne fait qu'uniformiser les nœuds (sur-lissage)."
        )

    # ─── Sur-lissage (oversmoothing) : effet de la profondeur ───
    if not is_sbm:
        st.markdown("---")
        st.markdown("**Sur-lissage (oversmoothing) — pourquoi empiler trop de couches nuit**")
        st.caption(
            "En empilant beaucoup de couches de graphe, chaque nœud agrège un voisinage "
            "de plus en plus large → à la limite tous les nœuds se ressemblent (leurs "
            "embeddings **s'effondrent**) et le réseau ne distingue plus les classes. On "
            "entraîne des GCN de profondeur croissante, **avec et sans BatchNorm**, et on "
            "mesure l'**énergie de Dirichlet** (différence moyenne entre voisins, → 0 = "
            "effondrement). Plus net sur les datasets complexes (Spirals, Circles)."
        )
        if st.checkbox("Lancer l'étude de profondeur (un peu lent)", key="oversmooth"):
            with st.spinner("Entraînement à profondeurs croissantes (sans / avec BatchNorm, 3 seeds)…"):
                res_os = oversmoothing_study(X, y, A_hat, edges, [2, 4, 6, 8, 12, 16, 20])
            st.pyplot(plot_oversmoothing(res_os, y))
            plt.close("all")
            st.caption(
                f"Énergie de Dirichlet : **{res_os['energy'][0]:.3f}** à "
                f"{res_os['depths'][0]} couche(s) → **{res_os['energy'][-1]:.3f}** à "
                f"{res_os['depths'][-1]} (→ 0 = effondrement). Sans BatchNorm, le latent "
                "s'effondre et l'accuracy chute en profondeur ; **la BatchNorm mitige** le "
                "sur-lissage — un rôle clé de la normalisation (cf. la demande de Barthe)."
            )

    # En mode graphe (GCN/GraphSAGE/GAT), on s'arrête là : les sections MLP ne s'appliquent pas.
    st.stop()


# ─────────────────────────────────────────────
# Mode multi-classes : note sur les sections en cours d'adaptation
# ─────────────────────────────────────────────
if is_multiclass and st.session_state.trainer is not None:
    st.markdown("---")
    st.info(
        f"🎨 Mode **multi-classes** actif ({n_classes_eff} classes). Toutes les "
        "visualisations sont disponibles : frontière, métriques (macro **et par "
        "classe**), matrice de confusion K×K, espace latent, généralisation et k-fold."
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
        colors = CLASS_COLORS
        for cls in range(n_classes_eff):
            m = by == cls
            ax_in.scatter(bX[m, 0], bX[m, 1], c=colors[cls % len(colors)],
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
        if latent_dim == 1:
            st.markdown("**Espace latent 1D** (sortie du bottleneck — un seul nombre)")
            fig_lat1d = plot_latent_space_1d(bm, bX, by)
            st.pyplot(fig_lat1d)
            plt.close(fig_lat1d)
        elif latent_dim == 2:
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
            st.session_state.train_report = None
            st.session_state.test_X = None
            st.session_state.test_y = None
            st.rerun()


# ─────────────────────────────────────────────
# Généralisation — train vs test
# ─────────────────────────────────────────────
if (
    st.session_state.train_report is not None
    and st.session_state.test_report is not None
    and st.session_state.test_X is not None
):
    st.markdown("---")
    st.subheader("Généralisation — le réseau tient-il sur des données jamais vues ?")
    st.caption(
        "On compare la performance sur les données d'entraînement (vues) et sur "
        "le test (jamais vues). Un grand écart train → test signale de "
        "l'**overfitting** : le réseau a mémorisé le train au lieu d'apprendre "
        "la règle générale."
    )

    tr = st.session_state.train_report
    te = st.session_state.test_report
    gap = tr.accuracy - te.accuracy

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Accuracy train", f"{tr.accuracy * 100:.1f} %")
    with c2:
        st.metric("Accuracy test", f"{te.accuracy * 100:.1f} %")
    with c3:
        st.metric(
            "Écart train → test", f"{gap * 100:+.1f} pts",
            help="train − test. Proche de 0 = généralise bien. Grand = overfitting.",
        )

    # Diagnostic automatique de l'écart
    if gap > 0.15:
        st.warning(
            f"⚠️ Overfitting marqué : {gap * 100:.0f} points de mieux sur le "
            "train que sur le test → le réseau mémorise au lieu de généraliser. "
            "Pistes : dropout, moins de capacité, plus de points, early stopping."
        )
    elif gap > 0.05:
        st.info(f"ℹ️ Léger overfitting ({gap * 100:.0f} pts d'écart train → test). À surveiller.")
    else:
        st.success(
            f"✅ Bonne généralisation : seulement {gap * 100:.0f} pts d'écart "
            "train → test. Le réseau se comporte presque aussi bien sur des "
            "données nouvelles."
        )

    # Frontière avec les points de test (mal classés entourés).
    # On restaure le best epoch : le slider d'exploration a pu déplacer le modèle.
    if st.session_state.trainer.best_state_dict is not None:
        st.session_state.trainer.model.load_state_dict(
            st.session_state.trainer.best_state_dict
        )
    fig_gen = plot_generalization(
        st.session_state.trainer.model,
        st.session_state.trained_X,
        st.session_state.trained_y,
        st.session_state.test_X,
        st.session_state.test_y,
        mode=st.session_state.trained_mode,
    )
    st.pyplot(fig_gen)
    plt.close(fig_gen)


# ─────────────────────────────────────────────
# Évaluation robuste : validation croisée k-fold
# ─────────────────────────────────────────────
st.markdown("---")
with st.expander("Évaluation robuste — validation croisée (k-fold)"):
    st.caption(
        "Un seul découpage train/test peut être chanceux. La validation croisée "
        "k-fold entraîne le modèle sur **k découpages différents** et moyenne les "
        "résultats → estimation plus fiable de la généralisation "
        "(**accuracy moyenne ± écart-type**), surtout utile sur les petits "
        "datasets. ⚠️ Entraîne k modèles, donc plus long. Utilise l'architecture "
        "et les hyperparamètres configurés dans la sidebar."
    )
    kfold_k = st.slider("Nombre de folds (k)", 3, 10, 5)
    if st.button("Lancer l'évaluation k-fold"):
        prog = st.progress(0.0, text="Démarrage…")
        with st.spinner("Validation croisée en cours…"):
            accs = run_kfold(X, y, kfold_k, n_epochs, batch_size, progress=prog)
        prog.empty()

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Accuracy moyenne", f"{accs.mean() * 100:.1f} %")
        with c2:
            st.metric("Écart-type", f"± {accs.std() * 100:.1f} pts")
        with c3:
            st.metric("Min – Max", f"{accs.min() * 100:.0f} – {accs.max() * 100:.0f} %")

        st.caption(
            f"Accuracy par fold : {', '.join(f'{a * 100:.0f}%' for a in accs)}. "
            "Un écart-type faible = performance stable d'un découpage à l'autre "
            "(le modèle ne dépend pas d'un split chanceux)."
        )

        # Graphe de la dispersion des folds
        fig_kf = plot_kfold(accs)
        st.pyplot(fig_kf)
        plt.close(fig_kf)


# ─────────────────────────────────────────────
# Stabilité de la généralisation : repeated holdout (variance du découpage)
# ─────────────────────────────────────────────
st.markdown("---")
with st.expander("Stabilité de l'estimation — répéter le découpage (repeated holdout)"):
    st.caption(
        "Un **seul** découpage train/test peut être chanceux ou malchanceux. On "
        "re-tire ici le découpage **N fois** (stratifié), on ré-entraîne, et on "
        "regarde la **distribution** de l'accuracy de test. L'init des poids est "
        "fixe : la dispersion observée vient donc **uniquement du découpage**. "
        "💡 Augmente le **nombre de points** (sidebar) puis relance → l'écart-type "
        "diminue (la variance d'une proportion ∝ 1/√n_test). ⚠️ Entraîne N modèles."
    )
    n_holdout = st.slider("Nombre de découpages (N)", 5, 50, 20)
    if st.button("Évaluer la stabilité"):
        prog = st.progress(0.0, text="Démarrage…")
        with st.spinner("Repeated holdout en cours…"):
            te, tr = run_repeated_holdout(X, y, n_holdout, n_epochs, batch_size, test_ratio, progress=prog)
        prog.empty()

        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy test (moyenne)", f"{te.mean() * 100:.1f} %")
        c2.metric("Écart-type (instabilité)", f"± {te.std() * 100:.1f} pts")
        c3.metric("Écart train → test (moyen)", f"{(tr.mean() - te.mean()) * 100:+.1f} pts")

        spread = te.std() * 100
        if spread < 1.5:
            st.success(
                f"✅ Estimation **stable** : un seul découpage donne ≈ ±{spread:.1f} pts. "
                "La généralisation ne dépend quasiment pas du tirage."
            )
        elif spread < 4:
            st.info(
                f"ℹ️ Instabilité modérée : un seul split varie de ≈ ±{spread:.1f} pts. "
                "Moyenner plusieurs découpages (ou ajouter des points) fiabilise le chiffre."
            )
        else:
            st.warning(
                f"⚠️ Estimation **instable** : ±{spread:.1f} pts d'un découpage à l'autre → "
                "un seul split est peu fiable ici. Pistes : plus de points, ou moyenner N découpages."
            )

        st.caption(
            f"Accuracy test par découpage : {', '.join(f'{a * 100:.0f}%' for a in te)}."
        )
        fig_h = plot_holdout_distribution(te)
        st.pyplot(fig_h)
        plt.close(fig_h)


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
    cm = report.confusion
    K = cm.shape[0]
    multi_eval = K > 2          # affichage multi-classes (matrice K×K + table)

    # ─── Métriques principales (en multi : moyennes « macro » sur les K classes) ───
    suffix = " (macro)" if multi_eval else ""
    cols = st.columns(4)
    with cols[0]:
        st.metric("Accuracy",  f"{report.accuracy  * 100:.2f} %")
    with cols[1]:
        st.metric("Precision" + suffix, f"{report.precision * 100:.2f} %")
    with cols[2]:
        st.metric("Recall" + suffix,    f"{report.recall    * 100:.2f} %")
    with cols[3]:
        st.metric("F1-score" + suffix,  f"{report.f1        * 100:.2f} %")

    st.caption(f"Évalué sur **{report.n_samples} échantillons** du test set.")
    if multi_eval:
        st.caption(
            "En multi-classes, precision / recall / F1 sont des moyennes **macro** : "
            "on calcule la métrique pour chaque classe, puis on moyenne (chaque "
            "classe pèse autant, quel que soit son effectif)."
        )

    # ─── Matrice de confusion K×K ───
    cm_col, info_col = st.columns([2, 1])

    with cm_col:
        st.markdown("**Matrice de confusion**")
        fig_cm = plot_confusion_matrix(cm)
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    with info_col:
        st.markdown("**Lecture rapide**")
        if not multi_eval:
            # ─── Binaire : lecture TN / FP / FN / TP (comportement historique) ───
            tn, fp, fn, tp = cm.ravel()
            st.markdown(
                f"- **Vrais négatifs (TN)** : {tn} → classe 0 bien prédite\n"
                f"- **Faux positifs (FP)** : {fp} → classe 0 prédite à tort en classe 1\n"
                f"- **Faux négatifs (FN)** : {fn} → classe 1 ratée\n"
                f"- **Vrais positifs (TP)** : {tp} → classe 1 bien prédite"
            )
        else:
            # ─── Multi : diagonale = correct, hors-diagonale = confusions ───
            st.markdown(
                "- **Diagonale** = points bien classés.\n"
                "- **Hors-diagonale** = erreurs (ligne = vraie classe, "
                "colonne = classe prédite)."
            )
            off = cm.copy()
            np.fill_diagonal(off, 0)
            if off.max() > 0:
                i, j = np.unravel_index(int(off.argmax()), off.shape)
                st.markdown(
                    f"- Confusion la plus fréquente : **{int(off[i, j])} points** "
                    f"de la classe **{i}** prédits en classe **{j}**."
                )
            else:
                st.success("Aucune confusion : tout est sur la diagonale.")

    # ─── Métriques par classe (multi-classes) ───
    # Dérivées directement de la matrice K×K : la moyenne de chaque colonne
    # redonne la valeur « macro » affichée en haut.
    if multi_eval:
        st.markdown("**Métriques par classe**")
        st.table(per_class_table(cm))
        st.caption(
            "**Précision** d'une classe : parmi les points *prédits* dans cette "
            "classe, combien sont corrects (lecture en colonne). **Rappel** : parmi "
            "les points *réellement* de cette classe, combien sont retrouvés (lecture "
            "en ligne). **Support** : nombre de points de test de la classe. La "
            "moyenne de chaque colonne redonne les valeurs « macro » ci-dessus."
        )


# ─────────────────────────────────────────────
# Robustesse à la rotation des données (« Pb de rotation ? » — Barthe)
# ─────────────────────────────────────────────
if (st.session_state.test_report is not None
        and st.session_state.trainer is not None
        and st.session_state.test_X is not None):
    st.markdown("---")
    st.subheader("Robustesse à la rotation des données")
    st.caption(
        "On fait tourner le **test set** autour de l'origine (0 → 180°) et on ré-évalue "
        "le modèle **déjà entraîné**. Comme nos features sont les **coordonnées absolues** "
        "(x₁, x₂), le modèle n'est en général **pas invariant par rotation** : l'accuracy "
        "chute quand on tourne les données. Exception : un problème déjà invariant par "
        "rotation (ex. cercles concentriques, où la classe = le rayon)."
    )
    fig_rot, accs_rot = plot_rotation_robustness(
        st.session_state.trainer.model, st.session_state.test_X, st.session_state.test_y
    )
    c_rot1, c_rot2 = st.columns([2, 1])
    with c_rot1:
        st.pyplot(fig_rot)
        plt.close(fig_rot)
    with c_rot2:
        drop = accs_rot[0] - min(accs_rot)
        st.metric("Sans rotation", f"{accs_rot[0]:.0f} %")
        st.metric("Pire cas (rotation)", f"{min(accs_rot):.0f} %", delta=f"-{drop:.0f} pts")
        if drop < 10:
            st.success("Quasi invariant : ce problème ne dépend pas de l'orientation.")
        else:
            st.warning(
                "Pas invariant : l'accuracy chute → le modèle a appris une orientation "
                "précise (features = coordonnées absolues)."
            )
    st.caption(
        "**Pour Barthe** : rendre un modèle invariant par rotation demanderait des "
        "**features invariantes** (distances, rayon, angles relatifs…), pas seulement un "
        "graphe k-NN invariant — sur nos datasets, ce sont les coordonnées qui dominent."
    )
