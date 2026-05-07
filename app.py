import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from models.mlp import MLP
from data.datasets import make_gaussians, to_dataloader
from training.trainer import Trainer


# ─────────────────────────────────────────────
# Configuration de la page
# ─────────────────────────────────────────────
st.set_page_config(page_title="Neural-Viz", layout="wide")
st.title("Visualisation d'un réseau de neurones MLP")


# ─────────────────────────────────────────────
# Sidebar — paramètres
# ─────────────────────────────────────────────
st.sidebar.header("Architecture du réseau")

n_hidden_layers = st.sidebar.slider("Nombre de couches cachées", 1, 4, 1)
neurons_per_layer = st.sidebar.slider("Neurones par couche cachée", 2, 64, 8)
activation = st.sidebar.selectbox("Fonction d'activation", ["relu", "tanh", "sigmoid"])
use_batchnorm = st.sidebar.checkbox("Utiliser BatchNorm", value=False)

st.sidebar.header("Visualisation")
viz_mode = st.sidebar.radio(
    "Affichage de la sortie du réseau",
    ["Logits (sortie brute)", "Probabilités (sigmoid)"],
    index=0,
)
live_training = st.sidebar.checkbox("Entraînement en temps réel", value=True)
boundary_refresh = st.sidebar.slider(
    "Rafraîchir la frontière toutes les N époques",
    1, 50, 10,
    help="Plus petit = plus fluide mais plus lent. Plus grand = plus rapide.",
)

st.sidebar.header("Entraînement")
n_epochs = st.sidebar.slider("Nombre d'époques", 10, 500, 100)
learning_rate = st.sidebar.select_slider(
    "Taux d'apprentissage",
    options=[0.001, 0.005, 0.01, 0.05, 0.1],
    value=0.01,
)
batch_size = st.sidebar.slider("Batch size", 8, 64, 32)

st.sidebar.header("Dataset")
n_samples = st.sidebar.slider("Nombre de points", 100, 500, 200)
std = st.sidebar.slider("Écart-type des gaussiennes", 0.1, 1.5, 0.5)
seed = st.sidebar.number_input("Seed", value=42, step=1)


# ─────────────────────────────────────────────
# Génération du dataset
# ─────────────────────────────────────────────
X, y = make_gaussians(n_samples=n_samples, std=std, seed=int(seed))
loader = to_dataloader(X, y, batch_size=batch_size)


# ─────────────────────────────────────────────
# Construction du modèle
# ─────────────────────────────────────────────
hidden_layers = [neurons_per_layer] * n_hidden_layers
model = MLP(
    input_dim=2,
    hidden_layers=hidden_layers,
    output_dim=1,
    activation=activation,
    use_batchnorm=use_batchnorm,
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


# ─────────────────────────────────────────────
# Layout principal
# ─────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Données d'entraînement")
    fig_data, ax_data = plt.subplots(figsize=(7, 6))
    colors = ["#2196F3", "#F44336"]
    for cls in [0, 1]:
        mask = y == cls
        ax_data.scatter(X[mask, 0], X[mask, 1], c=colors[cls],
                        edgecolors="white", linewidths=0.5, s=40, label=f"Classe {cls}")
    ax_data.set_xlabel("x₁")
    ax_data.set_ylabel("x₂")
    ax_data.set_title("Dataset — deux gaussiennes")
    ax_data.legend()
    st.pyplot(fig_data)
    plt.close(fig_data)

with col2:
    st.subheader("Frontière de décision")

    boundary_placeholder = st.empty()
    status_placeholder = st.empty()

    if st.button("Entraîner le réseau", type="primary"):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion)

        mode = "logits" if viz_mode.startswith("Logits") else "probas"

        # Mise en place de la zone "courbe d'apprentissage" en bas
        st.subheader("Courbe d'apprentissage")
        loss_placeholder = st.empty()

        def update_callback(epoch: int, loss: float, current_model):
            """Callback appelé à chaque fin d'époque pendant l'entraînement."""
            status_placeholder.info(f"Époque {epoch + 1} / {n_epochs} — loss = {loss:.4f}")

            # Frontière de décision : seulement toutes les N époques (lourd)
            if (epoch + 1) % boundary_refresh == 0 or epoch == n_epochs - 1:
                # Bascule en eval pour la viz (BatchNorm utilise les running stats)
                current_model.eval()
                fig = plot_decision_boundary(current_model, X, y, mode=mode)
                boundary_placeholder.pyplot(fig)
                plt.close(fig)
                current_model.train()  # retour en mode train pour la suite

            # Courbe de loss : à chaque époque (léger)
            loss_placeholder.line_chart(trainer.history["loss"])

        callback = update_callback if live_training else None
        with st.spinner("Entraînement en cours..."):
            losses = trainer.train(loader, n_epochs=n_epochs, on_epoch_end=callback)

        # Rendu final (au cas où le live est désactivé ou si la dernière époque
        # ne tombait pas sur un rafraîchissement)
        model.eval()
        fig_boundary = plot_decision_boundary(model, X, y, mode=mode)
        boundary_placeholder.pyplot(fig_boundary)
        plt.close(fig_boundary)

        loss_placeholder.line_chart(trainer.history["loss"])
        status_placeholder.success(f"Entraînement terminé — loss finale : {losses[-1]:.4f}")

        # Persistance pour le slider d'exploration de l'historique
        st.session_state.trainer = trainer
        st.session_state.trained_X = X
        st.session_state.trained_y = y
        st.session_state.trained_mode = mode
    else:
        # S'il n'y a pas eu d'entraînement encore, on affiche le message d'invite
        if st.session_state.trainer is None:
            boundary_placeholder.info(
                "Configure les paramètres dans la sidebar, puis clique sur **Entraîner le réseau**."
            )


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
            label="Loss à cette époque",
            value=f"{trainer.history['loss'][selected_epoch - 1]:.4f}",
        )
        st.metric(
            label="Nombre total d'époques",
            value=n_snapshots,
        )
        if st.button("Réinitialiser l'historique"):
            st.session_state.trainer = None
            st.session_state.trained_X = None
            st.session_state.trained_y = None
            st.session_state.trained_mode = None
            st.rerun()
