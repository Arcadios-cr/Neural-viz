import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from models.mlp import MLP
from data.datasets import make_gaussians, to_dataloader


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
# Boucle d'entraînement
# ─────────────────────────────────────────────
def train(model, loader, n_epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    losses = []

    model.train()  # important pour BatchNorm (stats du batch en cours)
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(loader))

    return losses


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


def plot_loss_curve(losses):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(losses, color="#4CAF50", linewidth=2)
    ax.set_xlabel("Époque")
    ax.set_ylabel("Loss (BCE)")
    ax.set_title("Courbe d'apprentissage")
    ax.grid(True, alpha=0.3)
    return fig


# ─────────────────────────────────────────────
# Bouton d'entraînement
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
    if st.button("Entraîner le réseau", type="primary"):
        with st.spinner("Entraînement en cours..."):
            losses = train(model, loader, n_epochs, learning_rate)
        st.success(f"Entraînement terminé — loss finale : {losses[-1]:.4f}")

        mode = "logits" if viz_mode.startswith("Logits") else "probas"
        fig_boundary = plot_decision_boundary(model, X, y, mode=mode)
        st.pyplot(fig_boundary)
        plt.close(fig_boundary)

        st.subheader("Courbe d'apprentissage")
        fig_loss = plot_loss_curve(losses)
        st.pyplot(fig_loss)
        plt.close(fig_loss)
    else:
        st.info("Configure les paramètres dans la sidebar, puis clique sur **Entraîner le réseau**.")
