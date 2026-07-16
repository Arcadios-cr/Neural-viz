# Neural-Viz

Logiciel de visualisation de réseaux de neurones développé dans le cadre d'un TER à l'Université Paul Sabatier (IRIT).

## Objectif

Illustrer de façon claire et intelligible les transformations effectuées par un réseau de neurones pour arriver à une classification. Le projet couvre deux familles de modèles sur des nuages de points 2D :

- un **MLP** entièrement configurable, pour comprendre frontières de décision, capacité, régularisation et représentations internes ;
- des **réseaux de graphe** (GCN / GraphSAGE / GIN / GAT), pour poser la question centrale du projet : *quand un graphe aide-t-il un réseau de neurones ?*

## Fonctionnalités

### Mode MLP
- Architecture configurable : couches, neurones (contrôle séparé de la 1ère couche), activation, BatchNorm, dropout, bottleneck 1D/2D
- Une quinzaine de datasets 2D générés (gaussiennes, overlap, XOR, moons, circles, spirales, damier, îles, densité, structure…), binaires et multi-classes (K ≤ 5)
- Visualisations : frontière de décision, activations couche par couche, poids et gradients, espace latent, courbes de loss, généralisation (early stopping, k-fold), matrice de confusion et métriques par classe

### Mode réseau de graphe
- **Une seule architecture configurable** : couches de graphe + tête MLP, l'agrégation des voisins faisant l'architecture — moyenne (GCN), max (GraphSAGE), somme (GIN, la seule qui « compte » les voisins), attention (GAT, multi-têtes)
- Construction du graphe : **k-NN** (degré ~constant) ou **par rayon** (le degré reflète la densité locale) ; graphe communautaire **fourni** (SBM, homophilie réglable)
- **Features géométriques du voisinage** en option : longueur moyenne d'arête et verticalité — la géométrie que la binarisation du graphe jette
- Entraînement transductif, comparaison MLP vs réseau de graphe, régions de décision, espace latent, carte et inspecteur d'attention, champ réceptif
- Études intégrées : loi de l'homophilie (SBM), sur-lissage selon la profondeur (avec/sans BatchNorm), densité (MLP vs k-NN vs rayon vs somme vs degré en feature)

## Le résultat du projet en une phrase

L'information du voisinage n'aide un réseau de graphe que si elle est **absente des features** et **lisible par l'agrégation** — trois routes pour la rendre lisible : changer le graphe (rayon), changer l'agrégation (somme), ou extraire la géométrie des arêtes en features. Étude reproductible : `experiments/neighborhood_info_study.py` (3 scénarios × 7 modèles).

## Stack

**Python 3.10+** · **PyTorch** · **Streamlit** · **Matplotlib / Plotly** · **scikit-learn**

## Installation et lancement

```bash
# 1. Créer et activer un environnement virtuel (Python 3.10+)
python3 -m venv venv
source venv/bin/activate          # Windows : venv\Scripts\activate

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer l'application — s'ouvre sur http://localhost:8501
streamlit run app.py
```

Les études reproductibles se lancent depuis la racine du projet, par exemple :

```bash
python experiments/neighborhood_info_study.py   # figures dans experiments/figures/
```

## Structure du dépôt — qui fait quoi

### Racine

| Fichier | Rôle |
|---|---|
| `app.py` | L'application Streamlit complète : sidebar de configuration, mode MLP (frontière de décision, activations, poids/gradients, espace latent, généralisation) et mode réseau de graphe (construction du graphe, entraînement transductif, comparaisons MLP vs graphe, études intégrées). Toutes les figures interactives y sont produites. |
| `requirements.txt` | Dépendances Python (PyTorch, Streamlit, Matplotlib, Plotly, NumPy, scikit-learn). |
| `context.md` | Contexte et planning initial du TER. |
| `README.md` | Ce fichier. |

### `models/` — les réseaux

| Fichier | Rôle |
|---|---|
| `mlp.py` | MLP configurable (nombre de couches, neurones par couche, activation, BatchNorm, dropout) ; sortie de dimension 1 (binaire) ou K (multi-classes). |
| `mlp_bottleneck.py` | Variante avec goulot d'étranglement 1D/2D et méthode `encode()` : visualiser l'espace latent appris. |
| `gcn.py` | La couche de graphe **unifiée** `GCNLayer` — une seule couche, quatre agrégations : moyenne (GCN), max (GraphSAGE), somme (GIN), attention multi-têtes (GAT) — et le réseau `GCN` (couches de graphe + tête MLP par nœud). |

### `data/` — données et graphes

| Fichier | Rôle |
|---|---|
| `datasets.py` | Une quinzaine de générateurs de nuages 2D (gaussiennes, overlap, XOR, moons, circles, spirales, damier, îles, densité, densité découplée, structure…), le registre `DATASETS`, le split train/val/test et la conversion en DataLoaders. |
| `graphs.py` | Construction des graphes : k-NN (`build_knn`), par rayon (`build_radius`), normalisation de Kipf (`normalize_adj`), et les features géométriques d'arête (`knn_edge_stats` : longueur moyenne + verticalité du voisinage). |
| `sbm.py` | Le graphe communautaire SBM : features 2D faibles + graphe **fourni**, homophilie réglable à degré moyen constant. |

### `training/` — entraînement et métriques

| Fichier | Rôle |
|---|---|
| `trainer.py` | Boucle d'entraînement du MLP : early stopping, snapshots des poids par époque (rejeu du curseur d'époque dans l'app), historique des gradients. |
| `metrics.py` | Évaluation binaire et multi-classes : accuracy, précision/rappel/F1 par classe, matrice de confusion. |

### `utils/`

| Fichier | Rôle |
|---|---|
| `hooks.py` | Forward hooks PyTorch : capture des activations couche par couche pour la visualisation. |

### `experiments/` — études reproductibles (détail dans `experiments/README.md`)

| Fichier | Rôle |
|---|---|
| `_common.py` | Helpers partagés : graines fixées, test set fixe de 1000 points, entraînement standard. |
| `density_study.py` | Effet de la densité de points sur la généralisation (courbe d'apprentissage). |
| `generalization_study.py` | Trois balayages : capacité (largeur), batch size, régularisation (dropout vs weight decay). |
| `multi_dataset_study.py` | Les conclusions densité/capacité tiennent-elles sur tous les datasets ? |
| `multiclass_study.py` | Performances réelles du mode multi-classes (config adaptée, test set fixe). |
| `neighborhood_info_study.py` | **La synthèse du projet** : quand l'info du voisinage aide-t-elle ? 3 scénarios × 7 modèles, en transductif. |

## Encadrant

Loïc Barthe — IRIT, Université Paul Sabatier
