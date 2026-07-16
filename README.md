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

## Lancer l'application

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Structure du dépôt

| Dossier | Contenu |
|---|---|
| `app.py` | application Streamlit (modes MLP et réseau de graphe) |
| `models/` | MLP, MLP bottleneck, couche de graphe unifiée (`gcn.py`) |
| `data/` | générateurs de datasets, graphes (k-NN, rayon, stats d'arêtes), SBM |
| `training/` | boucle d'entraînement (early stopping, snapshots), métriques |
| `experiments/` | études reproductibles par balayage (voir `experiments/README.md`) |

## Encadrant

Loïc Barthe — IRIT, Université Paul Sabatier
