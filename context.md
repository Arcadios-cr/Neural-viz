# Contexte TER — Viz-neural

## Sujet
Visualisation d'un réseau de neurones — produire un logiciel Python permettant d'illustrer de façon claire et intelligible les transformations effectuées par un réseau de neurones pour arriver à un résultat ou une classification.

## Encadrant
**Loïc Barthe** — IRIT  
Réunion hebdomadaire chaque **lundi à 9h à l'IRIT** (première réunion : lundi 4 mai 2026)

## Contraintes de travail imposées par Barthe
- Réunion hebdomadaire en physique à l'IRIT (lundi 9h)
- Logbook quotidien (document partagé) : ce qui a été fait, les expérimentations, les résultats
- Suivi de versions Git avec sauvegardes régulières

## Stack technique
- **Langage** : Python
- **Réseau de neurones** : PyTorch
- **Interface / visualisation** : Streamlit
- **Visualisation graphique** : Matplotlib / Plotly

## Architecture du projet
```
neural-viz/
├── app.py                        # Point d'entrée Streamlit
├── models/
│   └── mlp.py                    # Modèle MLP PyTorch (configurable)
├── training/
│   └── trainer.py                # Boucle d'entraînement + callbacks
├── data/
│   └── datasets.py               # Générateurs de datasets (gaussiennes, XOR, spirales...)
├── visualization/
│   ├── decision_boundary.py      # Frontière de décision R²→R
│   ├── activations.py            # Activations couche par couche
│   └── weights.py                # Poids / gradients
├── utils/
│   └── hooks.py                  # Forward hooks PyTorch pour capturer les activations
├── requirements.txt
├── logbook.md                    # (optionnel, si on veut une trace locale)
└── context.md                    # Ce fichier
```

## Organisation des sessions de travail
- On travaille **par semaine** : tout le code + logbook + commits sont préparés en une seule session
- À la fin de chaque session, je fournis :
  1. **Les entrées logbook** pour chaque jour de la semaine (lundi → vendredi) → à copier-coller dans le Google Doc
  2. **Les commits Git** à effectuer pour le code produit
- **Règle stricte** : on ne fait QUE ce qui est prévu pour la semaine en cours, on n'empiète pas sur les semaines suivantes
- **Git** : commit dès qu'un élément est terminé, push au minimum une fois par semaine (avant la réunion du lundi)

## Planning (~12 semaines)

### Semaine 1 — 27 avril au 2 mai 2026
**Objectif : fondations du projet**
- Mise en place du logbook et du document partagé
- Setup Git + environnement Python (requirements.txt)
- Définition de l'architecture logicielle et organisation des fichiers
- MLP configurable : 2 entrées, 1 sortie, nombre de neurones et activation variables
- Premier dataset : deux classes dans R² (deux gaussiennes)
- Première visualisation Streamlit : scatter plot des données + frontière de décision R²→R

### Semaines 2–4 — mai 2026
**Objectif : entraînement interactif + visualisation des activations**
- Entraînement du MLP avec visualisation en temps réel (ou pas à pas)
- Visualisation des activations couche par couche (forward hooks PyTorch)
- Interface interactive : changer l'architecture, les données, les hyperparamètres
- Courbe de loss en temps réel

### Semaines 5–7 — juin 2026
**Objectif : architectures plus complexes + autres datasets**
- Extension à plusieurs couches cachées
- Visualisation des poids, gradients
- Autres datasets : XOR, spirales, cercles concentriques
- Amélioration de l'interactivité

### Semaines 8–10 — juillet 2026
**Objectif : classification multi-classes + polish**
- Cas de classification multi-classes
- Amélioration de l'interface et de l'ergonomie
- Documentation du code

### Semaines 11–12 — fin juillet / début août 2026
**Objectif : finalisation**
- Tests, polish final
- Rédaction du rapport de TER
- Préparation de la soutenance
