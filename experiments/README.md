# experiments/ — études d'hyperparamètres

Scripts d'**étude de la généralisation** par balayage d'hyperparamètres.

Contrairement à l'app Streamlit (qui entraîne **un** modèle à la fois, de façon
interactive), ces scripts lancent **plusieurs** entraînements en faisant varier
un paramètre, puis tracent la courbe correspondante. C'est la façon standard de
mener une étude d'hyperparamètres (on automatise le balayage au lieu de cliquer
plusieurs fois).

Chaque script :
- **fixe les graines aléatoires** (résultats reproductibles d'une exécution à l'autre) ;
- mesure la généralisation sur un **test set fixe et indépendant** (1000 points),
  pour comparer proprement les configurations ;
- enregistre ses figures dans `experiments/figures/`.

## Scripts

| Script | Étudie | Figure(s) |
|---|---|---|
| `density_study.py` | effet de la **densité** de points (courbe d'apprentissage) | `density_learning_curve.png` |
| `generalization_study.py` | **capacité** (largeur), **batch size**, **régularisation** (dropout vs weight decay) | `capacity.png`, `batch_size.png`, `regularization.png` |

## Lancer

```bash
# depuis la racine du projet, environnement activé
python experiments/density_study.py
python experiments/generalization_study.py
```

## Note

Les figures qui montrent **un seul** entraînement (frontière de décision, espace
latent, k-fold, gradients…) sont, elles, produites **directement dans l'app**
(elles sont reproductibles par les contrôles de la sidebar). Ces scripts ne
couvrent que les **balayages** multi-entraînements, qui ne sont pas faisables en
un clic dans l'app.
