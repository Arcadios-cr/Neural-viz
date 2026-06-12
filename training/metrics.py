"""
Module d'évaluation : métriques de classification (binaire ET multi-classes)
calculées sur un dataset de test (jamais vu pendant l'entraînement).

Le mode est détecté automatiquement d'après la forme des logits du modèle :
1 sortie → binaire (seuil) ; K sorties → multi-classes (argmax, moyennes
« macro », matrice K×K). On s'appuie sur sklearn.metrics, déjà une dépendance
du projet (utilisée par data/datasets.py).
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from torch.utils.data import DataLoader


@dataclass
class ClassificationReport:
    """
    Résultat de l'évaluation : regroupe toutes les métriques calculées
    sur un set d'évaluation.

    Champs
    ------
    accuracy    : taux global de bonnes prédictions
    precision   : binaire = VP/(VP+FP) ; multi = moyenne macro des précisions
    recall      : binaire = VP/(VP+FN) ; multi = moyenne macro des rappels
    f1          : moyenne harmonique de precision et recall (macro en multi)
    confusion   : matrice K×K (lignes = vraies classes, colonnes = prédictions ;
                  2×2 en binaire)
    n_samples   : nombre total d'échantillons évalués
    """
    accuracy:   float
    precision:  float
    recall:     float
    f1:         float
    confusion:  np.ndarray
    n_samples:  int

    def as_dict(self) -> dict[str, float | np.ndarray | int]:
        return {
            "accuracy":   self.accuracy,
            "precision":  self.precision,
            "recall":     self.recall,
            "f1":         self.f1,
            "confusion":  self.confusion,
            "n_samples":  self.n_samples,
        }


# ─────────────────────────────────────────────
# Évaluation
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    threshold: float = 0.0,
) -> ClassificationReport:
    """
    Évalue un modèle PyTorch de classification sur un DataLoader.

    Détecte automatiquement le mode d'après la sortie du modèle :
      - 1 logit par échantillon  → binaire : pred = 1 si logit > threshold
        (threshold=0 ⇔ proba 0.5 après sigmoid) ;
      - K logits par échantillon → multi-classes : pred = argmax, métriques
        moyennées « macro » et matrice de confusion K×K.

    Paramètres
    ----------
    model : nn.Module
        Le modèle à évaluer (sera basculé en mode eval).
    loader : DataLoader
        Le DataLoader sur lequel évaluer (typiquement le test_loader).
    threshold : float
        Seuil de décision sur les logits (binaire uniquement). Par défaut 0.0
        (équivalent à p(classe 1) > 0.5).

    Retourne
    --------
    ClassificationReport avec toutes les métriques.
    """
    model.eval()

    all_logits = []
    all_targets = []
    for X_batch, y_batch in loader:
        all_logits.append(model(X_batch))
        all_targets.append(y_batch)

    logits = torch.cat(all_logits, dim=0)        # (N, 1) binaire | (N, K) multi
    targets = torch.cat(all_targets, dim=0)
    y_true = targets.long().view(-1).cpu().numpy()

    if logits.shape[1] > 1:
        # ─── Multi-classes : prédiction = argmax, métriques moyennées (macro) ───
        K = logits.shape[1]
        y_pred = logits.argmax(dim=1).cpu().numpy()
        average, labels = "macro", list(range(K))
    else:
        # ─── Binaire : seuil sur le logit unique ───
        y_pred = (logits.view(-1) > threshold).long().cpu().numpy()
        average, labels = "binary", [0, 1]

    # zero_division=0 : si une classe est absente des prédictions, on renvoie 0
    # au lieu de lever un warning
    return ClassificationReport(
        accuracy   = accuracy_score(y_true, y_pred),
        precision  = precision_score(y_true, y_pred, average=average, zero_division=0),
        recall     = recall_score(y_true, y_pred, average=average, zero_division=0),
        f1         = f1_score(y_true, y_pred, average=average, zero_division=0),
        confusion  = confusion_matrix(y_true, y_pred, labels=labels),
        n_samples  = len(y_true),
    )
