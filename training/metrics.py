"""
Module d'évaluation : métriques de classification binaire calculées sur
un dataset de test (jamais vu pendant l'entraînement).

On s'appuie sur sklearn.metrics qui est déjà une dépendance du projet
(utilisée par data/datasets.py).
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
    precision   : VP / (VP + FP) — fiabilité d'une prédiction positive
    recall      : VP / (VP + FN) — capacité à retrouver les positifs
    f1          : moyenne harmonique de precision et recall
    confusion   : matrice 2x2 (lignes = vraies classes, colonnes = prédictions)
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
    Évalue un modèle PyTorch de classification binaire sur un DataLoader.

    Le modèle est supposé renvoyer un logit unique par échantillon
    (sortie de la dernière couche linéaire, AVANT sigmoid).
    La prédiction est faite via :    pred = 1 si logit > threshold, sinon 0
    (par défaut threshold=0, ce qui correspond à proba=0.5 après sigmoid).

    Paramètres
    ----------
    model : nn.Module
        Le modèle à évaluer (sera basculé en mode eval).
    loader : DataLoader
        Le DataLoader sur lequel évaluer (typiquement le test_loader).
    threshold : float
        Seuil de décision sur les logits. Par défaut 0.0
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
