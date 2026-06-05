"""
Helpers communs aux scripts d'expérimentation.

Ces scripts étudient l'effet des hyperparamètres sur la généralisation en
lançant plusieurs entraînements (balayages). Pour une mesure fiable, la
performance de test est toujours évaluée sur un *test set fixe et indépendant*.
"""

import os
import sys

# Permet d'importer les modules du projet (models, data, training) quel que
# soit le répertoire depuis lequel on lance le script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.disable(logging.WARNING)

import torch
import torch.nn as nn

from models.mlp import MLP
from data.datasets import get_dataset, to_dataloader
from training.trainer import Trainer
from training.metrics import evaluate

# Graine par défaut pour l'initialisation des poids (reproductibilité).
SEED = 0


def fixed_test_loader(dataset="Overlap", n=1000, noise=0.5, seed=999, batch=64):
    """
    Construit un test set FIXE et indépendant du train.

    Mesurer la performance de test toujours sur ce même grand set permet de
    comparer proprement différentes tailles/configs de train, sans le bruit
    d'un petit test qui changerait à chaque expérience.
    """
    X_te, y_te = get_dataset(dataset, n_samples=n, noise=noise, seed=seed)
    return to_dataloader(X_te, y_te, batch, False)


def train_and_eval(
    X_tr, y_tr, test_loader, layers, *,
    dropout=0.0, weight_decay=0.0, batch_size=32,
    lr=0.01, n_epochs=200, activation="relu", seed=SEED,
):
    """
    Entraîne un MLP (initialisation reproductible) et renvoie (acc_train, acc_test).

    L'init des poids et le mélange des batches sont fixés par `seed` → résultat
    identique à chaque exécution.
    """
    torch.manual_seed(seed)                       # init des poids
    model = MLP(2, layers, 1, activation=activation, dropout_rate=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    trainer = Trainer(model, optimizer, nn.BCEWithLogitsLoss())

    torch.manual_seed(seed)                       # shuffle + dropout
    trainer.train(
        to_dataloader(X_tr, y_tr, batch_size, True),
        n_epochs=n_epochs, save_snapshots=False, restore_best=False,
    )

    acc_train = evaluate(model, to_dataloader(X_tr, y_tr, 64, False)).accuracy
    acc_test = evaluate(model, test_loader).accuracy
    return acc_train, acc_test
