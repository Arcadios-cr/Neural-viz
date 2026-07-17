"""
Tests de non-régression de bout en bout — sans navigateur.

Le script pilote la VRAIE application (`app.py`) via `streamlit.testing.v1.AppTest`
et vérifie qu'aucune manipulation de base ne lève d'exception :

  1. chargement de TOUS les datasets du sélecteur (mode MLP) ;
  2. entraînement MLP, en binaire puis en multi-classes (Blobs, K = 3) ;
  3. entraînement du réseau de graphe pour chaque agrégation (moyenne, max,
     somme, attention) sur graphe k-NN, puis moyenne et somme sur graphe par
     rayon (dataset « Density découplée ») ;
  4. features géométriques du voisinage activées : entraînement + régions de
     décision (grille augmentée) ;
  5. SBM (graphe fourni) en multi-classes (K = 3) ;
  6. étude densité intégrée (5 modèles, 3 seeds).

Les études lentes (sur-lissage, balayage d'homophilie) ne sont pas couvertes :
elles multiplient les entraînements et rendraient le test trop long.

Lancer depuis n'importe où :  python tests/smoke_test.py
(~3 à 5 minutes ; sortie sur stderr, une ligne par vérification ;
code de retour 0 si tout est vert, 1 sinon.)
"""

import os
import sys

# Toujours s'exécuter depuis la racine du projet (imports + AppTest("app.py")).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from streamlit.testing.v1 import AppTest

TIMEOUT = 300
RESULTS = []


def log(msg):
    # stdout est capturé par Streamlit pendant AppTest.run() -> on écrit sur stderr
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def find(widgets, label):
    for w in widgets:
        if w.label == label:
            return w
    raise KeyError(f"widget introuvable : {label}")


def find_prefix(widgets, prefix):
    for w in widgets:
        if str(w.label).startswith(prefix):
            return w
    raise KeyError(f"widget introuvable : {prefix}…")


def run_checked(at, step):
    at.run()
    assert not at.exception, f"{step} : {[str(e.value) for e in at.exception]}"


def check(name, fn):
    try:
        detail = fn() or ""
        RESULTS.append((name, True, detail))
        log(f"  OK   {name}{'  — ' + detail if detail else ''}")
    except Exception as e:
        RESULTS.append((name, False, str(e)))
        log(f"  FAIL {name} : {e}")


def fresh():
    at = AppTest.from_file("app.py", default_timeout=TIMEOUT)
    run_checked(at, "chargement")
    return at


def to_graph_mode(at):
    radio = find(at.sidebar.radio, "Type de modèle")
    radio.set_value([o for o in radio.options if str(o).startswith("Réseau")][0])
    run_checked(at, "passage en mode graphe")


def train_graph(at, step):
    [b for b in at.button if b.label == "Entraîner le réseau de graphe"][0].click()
    run_checked(at, step)
    return at.session_state["gcn"]


# ─── 1. Tous les datasets se chargent (mode MLP) ───
def all_datasets():
    at = fresh()
    sel = find(at.sidebar.selectbox, "Type de dataset")
    for name in list(sel.options):
        sel.set_value(name)
        run_checked(at, f"dataset {name}")
    return f"{len(sel.options)} datasets"


# ─── 2. Entraînement MLP, binaire puis multi-classes ───
def mlp_binary():
    at = fresh()
    [b for b in at.button if b.label == "Entraîner le réseau"][0].click()
    run_checked(at, "entraînement MLP binaire")


def mlp_multiclass():
    at = fresh()
    find(at.sidebar.selectbox, "Type de dataset").set_value("Blobs")
    find(at.sidebar.slider, "Nombre de classes (K)").set_value(3)
    run_checked(at, "Blobs K=3")
    [b for b in at.button if b.label == "Entraîner le réseau"][0].click()
    run_checked(at, "entraînement MLP multi-classes")


# ─── 3. Réseau de graphe : agrégations × constructions ───
def graph_aggregations():
    at = fresh()
    find(at.sidebar.selectbox, "Type de dataset").set_value("Density découplée")
    to_graph_mode(at)
    accs = []
    for agg in ("moyenne (GCN)", "max (GraphSAGE)", "somme (GIN)", "attention (GAT)"):
        find(at.sidebar.selectbox, "Agrégation des voisins").set_value(agg)
        run_checked(at, f"agrégation {agg}")
        g = train_graph(at, f"entraînement {agg} / k-NN")
        assert 0.0 <= g["acc_test"] <= 1.0
        accs.append(f"{agg.split()[0]}={g['acc_test'] * 100:.0f}%")
    find(at.radio, "Construction du graphe").set_value("rayon (r)")
    run_checked(at, "construction rayon")
    for agg in ("moyenne (GCN)", "somme (GIN)"):
        find(at.sidebar.selectbox, "Agrégation des voisins").set_value(agg)
        run_checked(at, f"agrégation {agg} (rayon)")
        g = train_graph(at, f"entraînement {agg} / rayon")
        accs.append(f"{agg.split()[0]}/rayon={g['acc_test'] * 100:.0f}%")
    return "  ".join(accs)


# ─── 4. Features géométriques + régions de décision ───
def geo_features():
    at = fresh()
    find(at.sidebar.selectbox, "Type de dataset").set_value("Density découplée")
    to_graph_mode(at)
    find_prefix(at.checkbox, "Features de voisinage").set_value(True)
    run_checked(at, "features géométriques ON")
    g = train_graph(at, "entraînement avec features")
    assert g["Xin"].shape[1] == 4, f"attendu 4 features, obtenu {g['Xin'].shape[1]}"
    find_prefix(at.checkbox, "Calculer les régions").set_value(True)
    run_checked(at, "régions de décision (grille augmentée)")
    return f"acc={g['acc_test'] * 100:.0f}%"


# ─── 5. SBM multi-classes (graphe fourni) ───
def sbm_multiclass():
    at = fresh()
    find(at.sidebar.selectbox, "Type de dataset").set_value("SBM (graphe communautaire)")
    find(at.sidebar.slider, "Nombre de classes (K)").set_value(3)
    to_graph_mode(at)
    g = train_graph(at, "entraînement SBM K=3")
    return f"acc={g['acc_test'] * 100:.0f}%"


# ─── 6. Étude densité intégrée ───
def density_study():
    at = fresh()
    find(at.sidebar.selectbox, "Type de dataset").set_value("Density")
    to_graph_mode(at)
    find(at.radio, "Construction du graphe").set_value("rayon (r)")
    run_checked(at, "construction rayon")
    find_prefix(at.checkbox, "Lancer l'étude densité").set_value(True)
    run_checked(at, "étude densité (5 modèles, 3 seeds)")


if __name__ == "__main__":
    log("=== Tests de non-régression Neural-Viz ===")
    check("1. chargement de tous les datasets (mode MLP)", all_datasets)
    check("2a. entraînement MLP binaire", mlp_binary)
    check("2b. entraînement MLP multi-classes (Blobs K=3)", mlp_multiclass)
    check("3. réseau de graphe : 4 agrégations (k-NN) + moyenne/somme (rayon)",
          graph_aggregations)
    check("4. features géométriques + régions de décision", geo_features)
    check("5. SBM multi-classes (graphe fourni)", sbm_multiclass)
    check("6. étude densité intégrée", density_study)

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    log(f"=== {len(RESULTS) - n_fail}/{len(RESULTS)} vérifications OK ===")
    sys.exit(1 if n_fail else 0)
