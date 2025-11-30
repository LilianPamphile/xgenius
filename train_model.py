# ========================= train_model.py — vSeg2_ADPoisson ==========================
# Objectif :
#   - prédire le total de buts
#   - baseline Poisson attaque/défense (non entraînée)
#   - modèle global HGB
#   - segmentation 2 classes (≤2 buts / >2 buts)
#   - modèles spécialisés low / high + fallback si incertitude
# ================================================================================

import os
import json
from decimal import Decimal
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import psycopg2

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
import joblib


# -------------------------- CONFIG --------------------------

OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)

DATABASE_URL = "postgresql://postgres:jDDqfaqpspVDBBwsqxuaiSDNXjTxjMmP@shortline.proxy.rlwy.net:36536/railway"
if not DATABASE_URL:
    raise ValueError("DATABASE_URL manquant.")


# -------------------------- UTILS ---------------------------

def np_rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metrics_block(y_true, y_pred) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": np_rmse(y_true, y_pred),
        "n": int(len(y_true)),
    }


def segment_from_y(y: float) -> int:
    """
    Segmentation simplifiée :
        0 -> match "low" (≤ 2 buts)
        1 -> match "high" (> 2 buts)
    """
    return 0 if y <= 2.0 else 1


# ----------------------- DATA LOADING -----------------------

conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()

query = """
    SELECT
        m.game_id,
        m.date::date AS date_match,
        m.competition,
        m.equipe_domicile,
        m.equipe_exterieur,

        sg1.moyenne_xg_dom AS xg_dom,
        sg1.tirs           AS tirs_dom,

        sg2.moyenne_xg_ext AS xg_ext,
        sg2.tirs           AS tirs_ext,

        s.buts_dom,
        s.buts_ext,
        s.buts_dom + s.buts_ext AS total_buts
    FROM matchs_v2 m
    JOIN stats_globales_v2 sg1
      ON m.equipe_domicile = sg1.equipe
     AND m.competition = sg1.competition
     AND m.saison = sg1.saison
    JOIN stats_globales_v2 sg2
      ON m.equipe_exterieur = sg2.equipe
     AND m.competition = sg2.competition
     AND m.saison = sg2.saison
    JOIN stats_matchs_v2 s
      ON m.game_id = s.game_id
    WHERE s.buts_dom IS NOT NULL
      AND s.buts_ext IS NOT NULL
"""
cur.execute(query)
rows = cur.fetchall()
cols = [col.name for col in cur.description]
conn.close()

df = pd.DataFrame(rows, columns=cols)

# Convertir Decimal → float si besoin
for c in df.columns:
    if df[c].notna().any() and isinstance(df[c].dropna().iloc[0], Decimal):
        df[c] = df[c].astype(float)

df["date_match"] = pd.to_datetime(df["date_match"])

# (Optionnel) filtrer sur les dernières saisons si tu veux
# df = df[df["date_match"] >= pd.to_datetime("2019-08-01")].reset_index(drop=True)


# ----------------------- FEATURE ENGINEERING -----------------------

df_hist = df[[
    "date_match", "competition",
    "equipe_domicile", "equipe_exterieur",
    "buts_dom", "buts_ext"
]].copy()
df_hist["total_buts"] = df_hist["buts_dom"] + df_hist["buts_ext"]


def forme_split(equipe: str, date_ref, role: str, n: int = 5):
    """
    Forme buts marqués / encaissés sur les n derniers matchs avant date_ref.
    role = "home" ou "away"
    """
    q = (df_hist["date_match"] < date_ref)

    if role == "home":
        q &= (df_hist["equipe_domicile"] == equipe)
    else:
        q &= (df_hist["equipe_exterieur"] == equipe)

    m = df_hist.loc[q].sort_values("date_match", ascending=False).head(n)
    if len(m) == 0:
        return 0.0, 0.0

    if role == "home":
        bm = m["buts_dom"].values
        be = m["buts_ext"].values
    else:
        bm = m["buts_ext"].values
        be = m["buts_dom"].values

    return float(np.mean(bm)), float(np.mean(be))


def league_avg_goals(competition: str, date_ref, window: int = 60) -> float:
    """
    Moyenne de buts par match dans la ligue sur les 'window' derniers matchs avant date_ref.
    """
    q = (df_hist["competition"] == competition) & (df_hist["date_match"] < date_ref)
    m = df_hist.loc[q].sort_values("date_match", ascending=False).head(window)
    if len(m) == 0:
        return 2.5
    return float(m["total_buts"].mean())


FEATURES: List[str] = [
    "forme_home_buts_marques",
    "forme_home_buts_encaisses",
    "forme_away_buts_marques",
    "forme_away_buts_encaisses",
    "attaque_home",
    "defense_home",
    "attaque_away",
    "defense_away",
    "xg_dom",
    "xg_ext",
    "tirs_dom",
    "tirs_ext",
    "league_avg_goals_60d",
    "xg_exp_total",
]

records: List[Dict[str, Any]] = []

for _, r in df.sort_values("date_match").iterrows():
    dom = r["equipe_domicile"]
    ext = r["equipe_exterieur"]
    dref = r["date_match"]
    comp = r["competition"]

    bm_home, be_home = forme_split(dom, dref, role="home")
    bm_away, be_away = forme_split(ext, dref, role="away")

    # Attaque/défense basées sur buts récents
    attaque_home = bm_home
    defense_home = be_home
    attaque_away = bm_away
    defense_away = be_away

    # "xg" attendu à partir de ces forces
    xg_exp_total = attaque_home * defense_away + attaque_away * defense_home

    league_avg = league_avg_goals(comp, dref, window=60)

    rec = {
        "game_id": r["game_id"],
        "date_match": dref,
        "competition": comp,
        "y_total": float(r["total_buts"]),

        # forme buts bruts
        "forme_home_buts_marques": bm_home,
        "forme_home_buts_encaisses": be_home,
        "forme_away_buts_marques": bm_away,
        "forme_away_buts_encaisses": be_away,

        # attaque / défense
        "attaque_home": attaque_home,
        "defense_home": defense_home,
        "attaque_away": attaque_away,
        "defense_away": defense_away,

        # xG "macro" (saison) depuis stats_globales
        "xg_dom": float(r["xg_dom"] or 0.3),
        "xg_ext": float(r["xg_ext"] or 0.3),

        # rythme
        "tirs_dom": float(r["tirs_dom"] or 0.0),
        "tirs_ext": float(r["tirs_ext"] or 0.0),

        # ligue
        "league_avg_goals_60d": league_avg,

        # xg attendu à partir d'attaque/défense
        "xg_exp_total": xg_exp_total,
    }

    records.append(rec)

dfX = pd.DataFrame(records)

# clamp tirs (anti outliers)
dfX["tirs_dom"] = dfX["tirs_dom"].clip(0, 25)
dfX["tirs_ext"] = dfX["tirs_ext"].clip(0, 25)

X_all = dfX[FEATURES].astype(float).values
y_all = dfX["y_total"].astype(float).values
dates = dfX["date_match"].values
y_seg_all = np.array([segment_from_y(y) for y in y_all], dtype=int)


# ----------------------------- BASELINE POISSON -----------------------------

def baseline_poisson_row(row: pd.Series) -> float:
    """
    Poisson attaque/défense simple :
        λ_home ≈ (bm_home + be_away) / 2
        λ_away ≈ (bm_away + be_home) / 2
    """
    bm_home = row["forme_home_buts_marques"]
    be_home = row["forme_home_buts_encaisses"]
    bm_away = row["forme_away_buts_marques"]
    be_away = row["forme_away_buts_encaisses"]

    lam_home = 0.5 * (bm_home + be_away)
    lam_away = 0.5 * (bm_away + be_home)
    lam_total = max(0.0, lam_home + lam_away)
    return lam_total

baseline_all = dfX.apply(baseline_poisson_row, axis=1).values


# --------------------------- SPLITS CHRONO ---------------------------

order = np.argsort(dates)
X_all = X_all[order]
y_all = y_all[order]
y_seg_all = y_seg_all[order]
baseline_all = baseline_all[order]
dfX = dfX.iloc[order].reset_index(drop=True)

tscv = TimeSeriesSplit(n_splits=5)
folds = list(tscv.split(X_all))

trainval_idx, test_idx = folds[-1]
X_trainval, y_trainval = X_all[trainval_idx], y_all[trainval_idx]
y_seg_trainval = y_seg_all[trainval_idx]
X_test, y_test = X_all[test_idx], y_all[test_idx]
y_seg_test = y_seg_all[test_idx]
baseline_test = baseline_all[test_idx]


# --------------------------- BACKTEST MODELS ---------------------------

def backtest_models(proba_threshold: float = 0.6):
    """
    Backtest :
      - baseline Poisson
      - global HGB
      - HGB segmenté (low/high) avec fallback global si proba max < threshold
    """
    rows = []

    for fold_i, (tr, val) in enumerate(folds[:-1]):  # exclut le dernier (test final)
        X_tr, X_val = X_all[tr], X_all[val]
        y_tr, y_val = y_all[tr], y_all[val]
        seg_tr, seg_val = y_seg_all[tr], y_seg_all[val]
        base_val = baseline_all[val]

        # 1) modèle global HGB (tous matchs)
        model_global = HistGradientBoostingRegressor(
            max_depth=6,
            max_leaf_nodes=31,
            learning_rate=0.05,
            min_samples_leaf=20,
            random_state=42
        )
        model_global.fit(X_tr, y_tr)
        pred_global_val = model_global.predict(X_val)

        # 2) classifier segmentation (2 classes)
        seg_clf = HistGradientBoostingClassifier(
            max_depth=4,
            max_leaf_nodes=31,
            learning_rate=0.05,
            min_samples_leaf=20,
            random_state=42
        )
        seg_clf.fit(X_tr, seg_tr)
        seg_proba_val = seg_clf.predict_proba(X_val)  # shape (n, 2)
        seg_pred_val = seg_proba_val.argmax(axis=1)
        seg_conf_val = seg_proba_val.max(axis=1)

        # 3) modèles spécialisés par segment
        seg_models: Dict[int, HistGradientBoostingRegressor] = {}
        for seg_label in [0, 1]:
            idx_seg_tr = np.where(seg_tr == seg_label)[0]
            if len(idx_seg_tr) < 80:
                # trop peu de données -> on utilisera le modèle global comme fallback
                seg_models[seg_label] = None
                continue

            model_seg = HistGradientBoostingRegressor(
                max_depth=6,
                max_leaf_nodes=31,
                learning_rate=0.05,
                min_samples_leaf=20,
                random_state=42
            )
            model_seg.fit(X_tr[idx_seg_tr], y_tr[idx_seg_tr])
            seg_models[seg_label] = model_seg

        # prédictions segmentées avec fallback global
        pred_seg_val = np.zeros_like(y_val, dtype=float)
        for i_row in range(len(y_val)):
            label = seg_pred_val[i_row]
            conf = seg_conf_val[i_row]

            if conf < proba_threshold or seg_models[label] is None:
                # pas assez sûr ou pas de modèle spécialisé -> fallback global
                pred_seg_val[i_row] = pred_global_val[i_row]
            else:
                pred_seg_val[i_row] = seg_models[label].predict(X_val[i_row : i_row + 1])[0]

        rows.append({
            "fold": fold_i,
            "baseline": metrics_block(y_val, base_val),
            "global_hgb": metrics_block(y_val, pred_global_val),
            "segmented_hgb": metrics_block(y_val, pred_seg_val),
        })

    # résumé moyenne des folds
    def avg(model_key: str, metric_key: str) -> float:
        vals = [r[model_key][metric_key] for r in rows]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "baseline": {
            "mae": avg("baseline", "mae"),
            "rmse": avg("baseline", "rmse"),
        },
        "global_hgb": {
            "mae": avg("global_hgb", "mae"),
            "rmse": avg("global_hgb", "rmse"),
        },
        "segmented_hgb": {
            "mae": avg("segmented_hgb", "mae"),
            "rmse": avg("segmented_hgb", "rmse"),
        },
        "n_folds": len(rows),
        "proba_threshold": proba_threshold,
    }

    return rows, summary


back_rows, back_summary = backtest_models(proba_threshold=0.6)

print("\n=== Backtest (moyenne des folds, hors test final) ===")
print(json.dumps(back_summary, indent=2, ensure_ascii=False))


# --------------------------- TRAIN FINAL ---------------------------

# 1) modèle global
model_global_final = HistGradientBoostingRegressor(
    max_depth=6,
    max_leaf_nodes=31,
    learning_rate=0.05,
    min_samples_leaf=20,
    random_state=42
)
model_global_final.fit(X_trainval, y_trainval)

# 2) classifier segmentation final
seg_clf_final = HistGradientBoostingClassifier(
    max_depth=4,
    max_leaf_nodes=31,
    learning_rate=0.05,
    min_samples_leaf=20,
    random_state=42
)
seg_clf_final.fit(X_trainval, y_seg_trainval)

# 3) modèles spécialisés finaux
seg_models_final: Dict[int, HistGradientBoostingRegressor] = {}
for seg_label in [0, 1]:
    idx_seg = np.where(y_seg_trainval == seg_label)[0]
    if len(idx_seg) < 80:
        seg_models_final[seg_label] = None
        continue
    m_seg = HistGradientBoostingRegressor(
        max_depth=6,
        max_leaf_nodes=31,
        learning_rate=0.05,
        min_samples_leaf=20,
        random_state=42
    )
    m_seg.fit(X_trainval[idx_seg], y_trainval[idx_seg])
    seg_models_final[seg_label] = m_seg


# --------------------------- TEST FINAL ---------------------------

# baseline
pred_base_test = baseline_test

# global
pred_global_test = model_global_final.predict(X_test)

# segmenté avec fallback
seg_proba_test = seg_clf_final.predict_proba(X_test)
seg_pred_test = seg_proba_test.argmax(axis=1)
seg_conf_test = seg_proba_test.max(axis=1)

pred_seg_test = np.zeros_like(y_test, dtype=float)
threshold_final = 0.6

for i_row in range(len(y_test)):
    label = seg_pred_test[i_row]
    conf = seg_conf_test[i_row]

    if conf < threshold_final or seg_models_final[label] is None:
        pred_seg_test[i_row] = pred_global_test[i_row]
    else:
        pred_seg_test[i_row] = seg_models_final[label].predict(X_test[i_row : i_row + 1])[0]

test_metrics = {
    "baseline": metrics_block(y_test, pred_base_test),
    "global_hgb": metrics_block(y_test, pred_global_test),
    "segmented_hgb": metrics_block(y_test, pred_seg_test),
}

print("\n=== Test final ===")
print(json.dumps(test_metrics, indent=2, ensure_ascii=False))


# --------------------------- SAVE ARTEFACTS ---------------------------

# modèles
joblib.dump(model_global_final, os.path.join(OUT_DIR, "model_global_total_goals.pkl"))
joblib.dump(seg_clf_final, os.path.join(OUT_DIR, "segment_classifier.pkl"))

for seg_label in [0, 1]:
    m = seg_models_final[seg_label]
    if m is not None:
        joblib.dump(m, os.path.join(OUT_DIR, f"model_segment_{seg_label}.pkl"))

# features
with open(os.path.join(OUT_DIR, "FEATURES_TOTAL_BUTS.json"), "w", encoding="utf-8") as f:
    json.dump(FEATURES, f, indent=2, ensure_ascii=False)

# métriques
with open(os.path.join(OUT_DIR, "training_metrics_total_goals.json"), "w", encoding="utf-8") as f:
    json.dump(
        {
            "backtest_summary": back_summary,
            "test_final": test_metrics,
        },
        f,
        indent=2,
        ensure_ascii=False,
    )

print("\n=== DONE: modèles & métriques sauvegardés dans ./models ===")
