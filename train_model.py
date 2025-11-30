# ========================= train_model.py — vFINAL ==========================
# Objectif : prédire le total de buts avec 3 modèles (baseline, CatBoost, HGB)
# Pipeline simple, fiable, chrono, sans modèles inutiles
# ============================================================================

import os
import json
from decimal import Decimal
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import psycopg2

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
from catboost import CatBoostRegressor
import joblib


# -------------------------- CONFIG --------------------------

OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)

DATABASE_URL = "postgresql://postgres:jDDqfaqpspVDBBwsqxuaiSDNXjTxjMmP@shortline.proxy.rlwy.net:36536/railway"


# -------------------------- UTILS ---------------------------

def np_rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def metrics_block(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": np_rmse(y_true, y_pred),
        "n": int(len(y_true))
    }


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
      ON m.equipe_domicile = sg1.equipe AND m.competition = sg1.competition AND m.saison = sg1.saison
    JOIN stats_globales_v2 sg2
      ON m.equipe_exterieur = sg2.equipe AND m.competition = sg2.competition AND m.saison = sg2.saison
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

for c in df.columns:
    if isinstance(df[c].dropna().iloc[0], Decimal):
        df[c] = df[c].astype(float)

df["date_match"] = pd.to_datetime(df["date_match"])


# ----------------------- FEATURE ENGINEERING -----------------------

df_hist = df[[
    "date_match", "competition",
    "equipe_domicile", "equipe_exterieur",
    "buts_dom", "buts_ext"
]].copy()
df_hist["total_buts"] = df_hist["buts_dom"] + df_hist["buts_ext"]


def forme_split(equipe, date_ref, role):
    """On calcule forme buts marqués/encaissés (5 derniers matchs)."""
    q = df_hist["date_match"] < date_ref

    if role:  # domicile
        q &= (df_hist["equipe_domicile"] == equipe)
    else:      # extérieur
        q &= (df_hist["equipe_exterieur"] == equipe)

    m = df_hist.loc[q].sort_values("date_match", ascending=False).head(5)
    if len(m) == 0:
        return 0., 0.

    if role:
        bm = m["buts_dom"].values
        be = m["buts_ext"].values
    else:
        bm = m["buts_ext"].values
        be = m["buts_dom"].values

    return float(np.mean(bm)), float(np.mean(be))


def league_avg_goals(competition, date_ref):
    q = (df_hist["competition"] == competition) & (df_hist["date_match"] < date_ref)
    m = df_hist.loc[q].tail(60)
    if len(m) == 0:
        return 2.5
    return float(m["total_buts"].mean())


FEATURES = [
    "forme_home_buts_marques",
    "forme_home_buts_encaisses",
    "forme_away_buts_marques",
    "forme_away_buts_encaisses",
    "xg_dom",
    "xg_ext",
    "tirs_dom",
    "tirs_ext",
    "league_avg_goals_60d"
]


records: List[Dict[str, Any]] = []

for _, r in df.sort_values("date_match").iterrows():

    bm_home, be_home = forme_split(r["equipe_domicile"], r["date_match"], role=True)
    bm_away, be_away = forme_split(r["equipe_exterieur"], r["date_match"], role=False)

    league_avg = league_avg_goals(r["competition"], r["date_match"])

    rec = {
        "game_id": r["game_id"],
        "date_match": r["date_match"],
        "competition": r["competition"],
        "y_total": float(r["total_buts"]),

        "forme_home_buts_marques": bm_home,
        "forme_home_buts_encaisses": be_home,
        "forme_away_buts_marques": bm_away,
        "forme_away_buts_encaisses": be_away,

        "xg_dom": float(r["xg_dom"] or 0.3),
        "xg_ext": float(r["xg_ext"] or 0.3),

        "tirs_dom": float(r["tirs_dom"] or 0.0),
        "tirs_ext": float(r["tirs_ext"] or 0.0),

        "league_avg_goals_60d": league_avg
    }

    records.append(rec)

dfX = pd.DataFrame(records)

# clamp tirs (sécurité anti-outliers)
dfX["tirs_dom"] = dfX["tirs_dom"].clip(0, 25)
dfX["tirs_ext"] = dfX["tirs_ext"].clip(0, 25)


# ----------------------------- DATASET -----------------------------

X_all = dfX[FEATURES].astype(float).values
y_all = dfX["y_total"].astype(float).values
dates = dfX["date_match"].values


# ----------------------------- BASELINE -----------------------------

def baseline_poisson(r):
    lam_home = 0.5 * (r["forme_home_buts_marques"] + r["forme_away_buts_encaisses"])
    lam_away = 0.5 * (r["forme_away_buts_marques"] + r["forme_home_buts_encaisses"])
    return max(0., lam_home + lam_away)

baseline_all = dfX.apply(baseline_poisson, axis=1).values


# --------------------------- SPLITS ---------------------------

tscv = TimeSeriesSplit(n_splits=5)
folds = list(tscv.split(X_all))

trainval_idx, test_idx = folds[-1]
X_trainval, y_trainval = X_all[trainval_idx], y_all[trainval_idx]
X_test, y_test = X_all[test_idx], y_all[test_idx]
baseline_test = baseline_all[test_idx]


# --------------------------- BACKTEST ---------------------------

def backtest_models():

    rows = []

    for fold_i, (tr, val) in enumerate(folds[:-1]):
        X_tr, X_val = X_all[tr], X_all[val]
        y_tr, y_val = y_all[tr], y_all[val]
        b_val = baseline_all[val]

        model_cat = CatBoostRegressor(
            depth=6, learning_rate=0.06, iterations=600,
            loss_function="RMSE", random_seed=42, verbose=False
        )
        model_cat.fit(X_tr, y_tr)
        p_cat = model_cat.predict(X_val)

        model_hgb = HistGradientBoostingRegressor(
            max_depth=6, max_leaf_nodes=31, learning_rate=0.05,
            min_samples_leaf=20, random_state=42
        )
        model_hgb.fit(X_tr, y_tr)
        p_hgb = model_hgb.predict(X_val)

        rows.append({
            "fold": fold_i,
            "baseline": metrics_block(y_val, b_val),
            "catboost": metrics_block(y_val, p_cat),
            "hgb": metrics_block(y_val, p_hgb)
        })

    return rows


back_rows = backtest_models()

# résumé
summary = {
    "baseline": {
        "mae": float(np.mean([r["baseline"]["mae"] for r in back_rows])),
        "rmse": float(np.mean([r["baseline"]["rmse"] for r in back_rows])),
    },
    "catboost": {
        "mae": float(np.mean([r["catboost"]["mae"] for r in back_rows])),
        "rmse": float(np.mean([r["catboost"]["rmse"] for r in back_rows])),
    },
    "hgb": {
        "mae": float(np.mean([r["hgb"]["mae"] for r in back_rows])),
        "rmse": float(np.mean([r["hgb"]["rmse"] for r in back_rows])),
    }
}

print("\n=== Backtest (moyenne folds) ===")
print(json.dumps(summary, indent=2, ensure_ascii=False))


# --------------------------- TRAIN FINAL ---------------------------

model_cat = CatBoostRegressor(
    depth=6, learning_rate=0.06, iterations=800,
    loss_function="RMSE", random_seed=42, verbose=False
)
model_cat.fit(X_trainval, y_trainval)

model_hgb = HistGradientBoostingRegressor(
    max_depth=6, max_leaf_nodes=31, learning_rate=0.05,
    min_samples_leaf=20, random_state=42
)
model_hgb.fit(X_trainval, y_trainval)


# --------------------------- TEST FINAL ---------------------------

p_cat = model_cat.predict(X_test)
p_hgb = model_hgb.predict(X_test)
p_mean = 0.5 * (p_cat + p_hgb)

test_metrics = {
    "baseline": metrics_block(y_test, baseline_test),
    "catboost": metrics_block(y_test, p_cat),
    "hgb": metrics_block(y_test, p_hgb),
    "mean_ensemble": metrics_block(y_test, p_mean)
}

print("\n=== Test final ===")
print(json.dumps(test_metrics, indent=2, ensure_ascii=False))


# --------------------------- SAVE ---------------------------

joblib.dump(model_cat, f"{OUT_DIR}/model_cat_total_goals.pkl")
joblib.dump(model_hgb, f"{OUT_DIR}/model_hgb_total_goals.pkl")

with open(f"{OUT_DIR}/FEATURES_TOTAL_BUTS.json", "w") as f:
    json.dump(FEATURES, f, indent=2, ensure_ascii=False)

with open(f"{OUT_DIR}/training_metrics_total_goals.json", "w") as f:
    json.dump({
        "backtest": summary,
        "test_final": test_metrics
    }, f, indent=2, ensure_ascii=False)

print("\n=== DONE: modèles & métriques sauvegardés dans ./models ===")
