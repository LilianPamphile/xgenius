# === train_model.py — v6 (chrono backtest, OOF stacking 3 learners, conformal@50% w/ normalization, calibrated O/U, league macros) ===
# -*- coding: utf-8 -*-

import os, pickle, json
from decimal import Decimal
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import psycopg2

# ML / Metrics
from sklearn.model_selection import TimeSeriesSplit, train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, brier_score_loss
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

# Optuna
import optuna
from optuna.integration import OptunaSearchCV
from optuna.distributions import IntDistribution, FloatDistribution

# -------------------------- CONFIG --------------------------

OUT_DIR_MODELS = "model_files"
OUT_DIR_ARTIFACTS = "artifacts"
os.makedirs(OUT_DIR_MODELS, exist_ok=True)
os.makedirs(OUT_DIR_ARTIFACTS, exist_ok=True)

DATABASE_URL = "postgresql://postgres:jDDqfaqpspVDBBwsqxuaiSDNXjTxjMmP@shortline.proxy.rlwy.net:36536/railway"
if not DATABASE_URL:
    raise ValueError("❌ DATABASE_URL manquant dans les variables d'environnement.")

# -------------------------- UTILS ---------------------------

def np_rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def metrics_block(y_true, y_pred) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": np_rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def to_float_series(s):
    return pd.to_numeric(s, errors="coerce").astype(float)

# ----------------------- DATA LOADING -----------------------

conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()

query = """
    SELECT
        m.game_id,
        m.date::date      AS date_match,
        m.saison          AS saison,
        m.competition     AS competition,
        m.equipe_domicile AS equipe_domicile,
        m.equipe_exterieur AS equipe_exterieur,

        sg1.moyenne_buts AS buts_dom,
        sg1.buts_encaisse::FLOAT / NULLIF(sg1.matchs_joues, 0) AS buts_encaissés_dom,
        sg1.pourcentage_over_2_5 AS over25_dom,
        sg1.pourcentage_over_1_5 AS over15_dom,
        sg1.pourcentage_BTTS     AS btts_dom,
        sg1.possession           AS poss_dom,
        sg1.corners              AS corners_dom,
        sg1.fautes               AS fautes_dom,
        sg1.cartons_jaunes       AS cj_dom,
        sg1.cartons_rouges       AS cr_dom,
        sg1.moyenne_xg_dom,
        sg1.tirs                 AS tirs_dom,
        sg1.tirs_cadres          AS tirs_cadres_dom,

        sg2.moyenne_buts AS buts_ext,
        sg2.buts_encaisse::FLOAT / NULLIF(sg2.matchs_joues, 0) AS buts_encaissés_ext,
        sg2.pourcentage_over_2_5 AS over25_ext,
        sg2.pourcentage_over_1_5 AS over15_ext,
        sg2.pourcentage_BTTS     AS btts_ext,
        sg2.possession           AS poss_ext,
        sg2.corners              AS corners_ext,
        sg2.fautes               AS fautes_ext,
        sg2.cartons_jaunes       AS cj_ext,
        sg2.cartons_rouges       AS cr_ext,
        sg2.moyenne_xg_ext,
        sg2.tirs                 AS tirs_ext,
        sg2.tirs_cadres          AS tirs_cadres_ext,

        s.buts_dom AS buts_m_dom,
        s.buts_ext AS buts_m_ext,
        s.buts_dom + s.buts_ext AS total_buts
    FROM matchs_v2 m
    JOIN stats_globales_v2 sg1 ON m.equipe_domicile = sg1.equipe AND m.competition = sg1.competition AND m.saison = sg1.saison
    JOIN stats_globales_v2 sg2 ON m.equipe_exterieur = sg2.equipe AND m.competition = sg2.competition AND m.saison = sg2.saison
    JOIN stats_matchs_v2 s ON m.game_id = s.game_id
    WHERE s.buts_dom IS NOT NULL AND s.buts_ext IS NOT NULL
"""
cur.execute(query)
rows = cur.fetchall()
cols = [d.name for d in cur.description]
conn.close()

df = pd.DataFrame(rows, columns=cols)

# Convert Decimal → float
for c in df.columns:
    if df[c].dtype == "object" and df[c].notna().any() and isinstance(df[c].dropna().iloc[0], Decimal):
        df[c] = df[c].astype(float)

df["date_match"] = pd.to_datetime(df["date_match"])

# ----------------------- FEATURE ENGINEERING -----------------------

# Historique brut pour calculs de forme & ligue
df_hist = df[[
    "date_match", "competition", "equipe_domicile", "equipe_exterieur",
    "buts_m_dom", "buts_m_ext"
]].copy()
df_hist["total_buts"] = df_hist["buts_m_dom"] + df_hist["buts_m_ext"]

def forme_split(dfh, equipe, date_ref, n=5, role=None, decay=0.85):
    q = dfh["date_match"] < date_ref
    if role is None:
        q &= ((dfh["equipe_domicile"] == equipe) | (dfh["equipe_exterieur"] == equipe))
    elif role is True:
        q &= (dfh["equipe_domicile"] == equipe)
    else:
        q &= (dfh["equipe_exterieur"] == equipe)

    m = dfh.loc[q].sort_values("date_match", ascending=False).head(n)
    if m.empty:
        return 0.0, 0.0, 0.0
    est_dom = (m["equipe_domicile"].values == equipe)
    bm = np.where(est_dom, m["buts_m_dom"].values, m["buts_m_ext"].values)
    be = np.where(est_dom, m["buts_m_ext"].values, m["buts_m_dom"].values)
    tb = m["total_buts"].values
    w = decay ** np.arange(len(m)); w = w / w.sum()
    return (
        float(np.average(bm, weights=w)),
        float(np.average(be, weights=w)),
        float(np.average((tb > 2.5).astype(float), weights=w)),
    )

def forme_complet(dfh, equipe, date_ref, n=5):
    m = dfh[
        ((dfh["equipe_domicile"] == equipe) | (dfh["equipe_exterieur"] == equipe)) &
        (dfh["date_match"] < date_ref)
    ].sort_values("date_match", ascending=False).head(n)
    if m.empty:
        return (0, 0, 0, 0, 0, 0)
    w = np.linspace(1.0, 2.0, len(m))
    est_dom = (m["equipe_domicile"].values == equipe)
    bm = np.where(est_dom, m["buts_m_dom"].values, m["buts_m_ext"].values)
    be = np.where(est_dom, m["buts_m_ext"].values, m["buts_m_dom"].values)
    o25 = (m["total_buts"].values > 2.5).astype(int)
    return (
        float(np.average(bm, weights=w)),
        float(np.average(be, weights=w)),
        float(np.average(o25, weights=w)),
        float(np.std(bm)),
        float(np.std(be)),
        int(np.sum(be == 0)),
    )

# Baselines ligue (rolling) au fil de l'itération
def league_baselines(dfh, competition, date_ref, window_days=60):
    mask = (dfh["competition"] == competition) & (dfh["date_match"] < date_ref) & \
           (dfh["date_match"] >= (date_ref - pd.Timedelta(days=window_days)))
    sub = dfh.loc[mask]
    if sub.empty:
        return 2.5, 0.0  # fallback neutre
    avg_goals = float(sub["total_buts"].mean())
    home_adv = float((sub["buts_m_dom"] - sub["buts_m_ext"]).mean())
    return avg_goals, home_adv

FEATURES_TOTAL_BUTS = [
    # forme split
    "forme_home_buts_marques", "forme_home_buts_encaisses", "forme_home_over25",
    "forme_away_buts_marques", "forme_away_buts_encaisses", "forme_away_over25",
    # variabilité + clean sheets
    "std_marq_dom", "std_enc_dom", "std_marq_ext", "std_enc_ext",
    "clean_dom", "clean_ext",
    # bruts (xG, enc.)
    "moyenne_xg_dom", "moyenne_xg_ext", "buts_encaissés_dom", "buts_encaissés_ext",
    # rythme
    "poss_moyenne", "tirs_dom", "tirs_ext", "tirs_cadres_dom", "tirs_cadres_ext",
    # discipline/corners
    "corners_dom", "corners_ext", "cartons_total",
    # macros ligue
    "league_avg_goals_60d", "league_home_adv_60d"
]

# Build features row-wise (chronologique)
records: List[Dict[str, Any]] = []
for _, r in df.sort_values("date_match").iterrows():
    dom, ext, dref, league = r["equipe_domicile"], r["equipe_exterieur"], r["date_match"], r["competition"]

    bm_home, be_home, o25_home = forme_split(df_hist, dom, dref, role=True)
    bm_away, be_away, o25_away = forme_split(df_hist, ext, dref, role=False)

    _, _, _, std_marq_dom, std_enc_dom, clean_dom = forme_complet(df_hist, dom, dref)
    _, _, _, std_marq_ext, std_enc_ext, clean_ext = forme_complet(df_hist, ext, dref)

    poss_moy = np.nanmean([r.get("poss_dom", np.nan), r.get("poss_ext", np.nan)])
    cj_dom, cr_dom = r.get("cj_dom", 0.0), r.get("cr_dom", 0.0)
    cj_ext, cr_ext = r.get("cj_ext", 0.0), r.get("cr_ext", 0.0)
    cartons_total = float((cj_dom or 0) + (cr_dom or 0) + (cj_ext or 0) + (cr_ext or 0))

    # macros ligue (60j)
    avg_goals_60, home_adv_60 = league_baselines(df_hist, league, dref, window_days=60)

    features = {
        "game_id": r["game_id"],
        "date_match": dref,
        "competition": league,
        "y_total": float(r["total_buts"]),
        # forme split
        "forme_home_buts_marques": bm_home,
        "forme_home_buts_encaisses": be_home,
        "forme_home_over25": o25_home,
        "forme_away_buts_marques": bm_away,
        "forme_away_buts_encaisses": be_away,
        "forme_away_over25": o25_away,
        # variabilité / clean
        "std_marq_dom": std_marq_dom, "std_enc_dom": std_enc_dom,
        "std_marq_ext": std_marq_ext, "std_enc_ext": std_enc_ext,
        "clean_dom": clean_dom, "clean_ext": clean_ext,
        # bruts
        "moyenne_xg_dom": float(r.get("moyenne_xg_dom", 0.3) or 0.3),
        "moyenne_xg_ext": float(r.get("moyenne_xg_ext", 0.3) or 0.3),
        "buts_encaissés_dom": float(r.get("buts_encaissés_dom", 0.0) or 0.0),
        "buts_encaissés_ext": float(r.get("buts_encaissés_ext", 0.0) or 0.0),
        # rythme/tempo
        "poss_moyenne": float(poss_moy) if not np.isnan(poss_moy) else 50.0,
        "tirs_dom": float(r.get("tirs_dom", 0.0) or 0.0),
        "tirs_ext": float(r.get("tirs_ext", 0.0) or 0.0),
        "tirs_cadres_dom": float(r.get("tirs_cadres_dom", 0.0) or 0.0),
        "tirs_cadres_ext": float(r.get("tirs_cadres_ext", 0.0) or 0.0),
        # discipline/corners
        "corners_dom": float(r.get("corners_dom", 0.0) or 0.0),
        "corners_ext": float(r.get("corners_ext", 0.0) or 0.0),
        "cartons_total": cartons_total,
        # macros ligue
        "league_avg_goals_60d": float(avg_goals_60),
        "league_home_adv_60d": float(home_adv_60),
    }
    records.append(features)

dfX = pd.DataFrame(records)

# Clamp & nettoyage basique pour outliers
caps = {
    "tirs_dom": 25, "tirs_ext": 25,
    "tirs_cadres_dom": 12, "tirs_cadres_ext": 12,
    "corners_dom": 15, "corners_ext": 15,
    "fautes_dom": 30, "fautes_ext": 30,
    "cartons_total": 12
}
for c, cap in caps.items():
    if c in dfX:
        dfX[c] = to_float_series(dfX[c]).clip(0, cap)

FEATURES = FEATURES_TOTAL_BUTS[:]  # copie

X_all = dfX[FEATURES].astype(float).values
y_all = dfX["y_total"].astype(float).values
dates = dfX["date_match"].values

# ----------------------- CHRONO SPLITS -----------------------

order = np.argsort(dates)
X_all, y_all, dfX = X_all[order], y_all[order], dfX.iloc[order].reset_index(drop=True)

n_splits = 5 if len(y_all) >= 500 else 4
tscv = TimeSeriesSplit(n_splits=n_splits)
folds = list(tscv.split(X_all))

# Dernier split = test final
trainval_idx, test_idx = folds[-1]
X_trainval, y_trainval = X_all[trainval_idx], y_all[trainval_idx]
X_test, y_test = X_all[test_idx], y_all[test_idx]

scaler = StandardScaler()
X_trainval_sc = scaler.fit_transform(X_trainval)
X_test_sc = scaler.transform(X_test)

# ------------------- OOF LEARNERS (3 bases) -------------------

inner_tscv = TimeSeriesSplit(n_splits=3 if len(y_trainval) >= 300 else 2)

oof_cat = np.zeros_like(y_trainval, dtype=float)
oof_hgb = np.zeros_like(y_trainval, dtype=float)
oof_lgb = np.zeros_like(y_trainval, dtype=float)

cat_models, hgb_models, lgb_models = [], [], []

# Optuna param spaces
cat_space = {
    "depth": IntDistribution(4, 10),
    "learning_rate": FloatDistribution(0.02, 0.12),
    "iterations": IntDistribution(300, 800),
    "l2_leaf_reg": FloatDistribution(1.0, 8.0),
}
lgb_space = {
    "num_leaves": IntDistribution(16, 64),
    "learning_rate": FloatDistribution(0.02, 0.12),
    "n_estimators": IntDistribution(300, 800),
    "min_data_in_leaf": IntDistribution(20, 80),
    "feature_fraction": FloatDistribution(0.6, 1.0),
    "bagging_fraction": FloatDistribution(0.6, 1.0),
}

for tr_idx, va_idx in inner_tscv.split(X_trainval_sc):
    X_tr, X_va = X_trainval_sc[tr_idx], X_trainval_sc[va_idx]
    y_tr, y_va = y_trainval[tr_idx], y_trainval[va_idx]

    # CatBoost (Optuna, n_trials=60)
    cat_search = OptunaSearchCV(
        estimator=CatBoostRegressor(
            verbose=0, random_seed=42, loss_function="RMSE", subsample=0.8, rsm=0.8
        ),
        param_distributions=cat_space,
        n_trials=20,
        cv=KFold(n_splits=3, shuffle=True, random_state=42),
        scoring="neg_mean_absolute_error",
        n_jobs=1
    )
    cat_search.fit(X_tr, y_tr)
    cat = cat_search.best_estimator_
    oof_cat[va_idx] = cat.predict(X_va)
    cat_models.append(cat)

    # HistGradientBoosting (grille fixe sobre)
    hgb = HistGradientBoostingRegressor(
        max_iter=350, learning_rate=0.06, max_depth=6,
        l2_regularization=0.0, min_samples_leaf=20, random_state=42
    )
    hgb.fit(X_tr, y_tr)
    oof_hgb[va_idx] = hgb.predict(X_va)
    hgb_models.append(hgb)

    # LightGBM reg (Optuna, n_trials=60)
    lgb_search = OptunaSearchCV(
        estimator=LGBMRegressor(objective="regression", random_state=42),
        param_distributions=lgb_space,
        n_trials=20,
        cv=KFold(n_splits=3, shuffle=True, random_state=42),
        scoring="neg_mean_absolute_error",
        n_jobs=1
    )
    lgb_search.fit(X_tr, y_tr)
    lgb = lgb_search.best_estimator_
    oof_lgb[va_idx] = lgb.predict(X_va)
    lgb_models.append(lgb)

# Learners FULL (sur tout trainval) pour l'inférence finale
cat_full = CatBoostRegressor(verbose=0, random_seed=42, loss_function="RMSE",
                             depth=8, learning_rate=0.06, iterations=600, subsample=0.8, rsm=0.8)
cat_full.fit(X_trainval_sc, y_trainval)

hgb_full = HistGradientBoostingRegressor(max_iter=350, learning_rate=0.06, max_depth=6,
                                         l2_regularization=0.0, min_samples_leaf=20, random_state=42)
hgb_full.fit(X_trainval_sc, y_trainval)

lgb_full = LGBMRegressor(objective="regression", random_state=42,
                         n_estimators=600, learning_rate=0.06, num_leaves=31,
                         min_data_in_leaf=30, feature_fraction=0.9, bagging_fraction=0.9)
lgb_full.fit(X_trainval_sc, y_trainval)

# Meta Stacking (Ridge) sur OOF
stack_X_train = np.vstack([oof_cat, oof_hgb, oof_lgb]).T
meta = RidgeCV(alphas=[0.1, 1.0, 10.0])
meta.fit(stack_X_train, y_trainval)

# Test final
pred_cat_test = cat_full.predict(X_test_sc)
pred_hgb_test = hgb_full.predict(X_test_sc)
pred_lgb_test = lgb_full.predict(X_test_sc)
stack_X_test = np.vstack([pred_cat_test, pred_hgb_test, pred_lgb_test]).T
pred_stack_test = meta.predict(stack_X_test)

results: Dict[str, Any] = {}
results["catboost"] = metrics_block(y_test, pred_cat_test)
results["hist_gbr"] = metrics_block(y_test, pred_hgb_test)
results["lightgbm"] = metrics_block(y_test, pred_lgb_test)
results["stacking"] = metrics_block(y_test, pred_stack_test)

# ------------------- BACKTEST (report multi-fold) -------------------

def evaluate_on_folds(X, y, folds) -> Dict[str, Any]:
    rows = []
    for k, (tr, te) in enumerate(folds, 1):
        # scaler fold
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])

        # simple learners re-fit (param full pour vitesse)
        c = CatBoostRegressor(verbose=0, random_seed=42, loss_function="RMSE",
                              depth=8, learning_rate=0.06, iterations=600, subsample=0.8, rsm=0.8)
        c.fit(Xtr, y[tr])
        h = HistGradientBoostingRegressor(max_iter=350, learning_rate=0.06, max_depth=6,
                                          l2_regularization=0.0, min_samples_leaf=20, random_state=42)
        h.fit(Xtr, y[tr])
        l = LGBMRegressor(objective="regression", random_state=42,
                          n_estimators=600, learning_rate=0.06, num_leaves=31,
                          min_data_in_leaf=30, feature_fraction=0.9, bagging_fraction=0.9)
        l.fit(Xtr, y[tr])

        pc, ph, pl = c.predict(Xte), h.predict(Xte), l.predict(Xte)
        sx = np.vstack([pc, ph, pl]).T
        m = RidgeCV(alphas=[0.1, 1.0, 10.0]).fit(sx, y[te])
        ps = m.predict(sx)

        rows.append({
            "fold": k,
            "cat_mae": mean_absolute_error(y[te], pc),
            "hgb_mae": mean_absolute_error(y[te], ph),
            "lgb_mae": mean_absolute_error(y[te], pl),
            "stack_mae": mean_absolute_error(y[te], ps),
            "stack_rmse": np_rmse(y[te], ps),
            "stack_r2": r2_score(y[te], ps)
        })
    dfrep = pd.DataFrame(rows)
    dfrep.to_csv(os.path.join(OUT_DIR_ARTIFACTS, "backtest_folds.csv"), index=False)
    summary = {
        "stack_mae_mean": float(dfrep["stack_mae"].mean()),
        "stack_mae_std":  float(dfrep["stack_mae"].std(ddof=1)),
        "stack_rmse_mean": float(dfrep["stack_rmse"].mean()),
        "stack_r2_mean": float(dfrep["stack_r2"].mean())
    }
    return {"folds": rows, "summary": summary}

results["backtest"] = evaluate_on_folds(X_all, y_all, folds[:-1])  # hors test final

# ------------------- CONFORMAL QUANTILES (50%) -------------------

# quantile models on trainval
q_params = dict(n_estimators=500, learning_rate=0.06, num_leaves=31,
                min_data_in_leaf=30, subsample=0.9, feature_fraction=0.9, random_state=42)
q25 = LGBMRegressor(objective="quantile", alpha=0.25, **q_params)
q75 = LGBMRegressor(objective="quantile", alpha=0.75, **q_params)
q25.fit(X_trainval_sc, y_trainval); q75.fit(X_trainval_sc, y_trainval)

# calibration set = dernière tranche de trainval (20%)
split_cut = int(len(X_trainval_sc) * 0.8)
X_tr_c, X_cal_c = X_trainval_sc[:split_cut], X_trainval_sc[split_cut:]
y_tr_c, y_cal_c = y_trainval[:split_cut], y_trainval[split_cut:]

q25_c = LGBMRegressor(objective="quantile", alpha=0.25, **q_params).fit(X_tr_c, y_tr_c)
q75_c = LGBMRegressor(objective="quantile", alpha=0.75, **q_params).fit(X_tr_c, y_tr_c)
p25_cal = q25_c.predict(X_cal_c)
p75_cal = q75_c.predict(X_cal_c)

def find_delta(y_true, lo, hi, target=0.50):
    deltas = np.linspace(0.0, 1.5, 31)
    best = 0.0; best_diff = 1e9
    for d in deltas:
        cover = np.mean((y_true >= (lo - d)) & (y_true <= (hi + d)))
        diff = abs(cover - target)
        if diff < best_diff:
            best, best_diff = d, diff
    return float(best)

OFFSET = find_delta(y_cal_c, p25_cal, p75_cal, target=0.50)

# local width normalization: predict |residual|
res_target = np.abs(y_trainval - cat_full.predict(X_trainval_sc))
res_norm_model = LGBMRegressor(objective="regression_l1", n_estimators=400, random_state=42)
# enrichir le modèle de largeur avec macros ligue
X_trainval_enriched = np.hstack([
    X_trainval_sc,
    to_float_series(dfX.loc[trainval_idx, "league_avg_goals_60d"]).values.reshape(-1,1),
    to_float_series(dfX.loc[trainval_idx, "league_home_adv_60d"]).values.reshape(-1,1),
])
res_norm_model.fit(X_trainval_enriched, res_target)

# test final bounds
p25_test = q25.predict(X_test_sc) - OFFSET
p75_test = q75.predict(X_test_sc) + OFFSET

scale_test = res_norm_model.predict(np.hstack([
    X_test_sc,
    to_float_series(dfX.loc[test_idx, "league_avg_goals_60d"]).values.reshape(-1,1),
    to_float_series(dfX.loc[test_idx, "league_home_adv_60d"]).values.reshape(-1,1),
]))
scale_test = np.clip(scale_test, 0.5, 2.5)
p25_test = pred_stack_test - (pred_stack_test - p25_test) * (1.0 / scale_test)
p75_test = pred_stack_test + (p75_test - pred_stack_test) * (1.0 / scale_test)

coverage = float(np.mean((y_test >= p25_test) & (y_test <= p75_test)))
width    = float(np.mean(p75_test - p25_test))
results["conformal"] = {"coverage": coverage, "width": width}

# ------------------- HEURISTIC (non-linéaire) -------------------

# construire features heuristiques (compat main.py)
dfH = dfX.copy()
dfH["tirs_cadres_total"] = dfH["tirs_cadres_dom"] + dfH["tirs_cadres_ext"]
dfH["forme_dom_marq"] = dfH["forme_home_buts_marques"]
dfH["forme_ext_marq"] = dfH["forme_away_buts_marques"]
dfH["solidite_dom"] = 1.0 / (dfH["buts_encaissés_dom"] + 0.1)
dfH["solidite_ext"] = 1.0 / (dfH["buts_encaissés_ext"] + 0.1)
dfH["corners"] = dfH["corners_dom"] + dfH["corners_ext"]
dfH["poss"] = dfH["poss_moyenne"]

FEATURES_HEUR = [
    "buts_dom", "buts_ext",   # approximées via forme marquée (proxy)
    "over25_dom", "over25_ext", "btts_dom", "btts_ext",  # on n'a pas directement → proxys simples
    "moyenne_xg_dom", "moyenne_xg_ext",
    "tirs_cadres_total",
    "forme_dom_marq", "forme_ext_marq",
    "solidite_dom", "solidite_ext",
    "corners", "cartons_total",
    "poss"
]

# proxys (on réutilise les colonnes existantes)
dfH["buts_dom"] = dfH["forme_home_buts_marques"]
dfH["buts_ext"] = dfH["forme_away_buts_marques"]
dfH["over25_dom"] = dfH["forme_home_over25"]
dfH["over25_ext"] = dfH["forme_away_over25"]
dfH["btts_dom"] = 0.0
dfH["btts_ext"] = 0.0

X_score_all = dfH[FEATURES_HEUR].astype(float).values
y_score_all = dfX["y_total"].astype(float).values

# split aligné
X_score_tr, X_score_te = X_score_all[trainval_idx], X_score_all[test_idx]
y_score_tr, y_score_te = y_score_all[trainval_idx], y_score_all[test_idx]

heur = CatBoostRegressor(
    depth=6, iterations=500, learning_rate=0.05, loss_function="MAE", random_seed=42, verbose=0
)
heur.fit(X_score_tr, y_score_tr)
preds_score_te = heur.predict(X_score_te)
results["score_heuristique"] = metrics_block(y_score_te, preds_score_te)

# ------------------- CLASSIFIER OVER 2.5 (calibrated) -------------------

label_over = (y_all > 2.5).astype(int)
y_class_tr, y_class_te = label_over[trainval_idx], label_over[test_idx]
X_class_tr, X_class_te = X_trainval_sc, X_test_sc

# calibration split DÉDIÉ sur trainval (dernier 20%)
cut_c = int(0.8 * len(X_class_tr))
X_c_train, X_c_cal = X_class_tr[:cut_c], X_class_tr[cut_c:]
y_c_train, y_c_cal = y_class_tr[:cut_c], y_class_tr[cut_c:]

base_clf = GradientBoostingClassifier(n_estimators=300, learning_rate=0.06, max_depth=4, random_state=42)
base_clf.fit(X_c_train, y_c_train)

cal_clf = CalibratedClassifierCV(base_clf, method="isotonic", cv="prefit")
cal_clf.fit(X_c_cal, y_c_cal)

yprob_te = cal_clf.predict_proba(X_class_te)[:, 1]
acc_te   = float(accuracy_score(y_class_te, (yprob_te >= 0.5).astype(int)))
brier_te = float(brier_score_loss(y_class_te, yprob_te))
results["over25_classifier"] = {"accuracy": acc_te, "brier": brier_te}

# ------------------- EXPORTS -------------------

# 1) Artefacts modèles
with open(os.path.join(OUT_DIR_MODELS, "model_total_buts_catboost_optuna.pkl"), "wb") as f:
    pickle.dump(cat_full, f)
with open(os.path.join(OUT_DIR_MODELS, "model_total_buts_hist_gradient_boosting.pkl"), "wb") as f:
    pickle.dump(hgb_full, f)
with open(os.path.join(OUT_DIR_MODELS, "model_total_buts_lightgbm.pkl"), "wb") as f:
    pickle.dump(lgb_full, f)
with open(os.path.join(OUT_DIR_MODELS, "model_total_buts_stacking.pkl"), "wb") as f:
    pickle.dump(meta, f)

with open(os.path.join(OUT_DIR_MODELS, "model_total_buts_conformal_p25.pkl"), "wb") as f:
    pickle.dump(q25, f)
with open(os.path.join(OUT_DIR_MODELS, "model_total_buts_conformal_p75.pkl"), "wb") as f:
    pickle.dump(q75, f)

with open(os.path.join(OUT_DIR_MODELS, "scaler_total_buts.pkl"), "wb") as f:
    pickle.dump(scaler, f)
with open(os.path.join(OUT_DIR_MODELS, "features_list.pkl"), "wb") as f:
    pickle.dump(FEATURES, f)

with open(os.path.join(OUT_DIR_MODELS, "regression_score_heuristique.pkl"), "wb") as f:
    pickle.dump(heur, f)
with open(os.path.join(OUT_DIR_MODELS, "features_list_score_heuristique.pkl"), "wb") as f:
    pickle.dump(FEATURES_HEUR, f)

with open(os.path.join(OUT_DIR_MODELS, "model_over25_classifier.pkl"), "wb") as f:
    pickle.dump(cal_clf, f)

with open(os.path.join(OUT_DIR_MODELS, "offset_conformal.pkl"), "wb") as f:
    pickle.dump(OFFSET, f)

mae_info = {
    "mae_cat": results["catboost"]["mae"],
    "mae_hgb": results["hist_gbr"]["mae"],
    "mae_lgb": results["lightgbm"]["mae"]
}
with open(os.path.join(OUT_DIR_MODELS, "mae_models.pkl"), "wb") as f:
    pickle.dump(mae_info, f)

# 2) Audits : OOF & résidus
pd.DataFrame({
    "y_trainval": y_trainval,
    "oof_cat": oof_cat,
    "oof_hgb": oof_hgb,
    "oof_lgb": oof_lgb
}).to_csv(os.path.join(OUT_DIR_ARTIFACTS, "oof_predictions.csv"), index=False)

pd.DataFrame({
    "y_trainval": y_trainval,
    "cat_full_pred_trainval": cat_full.predict(X_trainval_sc),
    "abs_residual": res_target
}).to_csv(os.path.join(OUT_DIR_ARTIFACTS, "residuals_trainval.csv"), index=False)

# 3) Rapport JSON synthétique
with open(os.path.join(OUT_DIR_ARTIFACTS, "results_summary.json"), "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

import subprocess, shutil

TOKEN_HUB = os.getenv("TOKEN_HUB")
if not TOKEN_HUB:
    raise ValueError("❌ Le token GitHub (TOKEN_HUB) n'est pas défini.")

REPO = "LilianPamphile/xgenius"   # ⚠️ ton nouveau repo
REPO_DIR = "train_push"
REPO_URL = f"https://{TOKEN_HUB}@github.com/{REPO}.git"

# Nettoyage / clone
if os.path.exists(REPO_DIR):
    shutil.rmtree(REPO_DIR)
subprocess.check_call(["git", "clone", REPO_URL, REPO_DIR])

# Copie des fichiers modèles et artefacts
import shutil
dst_models = os.path.join(REPO_DIR, "model_files")
dst_artifacts = os.path.join(REPO_DIR, "artifacts")
shutil.rmtree(dst_models, ignore_errors=True)
shutil.rmtree(dst_artifacts, ignore_errors=True)
shutil.copytree(OUT_DIR_MODELS, dst_models)
shutil.copytree(OUT_DIR_ARTIFACTS, dst_artifacts)

# Commit & push
subprocess.check_call(["git", "config", "user.email", "lilian.pamphile.bts@gmail.com"], cwd=REPO_DIR)
subprocess.check_call(["git", "config", "user.name", "LilianPamphile"], cwd=REPO_DIR)

subprocess.check_call(["git", "add", "."], cwd=REPO_DIR)
try:
    subprocess.check_call(["git", "commit", "-m", f"🤖 Update models {datetime.utcnow().isoformat()}"], cwd=REPO_DIR)
except subprocess.CalledProcessError:
    print("ℹ️ Aucun changement à committer.")

subprocess.check_call(["git", "push", "origin", "main"], cwd=REPO_DIR)
print("✅ Modèles et artefacts poussés sur GitHub.")

print("\n✅ Entraînement terminé.")
print("— Test final (stacking):", results["stacking"])
print("— Conformal p25–p75: coverage=%.2f, width=%.2f" % (coverage, width))
print("— Classif Over2.5 (test): acc=%.3f, brier=%.3f" % (acc_te, brier_te))
print("📁 Artefacts:", OUT_DIR_MODELS, "• Audits:", OUT_DIR_ARTIFACTS)
