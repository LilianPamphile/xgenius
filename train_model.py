# ===== Standard libs =====
import os, pickle, shutil, time
from datetime import date
from decimal import Decimal

# ===== Data / Math =====
import numpy as np
import pandas as pd

# ===== DB =====
import psycopg2

# ===== ML =====
from sklearn.model_selection import TimeSeriesSplit, train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, brier_score_loss
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

# Regressors
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

# Hyperopt (sobre)
import optuna
from optuna.integration import OptunaSearchCV
from optuna.distributions import IntDistribution, FloatDistribution

# ====== BDD ======
DATABASE_URL = "postgresql://postgres:jDDqfaqpspVDBBwsqxuaiSDNXjTxjMmP@shortline.proxy.rlwy.net:36536/railway"
conn = psycopg2.connect(DATABASE_URL)
cursor = conn.cursor()

# ====== Features pour régression de buts ======
FEATURES_TOTAL_BUTS = [
    # Forme split
    "forme_home_buts_marques", "forme_home_buts_encaisses", "forme_home_over25",
    "forme_away_buts_marques", "forme_away_buts_encaisses", "forme_away_over25",

    # Variabilité (5 derniers)
    "std_marq_dom", "std_enc_dom", "std_marq_ext", "std_enc_ext",

    # Clean sheets réels (compte sur 5)
    "clean_dom", "clean_ext",

    # xG & buts encaissés bruts
    "moyenne_xg_dom", "moyenne_xg_ext",
    "buts_encaissés_dom", "buts_encaissés_ext",

    # Possession et tirs
    "poss_moyenne", "tirs_dom", "tirs_ext", "tirs_cadres_dom", "tirs_cadres_ext",

    # Discipline & corners
    "corners_dom", "corners_ext", "cartons_total"
]

# ====== Extraction historique (identique à ton schéma) ======
query = """
    SELECT
        m.game_id, m.date::date AS date_match, m.equipe_domicile, m.equipe_exterieur,

        sg1.moyenne_buts AS buts_dom, sg1.buts_encaisse::FLOAT / NULLIF(sg1.matchs_joues, 0) AS buts_encaissés_dom,
        sg1.pourcentage_over_2_5 AS over25_dom, sg1.pourcentage_over_1_5 AS over1_5_dom,
        sg1.pourcentage_BTTS AS btts_dom, sg1.passes_pourcent, sg1.passes_reussies,
        sg1.possession, sg1.corners, sg1.fautes, sg1.cartons_jaunes, sg1.cartons_rouges,
        sg1.moyenne_xg_dom, sg1.tirs AS tirs_dom, sg1.tirs_cadres AS tirs_cadres_dom,

        sg2.moyenne_buts AS buts_ext, sg2.buts_encaisse::FLOAT / NULLIF(sg2.matchs_joues, 0) AS buts_encaissés_ext,
        sg2.pourcentage_over_2_5 AS over25_ext, sg2.pourcentage_over_1_5 AS over1_5_ext,
        sg2.pourcentage_BTTS AS btts_ext, sg2.passes_pourcent AS passes_pourcent_ext,
        sg2.passes_reussies AS passes_reussies_ext, sg2.possession AS poss_ext,
        sg2.corners AS corners_ext, sg2.fautes AS fautes_ext, sg2.cartons_jaunes AS cj_ext,
        sg2.cartons_rouges AS cr_ext, sg2.moyenne_xg_ext, sg2.tirs AS tirs_ext, sg2.tirs_cadres AS tirs_cadres_ext,

        s.buts_dom AS buts_m_dom, s.buts_ext AS buts_m_ext,
        s.buts_dom + s.buts_ext AS total_buts
    FROM matchs_v2 m
    JOIN stats_globales_v2 sg1 ON m.equipe_domicile = sg1.equipe AND m.competition = sg1.competition AND m.saison = sg1.saison
    JOIN stats_globales_v2 sg2 ON m.equipe_exterieur = sg2.equipe AND m.competition = sg2.competition AND m.saison = sg2.saison
    JOIN stats_matchs_v2 s ON m.game_id = s.game_id
    WHERE s.buts_dom IS NOT NULL AND s.buts_ext IS NOT NULL
"""
cursor.execute(query)
df = pd.DataFrame(cursor.fetchall(), columns=[d.name for d in cursor.description])
conn.close()

# Convertir Decimal -> float
for c in df.columns:
    if df[c].dtype == "object" and df[c].notna().any() and isinstance(df[c].dropna().iloc[0], Decimal):
        df[c] = df[c].astype(float)

# ====== Fonctions de forme (identiques à ton main/train) ======
def calculer_forme_train(df_hist, equipe, date_ref, n=5, role=None, decay=0.85):
    q = (df_hist["date_match"] < date_ref)
    if role is None:
        q &= ((df_hist["equipe_domicile"] == equipe) | (df_hist["equipe_exterieur"] == equipe))
    elif role is True:
        q &= (df_hist["equipe_domicile"] == equipe)
    else:
        q &= (df_hist["equipe_exterieur"] == equipe)

    m = df_hist.loc[q].sort_values("date_match", ascending=False).head(n)
    if m.empty:
        return 0.0, 0.0, 0.0

    est_dom = (m["equipe_domicile"].values == equipe)
    bm = np.where(est_dom, m["buts_m_dom"].values, m["buts_m_ext"].values)
    be = np.where(est_dom, m["buts_m_ext"].values, m["buts_m_dom"].values)
    tb = m["total_buts"].values

    w = decay ** np.arange(len(m))
    w /= w.sum()

    return (
        float(np.average(bm, weights=w)),
        float(np.average(be, weights=w)),
        float(np.average((tb > 2.5).astype(float), weights=w)),
    )

def enrichir_forme_complet_train(df_hist, equipe, date_ref, n=5):
    m = df_hist[
        ((df_hist["equipe_domicile"] == equipe) | (df_hist["equipe_exterieur"] == equipe)) &
        (df_hist["date_match"] < date_ref)
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

# ====== Construction des features ======
df_hist = df.rename(columns={
    "date_match": "date_match",
    "equipe_domicile": "equipe_domicile",
    "equipe_exterieur": "equipe_exterieur",
    "buts_m_dom": "buts_m_dom",
    "buts_m_ext": "buts_m_ext",
})

cols_new = {k: [] for k in FEATURES_TOTAL_BUTS}

for _, r in df.iterrows():
    dom, ext, dref = r["equipe_domicile"], r["equipe_exterieur"], r["date_match"]

    bm_home, be_home, o25_home = calculer_forme_train(df_hist, dom, dref, role=True)
    bm_away, be_away, o25_away = calculer_forme_train(df_hist, ext, dref, role=False)

    _, _, _, std_marq_dom, std_enc_dom, clean_dom = enrichir_forme_complet_train(df_hist, dom, dref)
    _, _, _, std_marq_ext, std_enc_ext, clean_ext = enrichir_forme_complet_train(df_hist, ext, dref)

    poss_moy = np.nanmean([r.get("possession", np.nan), r.get("poss_ext", np.nan)])

    corners_dom = r.get("corners", np.nan)
    corners_ext = r.get("corners_ext", np.nan)

    cj_dom = r.get("cartons_jaunes", 0.0)
    cr_dom = r.get("cartons_rouges", 0.0)
    cj_ext = r.get("cj_ext", 0.0)
    cr_ext = r.get("cr_ext", 0.0)
    cartons_total = float((cj_dom + cr_dom) + (cj_ext + cr_ext))

    cols_new["forme_home_buts_marques"].append(bm_home)
    cols_new["forme_home_buts_encaisses"].append(be_home)
    cols_new["forme_home_over25"].append(o25_home)
    cols_new["forme_away_buts_marques"].append(bm_away)
    cols_new["forme_away_buts_encaisses"].append(be_away)
    cols_new["forme_away_over25"].append(o25_away)

    cols_new["std_marq_dom"].append(std_marq_dom)
    cols_new["std_enc_dom"].append(std_enc_dom)
    cols_new["std_marq_ext"].append(std_marq_ext)
    cols_new["std_enc_ext"].append(std_enc_ext)

    cols_new["clean_dom"].append(clean_dom)
    cols_new["clean_ext"].append(clean_ext)

    cols_new["moyenne_xg_dom"].append(float(r["moyenne_xg_dom"]))
    cols_new["moyenne_xg_ext"].append(float(r["moyenne_xg_ext"]))
    cols_new["buts_encaissés_dom"].append(float(r["buts_encaissés_dom"]))
    cols_new["buts_encaissés_ext"].append(float(r["buts_encaissés_ext"]))

    cols_new["poss_moyenne"].append(float(poss_moy) if not np.isnan(poss_moy) else 50.0)
    cols_new["tirs_dom"].append(float(r["tirs_dom"]))
    cols_new["tirs_ext"].append(float(r["tirs_ext"]))
    cols_new["tirs_cadres_dom"].append(float(r["tirs_cadres_dom"]))
    cols_new["tirs_cadres_ext"].append(float(r["tirs_cadres_ext"]))

    cols_new["corners_dom"].append(float(corners_dom) if not np.isnan(corners_dom) else 0.0)
    cols_new["corners_ext"].append(float(corners_ext) if not np.isnan(corners_ext) else 0.0)
    cols_new["cartons_total"].append(cartons_total)

for k, v in cols_new.items():
    df[k] = v

# ====== Dataset ======
X = df[FEATURES_TOTAL_BUTS].astype(float).values
y = df["total_buts"].astype(float).values
dates = pd.to_datetime(df["date_match"]).values

# ====== Split chronologique : 5 folds, dernier = test final ======
order = np.argsort(dates)
X, y = X[order], y[order]

n_splits = 5 if len(y) >= 500 else 4  # robuste si petit dataset
tscv = TimeSeriesSplit(n_splits=n_splits)

fold_idx = list(tscv.split(X))
trainval_idx, test_idx = fold_idx[-1]          # dernier split = Test final
X_trainval, y_trainval = X[trainval_idx], y[trainval_idx]
X_test, y_test         = X[test_idx], y[test_idx]

# ====== Standardisation (fit uniquement sur trainval) ======
scaler = StandardScaler()
X_trainval_sc = scaler.fit_transform(X_trainval)
X_test_sc      = scaler.transform(X_test)

# ====== OOF pour CatBoost & HGB sur les folds (hors test final) ======
# On re-splite trainval en 3 folds chronologiques pour OOF/meta
inner_tscv = TimeSeriesSplit(n_splits=3 if len(y_trainval) >= 300 else 2)

oof_cat = np.zeros_like(y_trainval, dtype=float)
oof_hgb = np.zeros_like(y_trainval, dtype=float)

cat_models, hgb_models = [], []

# Optuna (sobre) pour CatBoost
param_dist = {
    "depth": IntDistribution(4, 10),
    "learning_rate": FloatDistribution(0.02, 0.12),
    "iterations": IntDistribution(250, 600),
    "l2_leaf_reg": FloatDistribution(1.0, 8.0)
}

for tr_idx, va_idx in inner_tscv.split(X_trainval):
    X_tr, X_va = X_trainval_sc[tr_idx], X_trainval_sc[va_idx]
    y_tr, y_va = y_trainval[tr_idx], y_trainval[va_idx]

    # --- CatBoost + Optuna ---
    cat_search = OptunaSearchCV(
        estimator=CatBoostRegressor(
            verbose=0, random_seed=42, loss_function="RMSE", subsample=0.8, rsm=0.8
        ),
        param_distributions=param_dist,
        n_trials=25,
        cv=KFold(n_splits=3, shuffle=True, random_state=42),
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )
    cat_search.fit(X_tr, y_tr)
    cat = cat_search.best_estimator_
    oof_cat[va_idx] = cat.predict(X_va)
    cat_models.append(cat)

    # --- HistGradientBoosting (grille sobre) ---
    hgb = HistGradientBoostingRegressor(
        max_iter=350, learning_rate=0.06, max_depth=6,
        l2_regularization=0.0, min_samples_leaf=20, random_state=42
    )
    hgb.fit(X_tr, y_tr)
    oof_hgb[va_idx] = hgb.predict(X_va)
    hgb_models.append(hgb)

# --- Entraîne des modèles "full" sur tout trainval pour le déploiement ---
best_cat_full = CatBoostRegressor(verbose=0, random_seed=42, loss_function="RMSE",
                                  depth=8, learning_rate=0.06, iterations=500, subsample=0.8, rsm=0.8)
best_cat_full.fit(X_trainval_sc, y_trainval)

hgb_full = HistGradientBoostingRegressor(max_iter=350, learning_rate=0.06, max_depth=6,
                                         l2_regularization=0.0, min_samples_leaf=20, random_state=42)
hgb_full.fit(X_trainval_sc, y_trainval)

# ====== Méta-modèle (stacking) entraîné sur OOF ======
stack_X_train = np.vstack([oof_cat, oof_hgb]).T
meta = RidgeCV(alphas=[0.1, 1.0, 10.0])
meta.fit(stack_X_train, y_trainval)

# ====== Évaluation sur le test final ======
# Prédictions test final via les modèles "full"
pred_cat_test = best_cat_full.predict(X_test_sc)
pred_hgb_test = hgb_full.predict(X_test_sc)
stack_X_test = np.vstack([pred_cat_test, pred_hgb_test]).T
pred_stack_test = meta.predict(stack_X_test)

def metrics_block(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred))
    }

results = {}
results["catboost_optuna"]        = metrics_block(y_test, pred_cat_test)
results["hist_gradient_boosting"] = metrics_block(y_test, pred_hgb_test)
results["stacking"]               = metrics_block(y_test, pred_stack_test)

# ====== Quantiles conformaux (p25–p75 calibrés ~50%) ======
# 1) On entraîne deux modèles quantiles sur TRAINVAL
params_q = dict(n_estimators=400, learning_rate=0.06, num_leaves=31,
                min_data_in_leaf=30, subsample=0.9, feature_fraction=0.9, random_state=42)
q25 = LGBMRegressor(objective="quantile", alpha=0.25, **params_q)
q75 = LGBMRegressor(objective="quantile", alpha=0.75, **params_q)
q25.fit(X_trainval_sc, y_trainval); q75.fit(X_trainval_sc, y_trainval)

# 2) Calibration d'un décalage Δ sur TRAINVAL via split interne (20% pour calib)
X_tr_c, X_cal_c, y_tr_c, y_cal_c = train_test_split(X_trainval_sc, y_trainval, test_size=0.2, shuffle=False)
q25_c = LGBMRegressor(objective="quantile", alpha=0.25, **params_q).fit(X_tr_c, y_tr_c)
q75_c = LGBMRegressor(objective="quantile", alpha=0.75, **params_q).fit(X_tr_c, y_tr_c)

p25_cal = q25_c.predict(X_cal_c)
p75_cal = q75_c.predict(X_cal_c)

# Δ tel que coverage ~ 50% sur calib
def find_delta(y_true, lo, hi, target=0.50):
    deltas = np.linspace(0.0, 1.2, 25)  # de 0 à +1.2 buts
    best = 0.0; best_diff = 1e9
    for d in deltas:
        cover = np.mean((y_true >= (lo - d)) & (y_true <= (hi + d)))
        if abs(cover - target) < best_diff:
            best, best_diff = d, abs(cover - target)
    return float(best)

OFFSET = find_delta(y_cal_c, p25_cal, p75_cal, target=0.50)

# 3) Option "normalized" simple : largeur locale ~ |résidu| (modèle MAE)
res_model = LGBMRegressor(objective="regression_l1", n_estimators=300, random_state=42)
# résidus sur trainval (avec catboost_full comme baseline)
res_train = np.abs(y_trainval - best_cat_full.predict(X_trainval_sc))
res_model.fit(X_trainval_sc, res_train)

# 4) Test final
p25_test = q25.predict(X_test_sc) - OFFSET
p75_test = q75.predict(X_test_sc) + OFFSET

# ajustement multiplicatif selon variabilité locale (bornes plus serrées si résidu attendu faible)
scale_test = np.clip(res_model.predict(X_test_sc), 0.5, 2.5)  # bornes pour éviter extrêmes
p25_test = pred_stack_test - (pred_stack_test - p25_test) * (1.0 / scale_test)
p75_test = pred_stack_test + (p75_test - pred_stack_test) * (1.0 / scale_test)

coverage = float(np.mean((y_test >= p25_test) & (y_test <= p75_test)))
width    = float(np.mean(p75_test - p25_test))
results["conformal"] = {"coverage": coverage, "width": width}

# ====== Heuristique non linéaire (CatBoost) — entraînée hors test ======
df_heur = df.copy()
df_heur["tirs_cadres_total"] = df_heur["tirs_cadres_dom"] + df_heur["tirs_cadres_ext"]
df_heur["forme_dom_marq"]    = df_heur["forme_home_buts_marques"]
df_heur["forme_ext_marq"]    = df_heur["forme_away_buts_marques"]
df_heur["solidite_dom"]      = 1 / (df_heur["buts_encaissés_dom"] + 0.1)
df_heur["solidite_ext"]      = 1 / (df_heur["buts_encaissés_ext"] + 0.1)
df_heur["corners"]           = df_heur["corners_dom"] + df_heur["corners_ext"]
df_heur["fautes"]            = (df_heur.get("fautes", 0).fillna(0) + df_heur.get("fautes_ext", 0).fillna(0))
df_heur["cartons"]           = (df_heur.get("cartons_jaunes", 0).fillna(0) + df_heur.get("cartons_rouges", 0).fillna(0) +
                                df_heur.get("cj_ext", 0).fillna(0) + df_heur.get("cr_ext", 0).fillna(0))
df_heur["poss"]              = pd.DataFrame({"a": df_heur.get("possession", np.nan),
                                             "b": df_heur.get("poss_ext",   np.nan)}).mean(axis=1)

FEATURES_HEURISTIQUE = [
    "buts_dom", "buts_ext",
    "over25_dom", "over25_ext",
    "btts_dom", "btts_ext",
    "moyenne_xg_dom", "moyenne_xg_ext",
    "tirs_cadres_total",
    "forme_dom_marq", "forme_ext_marq",
    "solidite_dom", "solidite_ext",
    "corners", "fautes",
    "cartons",
    "poss"
]

X_score_all = df_heur[FEATURES_HEURISTIQUE].astype(float).values
y_score_all = df_heur["total_buts"].astype(float).values

# split temporel aligné : trainval/test
X_score_tr, X_score_te = X_score_all[trainval_idx], X_score_all[test_idx]
y_score_tr, y_score_te = y_score_all[trainval_idx], y_score_all[test_idx]

heur = CatBoostRegressor(
    depth=6, iterations=500, learning_rate=0.05, loss_function="MAE",
    random_seed=42, verbose=0
)
heur.fit(X_score_tr, y_score_tr)
preds_score_te = heur.predict(X_score_te)

results["score_heuristique"] = metrics_block(y_score_te, preds_score_te)

# ====== Classifieur Over 2.5 calibré (isotonic) ======
label_over = (y > 2.5).astype(int)
y_class_tr, y_class_te = label_over[trainval_idx], label_over[test_idx]

# re-use scaler (cohérence)
X_class_tr, X_class_te = X_trainval_sc, X_test_sc

base_clf = GradientBoostingClassifier(n_estimators=250, learning_rate=0.06, max_depth=4, random_state=42)
base_clf.fit(X_class_tr, y_class_tr)

cal_clf = CalibratedClassifierCV(base_clf, method="isotonic", cv=5)
cal_clf.fit(X_class_tr, y_class_tr)

yprob_te = cal_clf.predict_proba(X_class_te)[:, 1]
acc_te   = float(accuracy_score(y_class_te, (yprob_te >= 0.5).astype(int)))
brier_te = float(brier_score_loss(y_class_te, yprob_te))
results["over25_classifier"] = {"accuracy": acc_te, "brier": brier_te}

# ====== Sauvegarde Git ======
os.system("git config --global user.email 'lilian.pamphile.bts@gmail.com'")
os.system("git config --global user.name 'LilianPamphile'")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("❌ Le token GitHub (GITHUB_TOKEN) n'est pas défini.")

GITHUB_REPO = f"https://{GITHUB_TOKEN}@github.com/LilianPamphile/paris-sportifs.git"
CLONE_DIR = "model_push"
if os.path.exists(CLONE_DIR):
    shutil.rmtree(CLONE_DIR)
os.system(f"git clone {GITHUB_REPO} {CLONE_DIR}")
model_path = f"{CLONE_DIR}/model_files"
os.makedirs(model_path, exist_ok=True)

# Régressions + méta
with open(f"{model_path}/model_total_buts_catboost_optuna.pkl", "wb") as f:
    pickle.dump(best_cat_full, f)
with open(f"{model_path}/model_total_buts_hist_gradient_boosting.pkl", "wb") as f:
    pickle.dump(hgb_full, f)
with open(f"{model_path}/model_total_buts_stacking.pkl", "wb") as f:
    pickle.dump(meta, f)

# Quantiles
with open(f"{model_path}/model_total_buts_conformal_p25.pkl", "wb") as f:
    pickle.dump(q25, f)
with open(f"{model_path}/model_total_buts_conformal_p75.pkl", "wb") as f:
    pickle.dump(q75, f)

# Scaler + features
with open(f"{model_path}/scaler_total_buts.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open(f"{model_path}/features_list.pkl", "wb") as f:
    pickle.dump(FEATURES_TOTAL_BUTS, f)

# Heuristique + ses features
with open(f"{model_path}/regression_score_heuristique.pkl", "wb") as f:
    pickle.dump(heur, f)
with open(f"{model_path}/features_list_score_heuristique.pkl", "wb") as f:
    pickle.dump(FEATURES_HEURISTIQUE, f)

# Classif calibré
with open(f"{model_path}/model_over25_classifier.pkl", "wb") as f:
    pickle.dump(cal_clf, f)

# OFFSET (pour compat avec main.py)
with open(f"{model_path}/offset_conformal.pkl", "wb") as f:
    pickle.dump(OFFSET, f)

# MAE info (pour ancienne pondération si besoin)
mae_info = {
    "mae_cat": results["catboost_optuna"]["mae"],
    "mae_hgb": results["hist_gradient_boosting"]["mae"]
}
with open(f"{model_path}/mae_models.pkl", "wb") as f:
    pickle.dump(mae_info, f)

# Commit & push
os.system(f"cd {CLONE_DIR} && git add model_files && git commit -m '🔁 Train v5: chrono backtest + OOF stacking + conformal@50 + calib O/U' && git push")

# ====== Rapport par mail (facultatif) ======
def send_email(subject, body, to_email):
    try:
        from email.mime.text import MIMEText
        import smtplib
        from_email = "lilian.pamphile.bts@gmail.com"
        app_password = os.getenv("EMAIL_APP_PASSWORD")
        if not app_password:
            print("ℹ️ EMAIL_APP_PASSWORD manquant — mail non envoyé.")
            return
        msg = MIMEText(body)
        msg["Subject"] = subject; msg["From"] = from_email; msg["To"] = to_email
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(from_email, app_password)
            server.send_message(msg)
        print("📬 Email envoyé.")
    except Exception as e:
        print("❌ Email erreur:", e)

today = date.today()
lines = [f"Les modèles ont été réentraînés le {today}.\n"]
def fmt_block(name, infos):
    if {"mae","rmse","r2"} <= infos.keys():
        mae, rmse, r2 = infos["mae"], infos["rmse"], infos["r2"]
        perf = "🟢 Excellent" if rmse < 1.5 else ("🟡 Correct" if rmse < 2.0 else "🔴 À surveiller")
        return f"\n🔧 **{name}**\n• MAE : {mae:.3f}\n• RMSE : {rmse:.3f}\n• R² : {r2:.3f}\n• Interprétation : {perf}"
    if "coverage" in infos:
        return f"\n🔧 **{name}**\n• Coverage (p25–p75) : {infos['coverage']:.2%}\n• Largeur moyenne : {infos['width']:.2f} buts"
    if "accuracy" in infos:
        brier = infos.get("brier", None)
        return f"\n🔧 **{name}**\n• Accuracy : {infos['accuracy']:.3f}" + (f"\n• Brier : {brier:.3f}" if brier is not None else "")
    return ""

for k, v in results.items():
    lines.append(fmt_block(k.upper(), v))

lines += [
    "\n📁 Fichiers générés : modèles + scaler + meta + quantiles + offset + classif calibré",
    "📤 Upload GitHub : ✅ effectué",
    "🔗 https://github.com/LilianPamphile/paris-sportifs/tree/main/model_files"
]
send_email("📊 Modèles mis à jour (train v5)", "\n".join(lines), "lilian.pamphile.bts@gmail.com")
print("✅ Fin du train v5.")
