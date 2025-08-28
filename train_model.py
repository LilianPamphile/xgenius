# === Standard Libraries ===
import os
import pickle
from datetime import date
from decimal import Decimal
import time
import shutil

# === Data & Math ===
import numpy as np
import pandas as pd

# === Database ===
import psycopg2

# === Scikit-learn ===
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, brier_score_loss
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import RidgeCV
from sklearn.calibration import CalibratedClassifierCV

# === Other Regressors ===
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

# === Optuna ===
import optuna
from optuna.integration import OptunaSearchCV
from optuna.distributions import IntDistribution, FloatDistribution


# --- Connexion BDD ---
DATABASE_URL = "postgresql://postgres:jDDqfaqpspVDBBwsqxuaiSDNXjTxjMmP@shortline.proxy.rlwy.net:36536/railway"
conn = psycopg2.connect(DATABASE_URL)
cursor = conn.cursor()

# === Features finales utilisées pour la régression de buts (inchangées) ===
FEATURES_TOTAL_BUTS = [
    # Forme split
    "forme_home_buts_marques", "forme_home_buts_encaisses", "forme_home_over25",
    "forme_away_buts_marques", "forme_away_buts_encaisses", "forme_away_over25",

    # Variabilité
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

# === Requête (identique à ta version) ===
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
df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
conn.close()

# --- Convertir Decimal en float si nécessaire ---
for col in df.columns:
    if df[col].dtype == 'object' and not df[col].dropna().empty and isinstance(df[col].dropna().iloc[0], Decimal):
        df[col] = df[col].astype(float)

# === Historique pour forme (identique à ta logique) ===
df_hist = df.rename(columns={
    "date_match": "date_match",
    "equipe_domicile": "equipe_domicile",
    "equipe_exterieur": "equipe_exterieur",
    "buts_m_dom": "buts_m_dom",
    "buts_m_ext": "buts_m_ext",
})

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

# === Construction des colonnes de FEATURES_TOTAL_BUTS ===
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

# === Dataset régression (y = total_buts) ===
X = df[FEATURES_TOTAL_BUTS].copy()
y = df["total_buts"].astype(float).copy()

# === Split chronologique (anti-fuite) ===
order = np.argsort(df["date_match"].values)
X_sorted = X.values[order]
y_sorted = y.values[order]

split_idx = int(0.80 * len(y_sorted))
X_tr, X_te = X_sorted[:split_idx], X_sorted[split_idx:]
y_tr, y_te = y_sorted[:split_idx], y_sorted[split_idx:]

# === Standardisation (fit uniquement sur train) ===
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc = scaler.transform(X_te)

results = {}

# === CatBoost (Optuna) sur train ===
param_distributions = {
    "depth": IntDistribution(4, 10),
    "learning_rate": FloatDistribution(0.01, 0.1),
    "iterations": IntDistribution(200, 500)
}
catboost_search = OptunaSearchCV(
    estimator=CatBoostRegressor(verbose=0, random_seed=42),
    param_distributions=param_distributions,
    n_trials=10,
    cv=KFold(n_splits=3, shuffle=True, random_state=42),
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)
catboost_search.fit(X_tr_sc, y_tr)
best_cat = catboost_search.best_estimator_
pred_cat_te = best_cat.predict(X_te_sc)

results["catboost_optuna"] = {
    "mae": mean_absolute_error(y_te, pred_cat_te),
    "rmse": np.sqrt(mean_squared_error(y_te, pred_cat_te)),
    "r2": r2_score(y_te, pred_cat_te)
}

# === HistGradientBoosting ===
hgb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=6, random_state=42)
hgb.fit(X_tr_sc, y_tr)
pred_hgb_te = hgb.predict(X_te_sc)

results["hist_gradient_boosting"] = {
    "mae": mean_absolute_error(y_te, pred_hgb_te),
    "rmse": np.sqrt(mean_squared_error(y_te, pred_hgb_te)),
    "r2": r2_score(y_te, pred_hgb_te)
}

# === STACKING (RidgeCV) ===
pred_cat_tr = best_cat.predict(X_tr_sc)
pred_hgb_tr = hgb.predict(X_tr_sc)
stack_X_tr = np.vstack([pred_cat_tr, pred_hgb_tr]).T
stack_X_te = np.vstack([pred_cat_te, pred_hgb_te]).T

meta = RidgeCV(alphas=[0.1, 1.0, 10.0])
meta.fit(stack_X_tr, y_tr)
pred_stack_te = meta.predict(stack_X_te)

results["stacking"] = {
    "mae": mean_absolute_error(y_te, pred_stack_te),
    "rmse": np.sqrt(mean_squared_error(y_te, pred_stack_te)),
    "r2": r2_score(y_te, pred_stack_te)
}

# === Quantiles p25/p75 (fit sur train uniquement) ===
params_base = {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05, "random_state": 42}
q25_model = LGBMRegressor(objective="quantile", alpha=0.25, **params_base)
q75_model = LGBMRegressor(objective="quantile", alpha=0.75, **params_base)

q25_model.fit(X_tr_sc, y_tr)
q75_model.fit(X_tr_sc, y_tr)

p25_te = q25_model.predict(X_te_sc)
p75_te = q75_model.predict(X_te_sc)

# OFFSET dynamique basé sur la largeur des intervalles sur TRAIN
p25_tr = q25_model.predict(X_tr_sc)
p75_tr = q75_model.predict(X_tr_sc)
train_width = p75_tr - p25_tr
OFFSET = float(np.percentile(train_width, 75) / 2.0)

p25_adj = p25_te - OFFSET
p75_adj = p75_te + OFFSET
coverage = np.mean((y_te >= p25_adj) & (y_te <= p75_adj))
width = float(np.mean(p75_adj - p25_adj))

results["conformal"] = {"coverage": coverage, "width": width}

# === Heuristique non linéaire (CatBoost reg) ===
# Variables heuristiques (identiques à ta version + cohérence main.py)
df_heur = df.copy()
df_heur["tirs_cadres_total"] = df_heur["tirs_cadres_dom"] + df_heur["tirs_cadres_ext"]
df_heur["forme_dom_marq"] = df_heur["forme_home_buts_marques"]
df_heur["forme_ext_marq"] = df_heur["forme_away_buts_marques"]
df_heur["solidite_dom"] = 1 / (df_heur["buts_encaissés_dom"] + 0.1)
df_heur["solidite_ext"] = 1 / (df_heur["buts_encaissés_ext"] + 0.1)
df_heur["corners"] = df_heur["corners_dom"] + df_heur["corners_ext"]

# fautes + cartons + poss : robustesse aux colonnes manquantes
df_heur["fautes"] = (df_heur.get("fautes", 0).fillna(0) +
                     df_heur.get("fautes_ext", 0).fillna(0))
df_heur["cartons"] = (df_heur.get("cartons_jaunes", 0).fillna(0) +
                      df_heur.get("cartons_rouges", 0).fillna(0) +
                      df_heur.get("cj_ext", 0).fillna(0) +
                      df_heur.get("cr_ext", 0).fillna(0))
df_heur["poss"] = pd.DataFrame({
    "p1": df_heur.get("possession", np.nan),
    "p2": df_heur.get("poss_ext", np.nan)
}).mean(axis=1)

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

X_score = df_heur[FEATURES_HEURISTIQUE].values
y_score = df_heur["total_buts"].astype(float).values

heur = CatBoostRegressor(
    depth=6, iterations=400, learning_rate=0.05,
    loss_function="MAE", random_seed=42, verbose=0
)
heur.fit(X_score, y_score)

preds_score = heur.predict(X_score)
results["score_heuristique"] = {
    "mae": mean_absolute_error(y_score, preds_score),
    "rmse": np.sqrt(mean_squared_error(y_score, preds_score)),
    "r2": r2_score(y_score, preds_score)
}

# === Classifieur Over 2.5 + Calibration Isotonic ===
df["label_over_25"] = (df["total_buts"] > 2.5).astype(int)
X_class = X_sorted  # mêmes features que régression, déjà ordonnées
y_class = df["label_over_25"].values[order]

X_trc, X_tec = X_class[:split_idx], X_class[split_idx:]
y_trc, y_tec = y_class[:split_idx], y_class[split_idx:]

# On réutilise le scaler régression pour rester cohérent
X_trc_sc = scaler.transform(X_trc)
X_tec_sc = scaler.transform(X_tec)

base_clf = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42
)
base_clf.fit(X_trc_sc, y_trc)

cal_clf = CalibratedClassifierCV(base_clf, method="isotonic", cv=5)
cal_clf.fit(X_trc_sc, y_trc)

yprob_te = cal_clf.predict_proba(X_tec_sc)[:, 1]
acc_over25 = accuracy_score(y_tec, (yprob_te >= 0.5).astype(int))
brier = brier_score_loss(y_tec, yprob_te)

results["over25_classifier"] = {
    "accuracy": acc_over25,
    "brier": brier
}

# === Sauvegarde GitHub ===
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

# — Régression: modèles + meta
with open(f"{model_path}/model_total_buts_catboost_optuna.pkl", "wb") as f:
    pickle.dump(best_cat, f)

with open(f"{model_path}/model_total_buts_hist_gradient_boosting.pkl", "wb") as f:
    pickle.dump(hgb, f)

with open(f"{model_path}/model_total_buts_stacking.pkl", "wb") as f:
    pickle.dump(meta, f)

# — Quantiles
with open(f"{model_path}/model_total_buts_conformal_p25.pkl", "wb") as f:
    pickle.dump(q25_model, f)
with open(f"{model_path}/model_total_buts_conformal_p75.pkl", "wb") as f:
    pickle.dump(q75_model, f)

# — Scaler & features
with open(f"{model_path}/scaler_total_buts.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open(f"{model_path}/features_list.pkl", "wb") as f:
    pickle.dump(FEATURES_TOTAL_BUTS, f)

# — Heuristique non linéaire
with open(f"{model_path}/regression_score_heuristique.pkl", "wb") as f:
    pickle.dump(heur, f)
with open(f"{model_path}/features_list_score_heuristique.pkl", "wb") as f:
    pickle.dump(FEATURES_HEURISTIQUE, f)

# — Classifieur calibré
with open(f"{model_path}/model_over25_classifier.pkl", "wb") as f:
    pickle.dump(cal_clf, f)

# — OFFSET dynamique
with open(f"{model_path}/offset_conformal.pkl", "wb") as f:
    pickle.dump(OFFSET, f)

# — MAE modèles de base (pour info)
mae_info = {
    "mae_cat": results["catboost_optuna"]["mae"],
    "mae_hgb": results["hist_gradient_boosting"]["mae"]
}
with open(f"{model_path}/mae_models.pkl", "wb") as f:
    pickle.dump(mae_info, f)

# Commit & push
os.system(f"cd {CLONE_DIR} && git add model_files && git commit -m '🔁 Train v4: stacking + calibration + anti-leak' && git push")
print("✅ Modèles commités et poussés sur GitHub.")

# === Email de notification ===
def send_email(subject, body, to_email):
    from email.mime.text import MIMEText
    import smtplib

    from_email = "lilian.pamphile.bts@gmail.com"
    app_password = os.getenv("EMAIL_APP_PASSWORD")  # GitHub Secret

    if not app_password:
        print("❌ EMAIL_APP_PASSWORD non défini dans l'environnement.")
        return

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(from_email, app_password)
            server.send_message(msg)
        print("📬 Email envoyé.")
    except Exception as e:
        print("❌ Email erreur :", e)

# === Rapport
today = date.today()
lines = [f"Les modèles ont été réentraînés le {today}.\n"]
for name, infos in results.items():
    lines.append(f"\n🔧 **{name.upper()}**")
    if "mae" in infos:
        mae, rmse, r2 = infos["mae"], infos["rmse"], infos["r2"]
        perf = "🟢 Excellent" if rmse < 1.5 else ("🟡 Correct" if rmse < 2.0 else "🔴 À surveiller")
        lines.append(f"• MAE : {mae:.3f}\n• RMSE : {rmse:.3f}\n• R² : {r2:.3f}\n• Interprétation : {perf}")
    elif "coverage" in infos:
        lines.append(f"• Coverage (p25–p75) : {infos['coverage']:.2%}\n• Largeur moyenne : {infos['width']:.2f} buts")
    elif "accuracy" in infos:
        lines.append(f"• Accuracy : {infos['accuracy']:.3f}")
        if "brier" in infos:
            lines.append(f"• Brier : {infos['brier']:.3f}")

lines += [
    "\n📁 Fichiers générés : modèles + scaler + meta + quantiles + offset + classif calibré",
    "📤 Upload GitHub : ✅ effectué",
    "🔗 https://github.com/LilianPamphile/paris-sportifs/tree/main/model_files"
]
send_email("📊 Modèles mis à jour (train v4)", "\n".join(lines), "lilian.pamphile.bts@gmail.com")
print("✅ Fin du train.")
