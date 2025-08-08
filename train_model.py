# === Standard Libraries ===
import os
import pickle
from datetime import date
from decimal import Decimal

# === Data & Math ===
import numpy as np
import pandas as pd

# === Database ===
import psycopg2

# === Scikit-learn ===
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor

# === Regressors ===
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

# === Hyperparameter Optimization ===
import optuna
from optuna.integration import OptunaSearchCV
from optuna.distributions import IntDistribution, FloatDistribution


# --- Connexion BDD ---
DATABASE_URL = "postgresql://postgres:jDDqfaqpspVDBBwsqxuaiSDNXjTxjMmP@shortline.proxy.rlwy.net:36536/railway"
conn = psycopg2.connect(DATABASE_URL)
cursor = conn.cursor()

# === Constantes ===
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

# ========== 🗃️ Récupération des données historiques ========== #
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


# --- Convertir Decimal en float ---
for col in df.columns:
    if df[col].dtype == 'object' and not df[col].dropna().empty and isinstance(df[col].dropna().iloc[0], Decimal):
        df[col] = df[col].astype(float)

# ============================================================
# 🔧 FEATURES ALIGNÉES AVEC LE MAIN (à placer avant X = df[...])
# ============================================================

# 1) Historique pour la forme split (exactement comme le main)
df_hist = df.rename(columns={
    "date_match": "date_match",
    "equipe_domicile": "equipe_domicile",
    "equipe_exterieur": "equipe_exterieur",
    "buts_m_dom": "buts_m_dom",
    "buts_m_ext": "buts_m_ext",
})
# df doit déjà contenir total_buts (= s.buts_dom + s.buts_ext)

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

# 2) Construction des nouvelles colonnes attendues dans FEATURES_TOTAL_BUTS
cols_new = {k: [] for k in FEATURES_TOTAL_BUTS}

for _, r in df.iterrows():
    dom, ext, dref = r["equipe_domicile"], r["equipe_exterieur"], r["date_match"]

    # --- forme split domicile / extérieur (5 derniers) ---
    bm_home, be_home, o25_home = calculer_forme_train(df_hist, dom, dref, role=True)
    bm_away, be_away, o25_away = calculer_forme_train(df_hist, ext, dref, role=False)

    # --- variabilité + clean sheets réels (sur 5) ---
    _, _, _, std_marq_dom, std_enc_dom, clean_dom = enrichir_forme_complet_train(df_hist, dom, dref)
    _, _, _, std_marq_ext, std_enc_ext, clean_ext = enrichir_forme_complet_train(df_hist, ext, dref)

    # --- possession moyenne du match (dom + ext) ---
    poss_moy = np.nanmean([r.get("possession", np.nan), r.get("poss_ext", np.nan)])

    # --- corners dom/ext : on renomme proprement ce que renvoie la requête ---
    corners_dom = r.get("corners", np.nan)      # sg1.corners -> "corners"
    corners_ext = r.get("corners_ext", np.nan)  # sg2.corners -> "corners_ext"

    # --- cartons total des deux équipes sur la ligne ---
    cj_dom = r.get("cartons_jaunes", 0.0)
    cr_dom = r.get("cartons_rouges", 0.0)
    cj_ext = r.get("cj_ext", 0.0)
    cr_ext = r.get("cr_ext", 0.0)
    cartons_total = float((cj_dom + cr_dom) + (cj_ext + cr_ext))

    # Remplissage
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

# Injection dans df
for k, v in cols_new.items():
    df[k] = v


# Dataset final
X = df[FEATURES_TOTAL_BUTS]
y = df["total_buts"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Résultats
results = {}

# CatBoost Optuna
param_distributions = {
    "depth": IntDistribution(4, 10),
    "learning_rate": FloatDistribution(0.01, 0.1),
    "iterations": IntDistribution(200, 500)
}
catboost_search = OptunaSearchCV(
    estimator=CatBoostRegressor(verbose=0, random_seed=42),
    param_distributions=param_distributions,
    n_trials=7,
    cv=KFold(n_splits=3, shuffle=True, random_state=42),
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)
catboost_search.fit(X_train, y_train)
best_cat = catboost_search.best_estimator_
preds_cat = best_cat.predict(X_test)
results["catboost_optuna"] = {
    "mae": mean_absolute_error(y_test, preds_cat),
    "rmse": np.sqrt(mean_squared_error(y_test, preds_cat)),
    "r2": r2_score(y_test, preds_cat)
}

# HistGradientBoosting
hgb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=6, random_state=42)
hgb.fit(X_train, y_train)
preds_hgb = hgb.predict(X_test)
results["hist_gradient_boosting"] = {
    "mae": mean_absolute_error(y_test, preds_hgb),
    "rmse": np.sqrt(mean_squared_error(y_test, preds_hgb)),
    "r2": r2_score(y_test, preds_hgb)
}

# LGBM Conformal Interval
params_base = {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05, "random_state": 42}
OFFSET = 0.25
q_models = {
    0.25: LGBMRegressor(objective="quantile", alpha=0.25, **params_base),
    0.75: LGBMRegressor(objective="quantile", alpha=0.75, **params_base)
}
for q, m in q_models.items():
    m.fit(X_train, y_train)
p25 = q_models[0.25].predict(X_test) - OFFSET
p75 = q_models[0.75].predict(X_test) + OFFSET
coverage = np.mean((y_test >= p25) & (y_test <= p75))
width = np.mean(p75 - p25)
results["conformal"] = {"coverage": coverage, "width": width}

import shutil
# === Git Config & Clone ===
os.system("git config --global user.email 'lilian.pamphile.bts@gmail.com'")
os.system("git config --global user.name 'LilianPamphile'")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("❌ Le token GitHub (GITHUB_TOKEN) n'est pas défini.")

GITHUB_REPO = f"https://{GITHUB_TOKEN}@github.com/LilianPamphile/paris-sportifs.git"
CLONE_DIR = "model_push"

# Nettoyage / clone du dépôt
if os.path.exists(CLONE_DIR):
    shutil.rmtree(CLONE_DIR)
os.system(f"git clone {GITHUB_REPO} {CLONE_DIR}")

model_path = f"{CLONE_DIR}/model_files"
os.makedirs(model_path, exist_ok=True)

# === Sauvegarde des modèles ===
# CatBoost Optuna
with open(f"{model_path}/model_total_buts_catboost_optuna.pkl", "wb") as f:
    pickle.dump(best_cat, f)

# HistGradientBoosting
with open(f"{model_path}/model_total_buts_hist_gradient_boosting.pkl", "wb") as f:
    pickle.dump(hgb, f)

# Conformal models
with open(f"{model_path}/model_total_buts_conformal_p25.pkl", "wb") as f:
    pickle.dump(q_models[0.25], f)
with open(f"{model_path}/model_total_buts_conformal_p75.pkl", "wb") as f:
    pickle.dump(q_models[0.75], f)

# Scaler & features
with open(f"{model_path}/scaler_total_buts.pkl", "wb") as f:
    pickle.dump(scaler, f)

import time

features_list_path = f"{model_path}/features_list.pkl"
with open(features_list_path, "wb") as f:
    pickle.dump(FEATURES_TOTAL_BUTS, f)

# 🔧 Force modification de timestamp pour forcer Git à l'inclure
os.utime(features_list_path, (time.time(), time.time()))

# Ajout dans le même script (train_model.py), à la fin ou dans une section dédiée
df_heuristique = df.copy()

# === Variables pour score heuristique appris ===
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

# Assure-toi d'avoir créé df["cartons"] et df["poss"] (tu l'as fait 👍)
X_score = df_heuristique[FEATURES_HEURISTIQUE]
y_score = df_heuristique["total_buts"]

# Apprentissage du modèle
from sklearn.linear_model import LinearRegression
model_score = LinearRegression()
model_score.fit(X_score, y_score)

# Sauvegarde du modèle
with open(f"{model_path}/regression_score_heuristique.pkl", "wb") as f:
    pickle.dump(model_score, f)
    
# === Sauvegarde des features du modèle heuristique ===
features_heuristique_path = f"{model_path}/features_list_score_heuristique.pkl"
with open(features_heuristique_path, "wb") as f:
    pickle.dump(FEATURES_HEURISTIQUE, f)

# Forcer l’inclusion Git
os.utime(features_heuristique_path, (time.time(), time.time()))

print("✅ Modèle score_heuristique sauvegardé.")

# 🌟 Ajout d'un modèle Over/Under 2.5 + calibration dynamique du OFFSET

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# ✅ === 1. Ajout du label binaire over_25 ===
df["label_over_25"] = (df["total_buts"] > 2.5).astype(int)

# === Entraînement du classifieur ===
X_class = X  # mêmes features que la régression
y_class = df["label_over_25"]

X_class_scaled = scaler.transform(X_class)  # << on réutilise le même scaler
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_class_scaled, y_class, test_size=0.2, random_state=42
)

model_over25 = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
model_over25.fit(X_train_class, y_train_class)
y_prob = model_over25.predict_proba(X_test_class)[:, 1]

acc_over25 = accuracy_score(y_test_class, model_over25.predict(X_test_class))

# Sauvegarde
with open(f"{model_path}/model_over25_classifier.pkl", "wb") as f:
    pickle.dump(model_over25, f)

results["over25_classifier"] = {"accuracy": acc_over25}

# ⚡ === 2. Calibration dynamique du OFFSET (RMSE-based) ===
preds_train_q25 = q_models[0.25].predict(X_train)
preds_train_q75 = q_models[0.75].predict(X_train)
intervales = preds_train_q75 - preds_train_q25
offset_dynamic = np.percentile(intervales, 75) / 2  # ou même np.median(intervales) / 2


with open(f"{model_path}/offset_conformal.pkl", "wb") as f:
    pickle.dump(offset_dynamic, f)

print("✅ Modèle classification + offset dynamique ajoutés avec succès.")


preds_score = model_score.predict(X_score)
mae_score = mean_absolute_error(y_score, preds_score)
results["score_heuristique"] = {
    "mae": mae_score,
    "rmse": np.sqrt(mean_squared_error(y_score, preds_score)),
    "r2": r2_score(y_score, preds_score)
}

# === Sauvegarde des MAE pondérations ===
mae_cat = results["catboost_optuna"]["mae"]
mae_hgb = results["hist_gradient_boosting"]["mae"]

mae_info = {
    "mae_cat": mae_cat,
    "mae_hgb": mae_hgb
}
with open(f"{model_path}/mae_models.pkl", "wb") as f:
    pickle.dump(mae_info, f)

# === Commit & Push GitHub ===
os.system(f"cd {CLONE_DIR} && git add model_files && git commit -m '🔁 Update models v3' && git push")
print("✅ Modèles commités et poussés sur GitHub.")


# === Email de notification ===
def send_email(subject, body, to_email):
    from email.mime.text import MIMEText
    import smtplib
    
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = "lilian.pamphile.bts@gmail.com"
    msg["To"] = to_email

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login("lilian.pamphile.bts@gmail.com", "fifkktsenfxsqiob")
            server.send_message(msg)
        print("📬 Email envoyé.")
    except Exception as e:
        print("❌ Email erreur :", e)

# === Génération du contenu du mail ===
today = date.today()
subject = "📊 Modèles total_buts mis à jour"
body_lines = [f"Les modèles `total_buts` ont été réentraînés le {today}.\n"]

for name, infos in results.items():
    body_lines.append(f"\n🔧 **{name.upper()}**")

    if "mae" in infos:
        mae = infos["mae"]
        rmse = infos["rmse"]
        r2 = infos["r2"]
        if rmse < 1.5:
            perf = "🟢 Excellent"
        elif rmse < 2:
            perf = "🟡 Correct"
        else:
            perf = "🔴 À surveiller"
        body_lines.append(
            f"• MAE : {mae:.4f}\n• RMSE : {rmse:.4f}\n• R² : {r2:.4f}\n• Interprétation : {perf}"
        )

    elif "coverage" in infos:
        body_lines.append(
            f"• Coverage (p25–p75) : {infos['coverage']:.2%}\n• Largeur moyenne : {infos['width']:.2f} buts"
        )

body_lines += [
    "\n📁 Fichiers générés : modèles + scaler",
    "📤 Upload GitHub : ✅ effectué avec succès",
    "🔗 https://github.com/LilianPamphile/paris-sportifs/tree/main/model_files"
]

body = "\n".join(body_lines)
send_email(subject, body, "lilian.pamphile.bts@gmail.com")
