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
    "buts_dom", "buts_ext", "buts_encaissés_dom", "buts_encaissés_ext",
    "over25_dom", "over25_ext", "btts_dom", "btts_ext",
    "moyenne_xg_dom", "moyenne_xg_ext", "diff_xg", "sum_xg",
    "forme_dom_marq", "forme_dom_enc", "forme_dom_over25",
    "forme_ext_marq", "forme_ext_enc", "forme_ext_over25",
    "sum_btts", "diff_over25", "total_tirs", "total_tirs_cadres",
    "clean_sheets_dom", "clean_sheets_ext", "solidite_dom", "solidite_ext",
    "std_marq_dom", "std_enc_dom", "std_marq_ext", "std_enc_ext",
    "clean_dom", "clean_ext", "solidite_def_dom", "solidite_def_ext",
    "forme_pond_dom", "forme_pond_ext",
    "cartons", "poss", "tirs_cadres_total"
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

# Nouvelle fonction : forme pondérée + std + clean_sheets
def enrichir_forme(df_hist, equipe, date_ref):
    matchs = df_hist[((df_hist["equipe_domicile"] == equipe) | (df_hist["equipe_exterieur"] == equipe)) &
                     (df_hist["date_match"] < date_ref)].sort_values("date_match", ascending=False).head(5)

    if matchs.empty:
        return (0, 0, 0, 0, 0, 0)

    weights = np.linspace(1.0, 2.0, len(matchs))
    buts_marques = []
    buts_encaisses = []
    over25 = []
    clean_sheets = []

    for _, row in matchs.iterrows():
        est_dom = (row["equipe_domicile"] == equipe)
        bm = row["buts_m_dom"] if est_dom else row["buts_m_ext"]
        be = row["buts_m_ext"] if est_dom else row["buts_m_dom"]

        buts_marques.append(bm)
        buts_encaisses.append(be)
        over25.append(int(row["total_buts"] > 2.5))
        clean_sheets.append(int(be == 0))

    # Moyenne pondérée
    marq = np.average(buts_marques, weights=weights)
    enc = np.average(buts_encaisses, weights=weights)
    over = np.average(over25, weights=weights)
    std_marq = np.std(buts_marques)
    std_enc = np.std(buts_encaisses)
    clean = np.sum(clean_sheets)

    return marq, enc, over, std_marq, std_enc, clean

# Application
forme_dom_marq, forme_dom_enc, forme_dom_over25 = [], [], []
forme_ext_marq, forme_ext_enc, forme_ext_over25 = [], [], []
std_marq_dom, std_enc_dom, std_marq_ext, std_enc_ext = [], [], [], []
clean_dom, clean_ext = [], []

df_hist = df.copy()


for _, row in df.iterrows():
    fdm, fde, fdo25, stdm, stde, cldm = enrichir_forme(df_hist, row["equipe_domicile"], row["date_match"])
    fem, fee, feo25, stde2, stee2, clext = enrichir_forme(df_hist, row["equipe_exterieur"], row["date_match"])

    forme_dom_marq.append(fdm)
    forme_dom_enc.append(fde)
    forme_dom_over25.append(fdo25)
    std_marq_dom.append(stdm)
    std_enc_dom.append(stde)
    clean_dom.append(cldm)

    forme_ext_marq.append(fem)
    forme_ext_enc.append(fee)
    forme_ext_over25.append(feo25)
    std_marq_ext.append(stde2)
    std_enc_ext.append(stee2)
    clean_ext.append(clext)

df["forme_dom_marq"] = forme_dom_marq
df["forme_dom_enc"] = forme_dom_enc
df["forme_dom_over25"] = forme_dom_over25
df["forme_ext_marq"] = forme_ext_marq
df["forme_ext_enc"] = forme_ext_enc
df["forme_ext_over25"] = forme_ext_over25
df["std_marq_dom"] = std_marq_dom
df["std_enc_dom"] = std_enc_dom
df["std_marq_ext"] = std_marq_ext
df["std_enc_ext"] = std_enc_ext
df["clean_dom"] = clean_dom
df["clean_ext"] = clean_ext

# Solidité
df["solidite_def_dom"] = df["clean_dom"] / (df["forme_dom_enc"] + 1)
df["solidite_def_ext"] = df["clean_ext"] / (df["forme_ext_enc"] + 1)


# --- Variables croisées enrichies ---
df["diff_xg"] = df["moyenne_xg_dom"] - df["moyenne_xg_ext"]
df["sum_xg"] = df["moyenne_xg_dom"] + df["moyenne_xg_ext"]
df["sum_btts"] = df["btts_dom"] + df["btts_ext"]
df["diff_over25"] = df["over25_dom"] - df["over25_ext"]
df["total_tirs"] = df["tirs_dom"] + df["tirs_ext"]
df["total_tirs_cadres"] = df["tirs_cadres_dom"] + df["tirs_cadres_ext"]

# --- Nouvelles features défensives ---
df["clean_sheets_dom"] = 100 - df["btts_dom"]
df["clean_sheets_ext"] = 100 - df["btts_ext"]
df["solidite_dom"] = 1 / (df["buts_encaissés_dom"] + 0.1)
df["solidite_ext"] = 1 / (df["buts_encaissés_ext"] + 0.1)

# === 🆕 Nouvelles features à inclure dans les modèles
df["forme_pond_dom"] = 0.6 * df["forme_dom_marq"] + 0.4 * df["forme_dom_over25"]
df["forme_pond_ext"] = 0.6 * df["forme_ext_marq"] + 0.4 * df["forme_ext_over25"]

df["cartons"] = (df["cartons_jaunes"] + df.get("cartons_rouges", 0)).fillna(3)
df["poss"] = df[["possession", "poss_ext"]].mean(axis=1)

df["tirs_cadres_total"] = df["total_tirs_cadres"]  # Pour cohérence avec main

# --- Clip des outliers ---
df["total_buts"] = df["total_buts"].clip(upper=5)

# Enrichissements features + dérivées (inchangé)

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
preds = best_cat.predict(X_test)
results["catboost_optuna"] = {
    "mae": mean_absolute_error(y_test, preds),
    "rmse": np.sqrt(mean_squared_error(y_test, preds)),
    "r2": r2_score(y_test, preds)
}

# HistGradientBoosting
hgb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=6, random_state=42)
hgb.fit(X_train, y_train)
preds = hgb.predict(X_test)
results["hist_gradient_boosting"] = {
    "mae": mean_absolute_error(y_test, preds),
    "rmse": np.sqrt(mean_squared_error(y_test, preds)),
    "r2": r2_score(y_test, preds)
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
    "buts_dom", "buts_ext", "over25_dom", "over25_ext", "btts_dom", "btts_ext",
    "moyenne_xg_dom", "moyenne_xg_ext", "total_tirs_cadres",
    "forme_dom_marq", "forme_ext_marq",
    "solidite_dom", "solidite_ext",
    "corners", "fautes", "cartons_jaunes", "possession"
]

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
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score

# ✅ === 1. Ajout du label binaire over_25 ===
df["label_over_25"] = (df["total_buts"] > 2.5).astype(int)

# === Entraînement du classifieur ===
X_class = df[FEATURES_TOTAL_BUTS]
y_class = df["label_over_25"]
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

model_over25 = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
model_over25.fit(X_train_class, y_train_class)
y_prob = model_over25.predict_proba(X_test_class)[:, 1]
acc_over25 = accuracy_score(y_test_class, model_over25.predict(X_test_class))

# === Calibration plot optionnel ===
# prob_true, prob_pred = calibration_curve(y_test_class, y_prob, n_bins=10)

# Sauvegarde
with open(f"{model_path}/model_over25_classifier.pkl", "wb") as f:
    pickle.dump(model_over25, f)
with open(f"{model_path}/features_list_over25.pkl", "wb") as f:
    pickle.dump(FEATURES_TOTAL_BUTS, f)

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

# === Commit & Push GitHub ===
os.system(f"cd {CLONE_DIR} && git add model_files && git commit -m '🔁 Update models v3' && git push")
print("✅ Modèles commités et poussés sur GitHub.")

mae_info = {
    "mae_cat": mae_cat,   # valeur réelle obtenue dans ton script d'entraînement
    "mae_hgb": mae_hgb
}

with open("model_files/mae_models.pkl", "wb") as f:
    pickle.dump(mae_info, f)

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
