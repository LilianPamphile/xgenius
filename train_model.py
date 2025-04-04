import psycopg2
import pandas as pd
import pickle
from decimal import Decimal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score, silhouette_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import numpy as np
import os
from datetime import date

# --- Connexion BDD ---
DATABASE_URL = "postgresql://postgres:jDDqfaqpspVDBBwsqxuaiSDNXjTxjMmP@shortline.proxy.rlwy.net:36536/railway"
conn = psycopg2.connect(DATABASE_URL)
cursor = conn.cursor()

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

# ========== 🗃️ Récupération des données historiques ========== #


# --- Convertir Decimal en float ---
for col in df.columns:
    if df[col].dtype == 'object' and not df[col].dropna().empty and isinstance(df[col].dropna().iloc[0], Decimal):
        df[col] = df[col].astype(float)

# --- Calcul forme récente ---
def calculer_forme(equipe, date_ref, df_hist):
    matchs = df_hist[((df_hist["equipe_domicile"] == equipe) | (df_hist["equipe_exterieur"] == equipe)) & (df_hist["date_match"] < date_ref)].sort_values("date_match", ascending=False).head(5)
    if matchs.empty:
        return 0, 0, 0
    buts_marques = np.mean([row["buts_m_dom"] if row["equipe_domicile"] == equipe else row["buts_m_ext"] for _, row in matchs.iterrows()])
    buts_encaisses = np.mean([row["buts_m_ext"] if row["equipe_domicile"] == equipe else row["buts_m_dom"] for _, row in matchs.iterrows()])
    over25 = np.mean([row["total_buts"] > 2.5 for _, row in matchs.iterrows()])
    return buts_marques, buts_encaisses, over25

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

# --- Clip des outliers ---
df["total_buts"] = df["total_buts"].clip(upper=5)

# Features finales (neutres, attaque + défense)
features = [
    "buts_dom", "buts_ext", "buts_encaissés_dom", "buts_encaissés_ext",
    "over25_dom", "over25_ext", "btts_dom", "btts_ext",
    "moyenne_xg_dom", "moyenne_xg_ext", "diff_xg", "sum_xg",
    "forme_dom_marq", "forme_dom_enc", "forme_dom_over25",
    "forme_ext_marq", "forme_ext_enc", "forme_ext_over25",
    "sum_btts", "diff_over25", "total_tirs", "total_tirs_cadres",
    "clean_sheets_dom", "clean_sheets_ext", "solidite_dom", "solidite_ext",
    "std_marq_dom", "std_enc_dom", "std_marq_ext", "std_enc_ext",
    "clean_dom", "clean_ext", "solidite_def_dom", "solidite_def_ext"
]

X = df[features]
y = df["total_buts"]

# Standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- KMEANS --- #
features_kmeans = [
    # 🔒 Défense et solidité
    "forme_dom_enc", "forme_ext_enc",
    "std_enc_dom", "std_enc_ext",
    "solidite_dom", "solidite_ext",
    "clean_sheets_dom", "clean_sheets_ext",
    
    # 💥 Menace offensive potentielle
    "diff_xg", "sum_xg",
    "total_tirs", "total_tirs_cadres",
    
    # ⚖️ Caractéristiques croisées utiles
    "diff_over25", "sum_btts",
]

X_kmeans = df[features_kmeans]
scaler_kmeans = StandardScaler()
X_kmeans_scaled = scaler_kmeans.fit_transform(X_kmeans)

kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(X_kmeans_scaled)

df["cluster_type"] = cluster_labels

# ♻️ Re-standardise avec la nouvelle colonne
X_scaled = scaler.fit_transform(X)

# 📊 Logging pour vérification
unique, counts = np.unique(cluster_labels, return_counts=True)
for label, count in zip(unique, counts):
    print(f"🔍 Cluster {label} → {count} matchs ({(count / len(cluster_labels)):.1%})")

sil_score = silhouette_score(X_kmeans_scaled, cluster_labels)
results["kmeans"] = {
    "model": kmeans,
    "silhouette": sil_score
}
print(f"✅ Silhouette Score (k=3 enrichi) : {sil_score:.4f}")

# --- FIN KMEANS --- #

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modèles
models = {
    "catboost": CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6, random_seed=42, verbose=0),
    "lightgbm": LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42),
    "xgboost": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42)
}

# RandomForest pour Simulation Monte Carlo
rf_simul = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
rf_simul.fit(X_train, y_train)

quantiles = [0.25, 0.75]
q_models = {}
for q in quantiles:
    model_q = LGBMRegressor(objective="quantile", alpha=q, n_estimators=200, max_depth=6, learning_rate=0.05)
    model_q.fit(X_train, y_train)
    q_models[q] = model_q

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    results[name] = {
        "model": model,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }

# Ajout des modèles quantiles au dico
results["quantile_p25"] = (q_models[0.25], None, None)
results["quantile_p75"] = (q_models[0.75], None, None)
results["rf_simul"] = (rf_simul, None, None)

# Quantile Evaluation
pred_p25 = q_models[0.25].predict(X_test)
pred_p75 = q_models[0.75].predict(X_test)

# ✅ Calibration simple des bornes (tu peux ajuster l'offset)
OFFSET = 0.25  # entre 0.2 et 0.4 selon largeur moyenne

pred_p25_adj = pred_p25 - OFFSET
pred_p75_adj = pred_p75 + OFFSET

# Recalcul coverage & largeur
coverage = np.mean((y_test >= pred_p25_adj) & (y_test <= pred_p75_adj))
width = np.mean(pred_p75_adj - pred_p25_adj)

results["quantile"] = {
    "model": (q_models[0.25], q_models[0.75]),
    "coverage": coverage,
    "width": width
}

# Simulation Monte Carlo
sim_preds = [rf_simul.predict(X_test + np.random.normal(0, 0.1, X_test.shape)) for _ in range(100)]
sim_preds = np.array(sim_preds)
sim_mean = np.mean(sim_preds)
sim_std = np.std(sim_preds)

results["rf_simul"] = {
    "model": rf_simul,
    "mean": sim_mean,
    "std": sim_std
}


# Git config
os.system("git config --global user.email 'lilian.pamphile.bts@gmail.com'")
os.system("git config --global user.name 'LilianPamphile'")

# Token
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("❌ Le token GitHub n'est pas défini.")

GITHUB_REPO = f"https://{GITHUB_TOKEN}@github.com/LilianPamphile/paris-sportifs.git"
CLONE_DIR = "model_push"

os.system(f"rm -rf {CLONE_DIR}")
os.system(f"git clone {GITHUB_REPO} {CLONE_DIR}")

model_path = f"{CLONE_DIR}/model_files"
os.makedirs(model_path, exist_ok=True)

# Sauvegarde
for name, infos in results.items():
    model = infos["model"] if isinstance(infos, dict) and "model" in infos else None
    if model:
        # 🔁 Cas spécial : quantile p25 / p75
        if name == "quantile":
            with open(f"{model_path}/model_total_buts_quantile_p25.pkl", "wb") as f:
                pickle.dump(q_models[0.25], f)
            with open(f"{model_path}/model_total_buts_quantile_p75.pkl", "wb") as f:
                pickle.dump(q_models[0.75], f)
            continue  # on saute ce cas pour pas l'écraser dans la suite

        with open(f"{model_path}/model_total_buts_{name}.pkl", "wb") as f:
            pickle.dump(model, f)

with open(f"{model_path}/scaler_total_buts.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open(f"{model_path}/features_list.pkl", "wb") as f:
    pickle.dump(features, f)

# --- KMEANS ---#
with open(f"{model_path}/features_kmeans_list.pkl", "wb") as f:
    pickle.dump(features_kmeans, f)

with open(f"{model_path}/scaler_kmeans.pkl", "wb") as f:
    pickle.dump(scaler_kmeans, f)

with open(f"{model_path}/kmeans_cluster.pkl", "wb") as f:
    pickle.dump(kmeans, f)



# Commit + push
os.system(f"cd {CLONE_DIR} && git add model_files && git commit -m '🔁 Update models v3' && git push")
print("✅ Modèles commités et poussés sur GitHub.")

# Email notif
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
        print("❌ Email erreur:", e)

# === Génération contenu du mail ===
today = date.today()

subject = "📊 Modèles total_buts mis à jour"
body_lines = [f"Les modèles `total_buts` ont été réentraînés le {today}.\n"]

for name, infos in results.items():
    body_lines.append(f"\n🔧 **{name.upper()}**")

    if name in ["catboost", "lightgbm", "xgboost"]:
        mae = infos["mae"]
        rmse = infos["rmse"]
        r2 = infos["r2"]
        if rmse < 1.8:
            perf = "🟢 Excellent"
        elif rmse < 2.2:
            perf = "🟡 Correct"
        else:
            perf = "🔴 À surveiller"
        body_lines.append(
            f"\n• MAE : {mae:.4f}\n• RMSE : {rmse:.4f}\n• R² : {r2:.4f}\n• Interprétation : {perf}"
        )

    elif name == "quantile":
        body_lines.append(
            f"\n• Coverage (p25–p75) : {infos['coverage']:.2%}\n• Largeur moyenne : {infos['width']:.2f} buts"
        )

    elif name == "rf_simul":
        body_lines.append(
            f"\n• Moyenne (simulée) : {infos['mean']:.2f} buts\n• Écart-type : {infos['std']:.2f}"
        )

    elif name == "kmeans":
        body_lines.append(
            f"\n• Silhouette Score (clustering matchs) : {infos['silhouette']:.4f}"
        )


body_lines += [
    "\n📁 Fichiers générés : modèles + scaler",
    "📤 Upload GitHub : ✅ effectué avec succès",
    "🔗 https://github.com/LilianPamphile/paris-sportifs/tree/main/model_files"
]

body = "\n".join(body_lines)

send_email(subject, body, "lilian.pamphile.bts@gmail.com")
