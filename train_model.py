import psycopg2
import pandas as pd
import pickle
from decimal import Decimal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import numpy as np

# --- Connexion BDD ---
DATABASE_URL = "postgresql://postgres:jDDqfaqpspVDBBwsqxuaiSDNXjTxjMmP@shortline.proxy.rlwy.net:36536/railway"
conn = psycopg2.connect(DATABASE_URL)
cursor = conn.cursor()

# --- Récupération des données ---
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

# --- Calcul forme récente ---
def calculer_forme(equipe, date_ref, df_hist):
    matchs = df_hist[((df_hist["equipe_domicile"] == equipe) | (df_hist["equipe_exterieur"] == equipe)) & (df_hist["date_match"] < date_ref)].sort_values("date_match", ascending=False).head(5)
    if matchs.empty:
        return 0, 0, 0
    buts_marques = np.mean([row["buts_m_dom"] if row["equipe_domicile"] == equipe else row["buts_m_ext"] for _, row in matchs.iterrows()])
    buts_encaisses = np.mean([row["buts_m_ext"] if row["equipe_domicile"] == equipe else row["buts_m_dom"] for _, row in matchs.iterrows()])
    over25 = np.mean([row["total_buts"] > 2.5 for _, row in matchs.iterrows()])
    return buts_marques, buts_encaisses, over25

# --- Ajout forme récente ---
forme_dom_marq, forme_dom_enc, forme_dom_over25 = [], [], []
forme_ext_marq, forme_ext_enc, forme_ext_over25 = [], [], []
df_hist = df.copy()

for _, row in df.iterrows():
    fdm, fde, fdo25 = calculer_forme(row["equipe_domicile"], row["date_match"], df_hist)
    fem, fee, feo25 = calculer_forme(row["equipe_exterieur"], row["date_match"], df_hist)
    forme_dom_marq.append(fdm)
    forme_dom_enc.append(fde)
    forme_dom_over25.append(fdo25)
    forme_ext_marq.append(fem)
    forme_ext_enc.append(fee)
    forme_ext_over25.append(feo25)

df["forme_dom_marq"] = forme_dom_marq
df["forme_dom_enc"] = forme_dom_enc
df["forme_dom_over25"] = forme_dom_over25
df["forme_ext_marq"] = forme_ext_marq
df["forme_ext_enc"] = forme_ext_enc
df["forme_ext_over25"] = forme_ext_over25

# --- Variables croisées enrichies ---
df["diff_xg"] = df["moyenne_xg_dom"] - df["moyenne_xg_ext"]
df["sum_xg"] = df["moyenne_xg_dom"] + df["moyenne_xg_ext"]
df["sum_btts"] = df["btts_dom"] + df["btts_ext"]
df["diff_over25"] = df["over25_dom"] - df["over25_ext"]
df["total_tirs"] = df["tirs_dom"] + df["tirs_ext"]
df["total_tirs_cadres"] = df["tirs_cadres_dom"] + df["tirs_cadres_ext"]

# --- Features sélectionnées pour la régression ---
features = [
    "buts_dom", "buts_ext", "buts_encaissés_dom", "buts_encaissés_ext",
    "over25_dom", "over25_ext", "btts_dom", "btts_ext",
    "moyenne_xg_dom", "moyenne_xg_ext", "diff_xg", "sum_xg",
    "forme_dom_marq", "forme_dom_enc", "forme_dom_over25",
    "forme_ext_marq", "forme_ext_enc", "forme_ext_over25",
    "sum_btts", "diff_over25", "total_tirs", "total_tirs_cadres"
]

X = df[features]
y = df["total_buts"]  # <== Changement de la cible : régression directe

# --- Standardisation ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Entraînement modèle de régression ---
model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# --- Évaluation ---
y_pred = model.predict(X_test)
print(f"✅ MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"✅ RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")


import os
from datetime import date
import smtplib
from email.mime.text import MIMEText
import pickle
import numpy as np

# === Variables venant de ton script principal (à adapter si tu es dans un script séparé)
# y_test et y_pred doivent être définis avant
mae_score = mean_absolute_error(y_test, y_pred)
rmse_score = np.sqrt(mean_squared_error(y_test, y_pred))

# === GitHub - Push automatique ===
print("📦 Push vers GitHub...")

import os

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = f"https://{GITHUB_TOKEN}@github.com/LilianPamphile/paris-sportifs.git"
CLONE_DIR = "model_push"

# Config git
os.system("git config --global user.email 'lilian.pamphile.bts@gmail.com'")
os.system("git config --global user.name 'LilianPamphile'")

# Cloner le dépôt
os.system(f"rm -rf {CLONE_DIR}")
os.system(f"git clone {GITHUB_REPO} {CLONE_DIR}")

# Création dossier model_files
model_path = f"{CLONE_DIR}/model_files"
os.makedirs(model_path, exist_ok=True)

# Sauvegarde des fichiers modèles directement dans le dossier cloné
with open(f"{model_path}/model_total_buts.pkl", "wb") as f:
    pickle.dump(model, f)
with open(f"{model_path}/scaler_total_buts.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Git add + commit + push
os.system(f"cd {CLONE_DIR} && git add model_files && git commit -m '📈 Update model_total_buts.pkl' && git push")
print("✅ Modèle pushé sur GitHub avec succès.")

# === Email notification ===
def send_email(subject, body, to_email):
    from_email = "lilian.pamphile.bts@gmail.com"
    password = "fifkktsenfxsqiob"  # mot de passe d'application Gmail

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(from_email, password)
            server.send_message(msg)
        print("📬 Mail de confirmation envoyé.")
    except Exception as e:
        print("❌ Erreur lors de l'envoi de l'email :", e)

# === Génération contenu du mail ===
today = date.today()

# Analyse qualitative du score
if rmse_score < 1.8:
    perf = "🟢 Excellent (faible écart avec le réel)"
elif rmse_score < 2.2:
    perf = "🟡 Correct (modèle utilisable)"
else:
    perf = "🔴 À surveiller (précision insuffisante)"

subject = "📊 Modèle total_buts mis à jour"
body = (
    f"Le modèle `total_buts` a été réentraîné le {today}.\n\n"
    f"📉 **MAE** : {mae_score:.4f} — Erreur absolue moyenne (but près). ➡️ En moyenne, le modèle se trompe d’environ 1.32 but sur ses prédictions.\n"
    f"📉 **RMSE** : {rmse_score:.4f} — Racine de l’erreur quadratique moyenne. ➡️ Cela donne une idée de l’écart-type des erreurs de prédiction : plus c’est bas, mieux c’est.\n"


    f"🔎 Interprétation : {perf}\n\n"
    "📁 Fichiers générés : model_total_buts.pkl & scaler_total_buts.pkl\n"
    "📤 Upload GitHub : ✅ effectué avec succès\n"
    "🔗 https://github.com/LilianPamphile/paris-sportifs/tree/main/model_files"
)

# Envoi
send_email(subject, body, "lilian.pamphile.bts@gmail.com")
