# train_model.py (version optimisée XGBoost)
import psycopg2
import pandas as pd
import pickle
from decimal import Decimal
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import numpy as np
import smtplib
from email.mime.text import MIMEText
from datetime import date
import os
import shutil

# --- Connexion BDD PostgreSQL ---
DATABASE_URL = "postgresql://postgres:jDDqfaqpspVDBBwsqxuaiSDNXjTxjMmP@shortline.proxy.rlwy.net:36536/railway"
conn = psycopg2.connect(DATABASE_URL)
cursor = conn.cursor()

# --- Extraction des donnees historiques enrichies ---
query = """
    SELECT 
        m.game_id, m.date::date AS date_match, m.equipe_domicile, m.equipe_exterieur,
        sg1.moyenne_buts AS buts_dom, sg1.buts_encaisse::FLOAT / NULLIF(sg1.matchs_joues, 0) AS buts_encaissés_dom,
        sg1.pourcentage_over_2_5 AS over25_dom, sg1.pourcentage_BTTS AS btts_dom,
        sg1.tirs_cadres AS tirs_dom, sg1.possession AS poss_dom, sg1.corners AS corners_dom, sg1.fautes AS fautes_dom, 
        sg1.cartons_jaunes AS cj_dom, sg1.cartons_rouges AS cr_dom,

        sg2.moyenne_buts AS buts_ext, sg2.buts_encaisse::FLOAT / NULLIF(sg2.matchs_joues, 0) AS buts_encaissés_ext,
        sg2.pourcentage_over_2_5 AS over25_ext, sg2.pourcentage_BTTS AS btts_ext,
        sg2.tirs_cadres AS tirs_ext, sg2.possession AS poss_ext, sg2.corners AS corners_ext, sg2.fautes AS fautes_ext,
        sg2.cartons_jaunes AS cj_ext, sg2.cartons_rouges AS cr_ext,

        s.buts_dom AS buts_m_dom, s.buts_ext AS buts_m_ext,
        s.buts_dom + s.buts_ext AS total_buts
    FROM matchs m
    JOIN stats_globales sg1 ON m.equipe_domicile = sg1.equipe AND m.competition = sg1.competition AND m.saison = sg1.saison
    JOIN stats_globales sg2 ON m.equipe_exterieur = sg2.equipe AND m.competition = sg2.competition AND m.saison = sg2.saison
    JOIN stats_matchs s ON m.game_id = s.game_id
    WHERE s.buts_dom IS NOT NULL AND s.buts_ext IS NOT NULL
"""

cursor.execute(query)
rows = cursor.fetchall()
cols = [desc[0] for desc in cursor.description]
df = pd.DataFrame(rows, columns=cols)

# --- Préchargement de tous les anciens matchs pour calcul de forme ---
query_all = """
    SELECT m.date::date AS date_match, m.equipe_domicile, m.equipe_exterieur,
           s.buts_dom, s.buts_ext
    FROM matchs m
    JOIN stats_matchs s ON m.game_id = s.game_id
    WHERE s.buts_dom IS NOT NULL AND s.buts_ext IS NOT NULL
"""
cursor.execute(query_all)
rows_all = cursor.fetchall()
df_all_matchs = pd.DataFrame(rows_all, columns=["date", "dom", "ext", "buts_dom", "buts_ext"])

conn.close()

# --- Nouvelle fonction de forme sans SQL ---
def get_forme(df_hist, equipe, date_ref):
    matchs = df_hist[(
        (df_hist["dom"] == equipe) | (df_hist["ext"] == equipe)) &
        (df_hist["date"] < date_ref)
    ].sort_values("date", ascending=False).head(5)

    if matchs.empty:
        return 0.0, 0.0

    buts_marques, buts_encaissés = [], []
    for _, row in matchs.iterrows():
        est_dom = (equipe == row["dom"])
        buts_marques.append(row["buts_dom"] if est_dom else row["buts_ext"])
        buts_encaissés.append(row["buts_ext"] if est_dom else row["buts_dom"])

    return sum(buts_marques) / len(buts_marques), sum(buts_encaissés) / len(buts_encaissés)

forme_buts_dom, forme_enc_dom = [], []
forme_buts_ext, forme_enc_ext = [], []

for _, row in df.iterrows():
    b_m, b_e = get_forme(df_all_matchs, row['equipe_domicile'], row['date_match'])
    forme_buts_dom.append(b_m)
    forme_enc_dom.append(b_e)
    b_m2, b_e2 = get_forme(df_all_matchs, row['equipe_exterieur'], row['date_match'])
    forme_buts_ext.append(b_m2)
    forme_enc_ext.append(b_e2)

# Ajout au DataFrame
df["forme_buts_dom"] = forme_buts_dom
df["forme_buts_ext"] = forme_buts_ext
df["forme_encaissés_dom"] = forme_enc_dom
df["forme_encaissés_ext"] = forme_enc_ext

# --- Convertir Decimal → float ---
for col in df.columns:
    if df[col].dtype == 'object' and not df[col].dropna().empty and isinstance(df[col].dropna().iloc[0], Decimal):
        df[col] = df[col].astype(float)

# --- Feature engineering enrichie ---
df["tirs_cadres"] = df["tirs_dom"] + df["tirs_ext"]
df["possession"] = df["poss_dom"] + df["poss_ext"]
df["corners_fautes"] = df["corners_dom"] + df["corners_ext"] + df["fautes_dom"] + df["fautes_ext"]
df["cartons"] = df["cj_dom"] + df["cj_ext"] + 2 * df["cr_dom"] + 2 * df["cr_ext"]

# Score heuristique réajusté
df["score_heuristique"] = (
    0.20 * (df["buts_dom"] + df["buts_ext"]) +
    0.15 * (df["forme_buts_dom"] + df["forme_buts_ext"]) +
    0.15 * (df["over25_dom"] + df["over25_ext"]) +
    0.15 * (df["btts_dom"] + df["btts_ext"]) +
    0.10 * df["tirs_cadres"] +
    0.10 * df["possession"] +
    0.05 * (df["forme_encaissés_dom"] + df["forme_encaissés_ext"]) +
    0.05 * df["corners_fautes"] +
    0.05 * df["cartons"]
)

# Cible
df["over_2_5"] = df["total_buts"] > 2.5

# --- Features & target enrichis ---
features = [
    "buts_dom", "buts_ext",
    "buts_encaissés_dom", "buts_encaissés_ext",
    "forme_buts_dom", "forme_buts_ext", "forme_encaissés_dom", "forme_encaissés_ext",
    "over25_dom", "over25_ext",
    "btts_dom", "btts_ext",
    "tirs_cadres", "possession", "corners_fautes", "cartons",
    "score_heuristique"
]
X = df[features]
y = df["over_2_5"].astype(int)

# --- Normalisation ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- XGBoost Classifier ---
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

# --- Cross-validation ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_scaled, y, cv=skf, scoring='roc_auc')
print(f"✅ Cross-validated AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# --- Train final model ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
model.fit(X_train, y_train)

# --- Evaluation ---
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("AUC Test:", roc_auc_score(y_test, y_proba))

# --- Export modèle + scaler ---
with open("model_over25.pkl", "wb") as f:
    pickle.dump(model, f)
with open("scaler_over25.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Modèle XGBoost entraîné, validé et sauvegardé avec succès.")

# === Envoi sur GitHub ===
print("📦 Push vers GitHub...")

GITHUB_REPO = "https://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@github.com/LilianPamphile/paris-sportifs.git"
CLONE_DIR = "model_push"

os.system(f"git config --global user.email 'lilian.pamphile.bts@gmail.com'")
os.system(f"git config --global user.name 'LilianPamphile'")
os.system(f"rm -rf {CLONE_DIR}")
os.system(f"git clone {GITHUB_REPO} {CLONE_DIR}")
os.makedirs(f"{CLONE_DIR}/model_files", exist_ok=True)

# Copier les fichiers dans le bon dossier
shutil.copy("model_over25.pkl", f"{CLONE_DIR}/model_files/model_over25.pkl")
shutil.copy("scaler_over25.pkl", f"{CLONE_DIR}/model_files/scaler_over25.pkl")

# S'assurer que Git suive à nouveau le dossier
os.system(f"cd {CLONE_DIR} && git add model_files/*.pkl")
os.system(f"cd {CLONE_DIR} && git commit -m '🧠 Update model files' || echo '🔁 Rien à commit'")
os.system(f"cd {CLONE_DIR} && git push")

print("✅ Modèle pushé sur GitHub avec succès.")

# === Envoi d'email final ===
def send_email(subject, body, to_email):
    from_email = "lilian.pamphile.bts@gmail.com"
    password = "fifkktsenfxsqiob"

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

# === Confirmation ===
today = date.today()
auc_test = roc_auc_score(y_test, y_proba)

if auc_test < 0.70:
    subject = "⚠️ Alerte - AUC du modèle en baisse"
    body = (
        f"Le modèle a été réentraîné le {today}, mais l'AUC est faible : {auc_test:.4f} ❌\n"
        "Vérifie les données ou réentraîne manuellement si besoin.\n\n"
        "📁 Fichiers générés : model_over25.pkl, scaler_over25.pkl\n"
        "📤 Upload GitHub : ✅ effectué avec succès\n"
        "🔗 https://github.com/LilianPamphile/paris-sportifs/model_files"
    )
else:
    subject = "✅ Modèle mis à jour avec succès"
    body = (
        f"Le modèle a bien été réentraîné le {today}.\n"
        f"AUC Test : {auc_test:.4f} ✅\n\n"
        "📁 Fichiers générés : model_over25.pkl & scaler_over25.pkl\n"
        "📤 Upload GitHub : ✅ effectué avec succès\n"
        "🔗 https://github.com/LilianPamphile/paris-sportifs/model_files"
    )

send_email(subject, body, to_email="lilian.pamphile.bts@gmail.com")
