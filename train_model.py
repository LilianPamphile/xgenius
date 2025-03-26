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


# --- Connexion BDD PostgreSQL ---
DATABASE_URL = "postgresql://postgres:jDDqfaqpspVDBBwsqxuaiSDNXjTxjMmP@shortline.proxy.rlwy.net:36536/railway"
conn = psycopg2.connect(DATABASE_URL)
cursor = conn.cursor()

# --- Extraction des donnees historiques ---
query = """
    SELECT 
        m.game_id, m.equipe_domicile, m.equipe_exterieur,
        sg1.moyenne_buts AS buts_dom, sg1.pourcentage_over_2_5 AS over25_dom, sg1.pourcentage_BTTS AS btts_dom,
        sg1.tirs_cadres AS tirs_dom, sg1.possession AS poss_dom, sg1.corners AS corners_dom, sg1.fautes AS fautes_dom, 
        sg1.cartons_jaunes AS cj_dom, sg1.cartons_rouges AS cr_dom,
        sg2.moyenne_buts AS buts_ext, sg2.pourcentage_over_2_5 AS over25_ext, sg2.pourcentage_BTTS AS btts_ext,
        sg2.tirs_cadres AS tirs_ext, sg2.possession AS poss_ext, sg2.corners AS corners_ext, sg2.fautes AS fautes_ext,
        sg2.cartons_jaunes AS cj_ext, sg2.cartons_rouges AS cr_ext,
        c.cote_over, 
        s.buts_dom + s.buts_ext AS total_buts
    FROM matchs m
    JOIN stats_globales sg1 ON m.equipe_domicile = sg1.equipe
    JOIN stats_globales sg2 ON m.equipe_exterieur = sg2.equipe
    JOIN stats_matchs s ON m.game_id = s.game_id
    JOIN cotes c ON m.game_id = c.game_id
    WHERE s.buts_dom IS NOT NULL AND s.buts_ext IS NOT NULL AND c.cote_over IS NOT NULL
"""

cursor.execute(query)
rows = cursor.fetchall()
cols = [desc[0] for desc in cursor.description]
df = pd.DataFrame(rows, columns=cols)
cursor.close()
conn.close()

# --- Convertir Decimal → float ---
for col in df.columns:
    if df[col].dtype == 'object' and isinstance(df[col].dropna().iloc[0], Decimal):
        df[col] = df[col].astype(float)

# --- Feature engineering ---
df["tirs_cadres"] = df["tirs_dom"] + df["tirs_ext"]
df["possession"] = df["poss_dom"] + df["poss_ext"]
df["corners_fautes"] = df["corners_dom"] + df["corners_ext"] + df["fautes_dom"] + df["fautes_ext"]
df["cartons"] = df["cj_dom"] + df["cj_ext"] + 2 * df["cr_dom"] + 2 * df["cr_ext"]
df["score_heuristique"] = (
    0.20 * (df["buts_dom"] + df["buts_ext"]) +
    0.20 * (df["over25_dom"] + df["over25_ext"]) +
    0.15 * (df["btts_dom"] + df["btts_ext"]) +
    0.10 * df["tirs_cadres"] +
    0.15 * (2.5 / df["cote_over"]) +
    0.05 * df["possession"] +
    0.05 * df["corners_fautes"] +
    0.05 * df["cartons"]
)
df["over_2_5"] = df["total_buts"] > 2.5

# --- Features & target ---
features = [
    "buts_dom", "buts_ext",
    "over25_dom", "over25_ext",
    "btts_dom", "btts_ext",
    "tirs_cadres", "possession", "corners_fautes", "cartons",
    "cote_over", "score_heuristique"
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


import os
import shutil

# === Envoi sur GitHub ===
print("📦 Push vers GitHub...")

GITHUB_REPO = "https://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@github.com/LilianPamphile/paris-sportifs.git"
CLONE_DIR = "model_push"

# Config git (si pas déjà globalement configuré)
os.system(f"git config --global user.email 'lilian.pamphile.bts@gmail.com'")
os.system(f"git config --global user.name 'LilianPamphile'")

# Clone propre
os.system(f"rm -rf {CLONE_DIR}")
os.system(f"git clone {GITHUB_REPO} {CLONE_DIR}")

# Créer le dossier si pas là
os.makedirs(f"{CLONE_DIR}/model_files", exist_ok=True)

# Copier les fichiers modèles
shutil.copy("model_over25.pkl", f"{CLONE_DIR}/model_files/model_over25.pkl")
shutil.copy("scaler_over25.pkl", f"{CLONE_DIR}/model_files/scaler_over25.pkl")

# Commit + push
os.system(f"cd {CLONE_DIR} && git add model_files && git commit -m '🧠 Update model files' && git push")

print("✅ Modèle pushé sur GitHub avec succès.")


def send_email(subject, body, to_email):
    from_email = "lilian.pamphile.bts@gmail.com"
    password = "fifkktsenfxsqiob"  # mot de passe d'application

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

# === Envoi d'email final avec confirmation GitHub ===
today = date.today()
auc_test = roc_auc_score(y_test, y_proba)

# Statut du modèle
if auc_test < 0.70:
    subject = "⚠️ Alerte - AUC du modèle en baisse"
    body = (
        f"Le modèle a été réentraîné le {today}, mais l'AUC est faible : {auc_test:.4f} ❌\n"
        "Vérifie les données ou réentraîne manuellement si besoin.\n\n"
        "📁 Fichiers générés : model_over25.pkl, scaler_over25.pkl\n"
        "📤 Upload GitHub : ✅ effectué avec succès\n"
        "🔗 https://github.com/LilianPamphile/paris-sportifs/tree/main/model_files"
    )
else:
    subject = "✅ Modèle mis à jour avec succès"
    body = (
        f"Le modèle a bien été réentraîné le {today}.\n"
        f"AUC Test : {auc_test:.4f} ✅\n\n"
        "📁 Fichiers générés : model_over25.pkl & scaler_over25.pkl\n"
        "📤 Upload GitHub : ✅ effectué avec succès\n"
        "🔗 https://github.com/LilianPamphile/paris-sportifs/tree/main/model_files"
    )

send_email(subject, body, to_email="lilian.pamphile.bts@gmail.com")
