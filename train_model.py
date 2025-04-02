import psycopg2
import pandas as pd
import pickle
from decimal import Decimal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
import numpy as np
import os

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
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modèles
models = {
    "catboost": CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6, random_seed=42, verbose=0),
    "lightgbm": LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42),
    "mlp": MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    results[name] = (model, mae, rmse)
    print(f"{name} - MAE: {mae:.4f} | RMSE: {rmse:.4f}")

# Push sur GitHub
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = f"https://{GITHUB_TOKEN}@github.com/LilianPamphile/paris-sportifs.git"
CLONE_DIR = "model_push"
os.system(f"rm -rf {CLONE_DIR}")
os.system(f"git clone {GITHUB_REPO} {CLONE_DIR}")
model_path = f"{CLONE_DIR}/model_files"
os.makedirs(model_path, exist_ok=True)

# Sauvegarde des modèles
for name, (model, _, _) in results.items():
    with open(f"{model_path}/model_total_buts_{name}.pkl", "wb") as f:
        pickle.dump(model, f)

# Sauvegarde scaler
with open(f"{model_path}/scaler_total_buts.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Git push
os.system(f"cd {CLONE_DIR} && git add model_files && git commit -m '🔁 Update models v3' && git push")

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
    f"📉 **MAE** : {mae_score:.4f} — Erreur absolue moyenne (but près). ➡️ En moyenne, le modèle se trompe d’environ {mae_score:.4f} but sur ses prédictions.\n"
    f"📉 **RMSE** : {rmse_score:.4f} — Racine de l’erreur quadratique moyenne. ➡️ Cela donne une idée de l’écart-type des erreurs de prédiction : plus c’est bas, mieux c’est.\n"


    f"🔎 Interprétation : {perf}\n\n"
    "📁 Fichiers générés : model_total_buts.pkl & scaler_total_buts.pkl\n"
    "📤 Upload GitHub : ✅ effectué avec succès\n"
    "🔗 https://github.com/LilianPamphile/paris-sportifs/tree/main/model_files"
)

# Envoi
send_email(subject, body, "lilian.pamphile.bts@gmail.com")
