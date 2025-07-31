# -*- coding: utf-8 -*-
import requests
import psycopg2
# Fonction de conversion sécurisée
import math
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
import os
import requests
import pickle
import pandas as pd
import numpy as np

# 🔑 Clé API SportsData.io
API_KEY = "b63f99b8e4mshb5383731d310a85p103ea1jsn47e34368f5df"

today = datetime.today().date()
yesterday = today - timedelta(days=1)

# 🏆 Liste des compétitions à récupérer
COMPETITIONS = {
    "Ligue 1": "61",
    "Premier League": "39",
    "Bundesliga": "78",
    "Serie A": "135",
    "Eredivisie": "88",
    "La Liga": "140",
    "UEFA Champions League": "2",
    "2. Bundesliga": "79",
    "UEFA Europa League": "3",
    "UEFA Europa Conference League": "848",
    "Saudi Professional League": "307"
}

# 🔌 Connexion PostgreSQL Railway
DATABASE_URL = "postgresql://postgres:jDDqfaqpspVDBBwsqxuaiSDNXjTxjMmP@shortline.proxy.rlwy.net:36536/railway"
conn = psycopg2.connect(DATABASE_URL)
cursor = conn.cursor()

print("Fin de la défintion de variables")

################################### Fontions utiles ###################################

def to_float(x):
    try:
        return float(x)
    except:
        return 0.0

def convert_to_int(value):
    try:
        if isinstance(value, str) and "%" in value:
            value = value.replace("%", "")
        return max(int(float(value)), 0)
    except:
        return 0


def extract_stat(stats, stat_name):
    for s in stats.get("statistics", []):
        if s["type"] == stat_name:
            try:
                value = s["value"]
                if isinstance(value, str) and "%" in value:
                    return int(float(value.replace("%", "")))
                return int(float(value))
            except:
                return 0
    return 0

def get_fixture_with_goals(fixture_id, headers):
    url = "https://api-football-v1.p.rapidapi.com/v3/fixtures"
    response = requests.get(url, headers=headers, params={"ids": fixture_id})
    if response.status_code == 200:
        data = response.json().get("response", [])
        if data:
            goals = data[0]["goals"]
            return goals["home"], goals["away"]
    return 0, 0

"""## **Envoie de mail et execution des fonction de récupération de données**"""
def send_email(subject, body, to_email):
    from_email = "lilian.pamphile.bts@gmail.com"
    password = "fifkktsenfxsqiob"  # mot de passe d'application

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = ", ".join(to_email)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(from_email, password)
            server.send_message(msg, from_addr=from_email, to_addrs=to_email)
        print("📬 Email envoyé avec succès.")
    except Exception as e:
        print("❌ Erreur lors de l'envoi de l'email :", e)

# --- Téléchargement des fichiers modèle/scaler depuis GitHub ---
def telecharger_model_depuis_github():
    # Infos du repo
    REPO = "LilianPamphile/paris-sportifs"
    BRANCH = "main"
    TOKEN = os.getenv("GITHUB_TOKEN")
    
    # Liste des fichiers à télécharger (avec chemin dans le repo GitHub)
    fichiers = {
        "model_files/model_total_buts_catboost_optuna.pkl": "model_files/model_total_buts_catboost_optuna.pkl",
        "model_files/model_total_buts_hist_gradient_boosting.pkl": "model_files/model_total_buts_hist_gradient_boosting.pkl",
        "model_files/model_total_buts_conformal_p25.pkl": "model_files/model_total_buts_conformal_p25.pkl",
        "model_files/model_total_buts_conformal_p75.pkl": "model_files/model_total_buts_conformal_p75.pkl",
        "model_files/scaler_total_buts.pkl": "model_files/scaler_total_buts.pkl",
        "model_files/features_list.pkl": "model_files/features_list.pkl"
    }

    for chemin_dist, chemin_local in fichiers.items():
        url = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/{chemin_dist}"
        headers = {"Authorization": f"token {TOKEN}"}

        # Crée le dossier local si besoin
        dossier = os.path.dirname(chemin_local)
        if not os.path.exists(dossier):
            os.makedirs(dossier)

        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            with open(chemin_local, "wb") as f:
                f.write(response.content)
            print(f"✅ Fichier téléchargé : {chemin_local}")

        else:
            print(f"❌ Échec du téléchargement de {chemin_local} ({response.status_code})")

def compute_gmos(pred_ml, p25, p75, score_heuristique):
    """Calcule un score GMOS (entre 0 et 100) à partir des éléments clés."""

    variance_range = max(p75 - p25, 0.1)  # éviter division par 0
    range_score = 1 - (variance_range / 5)  # plus c’est resserré, mieux c’est

    base_score = 0.4 * (min(pred_ml / 5, 1) * 100) + 0.3 * score_heuristique + 0.3 * (range_score * 100)
    return round(min(max(base_score, 0), 100), 2)

###################################################################################################

# === 📌 1️⃣ Récupération des Matchs ===
def recuperer_matchs(date, API_KEY):
    url_base = "https://api-football-v1.p.rapidapi.com/v3/fixtures"

    headers = {
        "x-rapidapi-key": API_KEY,
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com"
    }

    saison_api = datetime.now().year if datetime.now().month >= 7 else datetime.now().year - 1
    total_matchs = 0

    print(f"📅 Récupération des matchs pour le {date}")

    for competition_name, competition_id in COMPETITIONS.items():
        params = {
            "league": competition_id,
            "season": saison_api,
            "date": date,
            "timezone": "Europe/Paris"
        }

        response = requests.get(url_base, headers=headers, params=params)

        if response.status_code == 200:
            data = response.json().get("response", [])
            matchs_inseres_competition = 0

            for match in data:
                game_id = match["fixture"]["id"]
                date_match = match["fixture"]["date"]
                saison = int(match["league"]["season"])
                statut = match["fixture"]["status"]["long"]
                equipe_domicile = match["teams"]["home"]["name"]
                equipe_exterieur = match["teams"]["away"]["name"]

                try:
                    cursor.execute("""
                        INSERT INTO matchs_v2 (game_id, saison, date, statut, equipe_domicile, equipe_exterieur, competition)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (game_id) DO NOTHING
                    """, (game_id, saison, date_match, statut, equipe_domicile, equipe_exterieur, competition_name))
                    matchs_inseres_competition += 1
                    total_matchs += 1
                except Exception as e:
                    print(f"❌ Erreur insertion match {game_id} : {e}")

            if matchs_inseres_competition > 0:
                print(f"   ✅ {matchs_inseres_competition} matchs insérés pour {competition_name}")
        else:
            print(f"❌ Erreur API pour {competition_name} : {response.status_code}")

    conn.commit()
    print(f"📊 Total : {total_matchs} matchs insérés pour {date}")

# === 📌 Récupération des Stats ===
def recuperer_stats_matchs(date, API_KEY):
    url_fixtures = "https://api-football-v1.p.rapidapi.com/v3/fixtures"
    url_stats = "https://api-football-v1.p.rapidapi.com/v3/fixtures/statistics"
    url_xg = "https://api-football-v1.p.rapidapi.com/v3/fixtures"  # même endpoint, xG inclus dans statistics parfois

    headers = {
        "x-rapidapi-key": API_KEY,
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com"
    }

    print(f"📅 Stats pour {date}")

    for competition_name, competition_id in COMPETITIONS.items():
        params = {
            "league": competition_id,
            "season": datetime.now().year if datetime.now().month >= 7 else datetime.now().year - 1,
            "date": date,
            "timezone": "Europe/Paris"
        }

        response = requests.get(url_fixtures, headers=headers, params=params)
        if response.status_code != 200:
            continue

        fixtures = response.json().get("response", [])
        if not fixtures:
            continue

        for match in fixtures:
            fixture_id = match["fixture"]["id"]
            equipe_dom = match["teams"]["home"]["name"]
            equipe_ext = match["teams"]["away"]["name"]

            try:
                response_stats = requests.get(url_stats, headers=headers, params={"fixture": fixture_id})
                response_stats.raise_for_status()
            except:
                continue

            stats_data = response_stats.json().get("response", [])
            if len(stats_data) != 2:
                continue

            stats_dom = stats_data[0] if stats_data[0]["team"]["name"] == equipe_dom else stats_data[1]
            stats_ext = stats_data[1] if stats_dom == stats_data[0] else stats_data[0]

            def get_xg(stats):
                for s in stats.get("statistics", []):
                    if s["type"].lower() in ["expected goals", "xg"]:
                        try:
                            val = float(s["value"])
                            return val if val > 0 else 0.3
                        except:
                            return 0.3
                return 0.3

            xg_dom = get_xg(stats_dom)
            xg_ext = get_xg(stats_ext)

            buts_dom, buts_ext = get_fixture_with_goals(fixture_id, headers)


            values = (
                fixture_id,
                extract_stat(stats_dom, 'Ball Possession'), extract_stat(stats_ext, 'Ball Possession'),
                extract_stat(stats_dom, 'Total Shots'), extract_stat(stats_ext, 'Total Shots'),
                extract_stat(stats_dom, 'Shots on Goal'), extract_stat(stats_ext, 'Shots on Goal'),
                extract_stat(stats_dom, 'Shots off Goal'), extract_stat(stats_ext, 'Shots off Goal'),
                extract_stat(stats_dom, 'Blocked Shots'), extract_stat(stats_ext, 'Blocked Shots'),
                extract_stat(stats_dom, 'Shots insidebox'), extract_stat(stats_ext, 'Shots insidebox'),
                extract_stat(stats_dom, 'Shots outsidebox'), extract_stat(stats_ext, 'Shots outsidebox'),
                extract_stat(stats_dom, 'Goalkeeper Saves'), extract_stat(stats_ext, 'Goalkeeper Saves'),
                buts_dom, buts_ext,
                extract_stat(stats_dom, 'Total passes'), extract_stat(stats_ext, 'Total passes'),
                extract_stat(stats_dom, 'Passes accurate'), extract_stat(stats_ext, 'Passes accurate'),
                extract_stat(stats_dom, 'Passes %'), extract_stat(stats_ext, 'Passes %'),
                extract_stat(stats_dom, 'Corner Kicks'), extract_stat(stats_ext, 'Corner Kicks'),
                extract_stat(stats_dom, 'Fouls'), extract_stat(stats_ext, 'Fouls'),
                extract_stat(stats_dom, 'Offsides'), extract_stat(stats_ext, 'Offsides'),
                extract_stat(stats_dom, 'Yellow Cards'), extract_stat(stats_ext, 'Yellow Cards'),
                extract_stat(stats_dom, 'Red Cards'), extract_stat(stats_ext, 'Red Cards')
            )

            query = (
                "INSERT INTO stats_matchs_v2 ("
                "game_id, possession_dom, possession_ext, tirs_dom, tirs_ext, "
                "tirs_cadres_dom, tirs_cadres_ext, tirs_hors_cadre_dom, tirs_hors_cadre_ext, "
                "tirs_bloques_dom, tirs_bloques_ext, tirs_dans_boite_dom, tirs_dans_boite_ext, "
                "tirs_hors_boite_dom, tirs_hors_boite_ext, arrets_dom, arrets_ext, "
                "buts_dom, buts_ext, passes_dom, passes_ext, passes_reussies_dom, passes_reussies_ext, "
                "passes_pourcent_dom, passes_pourcent_ext, corners_dom, corners_ext, "
                "fautes_dom, fautes_ext, hors_jeu_dom, hors_jeu_ext, "
                "cartons_jaunes_dom, cartons_jaunes_ext, cartons_rouges_dom, cartons_rouges_ext"
                ") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
                "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                " ON CONFLICT (game_id) DO NOTHING"
            )


            cursor.execute(query, values)

    conn.commit()
    print(f"✅ Stats enrichies insérées avec succès pour {date}")

### Mettre a jout table stats globals

def mettre_a_jour_stats_globales(date_reference):
    print("📊 Mise à jour des stats globales des équipes ayant joué le", date_reference)

    cursor.execute("""
        SELECT DISTINCT m.saison, m.competition, m.equipe_domicile AS equipe
        FROM matchs_v2 m
        JOIN stats_matchs_v2 s ON m.game_id = s.game_id
        WHERE m.date::date = %s
        UNION
        SELECT DISTINCT m.saison, m.competition, m.equipe_exterieur AS equipe
        FROM matchs_v2 m
        JOIN stats_matchs_v2 s ON m.game_id = s.game_id
        WHERE m.date::date = %s
    """, (date_reference, date_reference))
    equipes = cursor.fetchall()

    for saison, competition, equipe in equipes:
        cursor.execute("""
            SELECT m.game_id, m.equipe_domicile, m.equipe_exterieur, s.*
            FROM matchs_v2 m
            JOIN stats_matchs_v2 s ON m.game_id = s.game_id
            WHERE m.saison = %s AND m.competition = %s
              AND (m.equipe_domicile = %s OR m.equipe_exterieur = %s)
        """, (saison, competition, equipe, equipe))
        matchs = cursor.fetchall()

        if not matchs:
            continue

        # Initialise tous les champs à 0
        total = {field: 0 for field in [
            "matchs_joues", "victoires", "nuls", "defaites", "buts_marques", "buts_encaisse",
            "difference_buts", "tirs", "tirs_cadres", "tirs_hors_cadre", "tirs_bloques",
            "tirs_dans_boite", "tirs_hors_boite", "arrets", "passes", "passes_reussies",
            "passes_pourcent", "possession", "corners", "cartons_jaunes", "cartons_rouges",
            "fautes", "hors_jeu", "btts", "over_2_5", "over_1_5", "clean_sheets",
            "xg_dom", "xg_ext"
        ]}

        for match in matchs:
            data = dict(zip([desc[0] for desc in cursor.description], match))
            est_domicile = (equipe == data["equipe_domicile"])

            def get(field):
                dom_key = f"{field}_dom"
                ext_key = f"{field}_ext"
                if est_domicile:
                    return data.get(dom_key) or 0
                else:
                    return data.get(ext_key) or 0

            buts_marques = get("buts")

            buts_dom = data.get("buts_dom") or 0
            buts_ext = data.get("buts_ext") or 0
            buts_encaisse = buts_ext if est_domicile else buts_dom


            total["matchs_joues"] += 1
            total["buts_marques"] += buts_marques
            total["buts_encaisse"] += buts_encaisse
            total["difference_buts"] += (buts_marques - buts_encaisse)
            total["victoires"] += int(buts_marques > buts_encaisse)
            total["nuls"] += int(buts_marques == buts_encaisse)
            total["defaites"] += int(buts_marques < buts_encaisse)

            for f in [
                "tirs", "tirs_cadres", "tirs_hors_cadre", "tirs_bloques", "tirs_dans_boite",
                "tirs_hors_boite", "arrets", "passes", "passes_reussies", "passes_pourcent",
                "possession", "corners", "cartons_jaunes", "cartons_rouges",
                "fautes", "hors_jeu"
            ]:
                total[f] += get(f)

            buts_dom = data.get("buts_dom") or 0
            buts_ext = data.get("buts_ext") or 0

            if buts_dom > 0 and buts_ext > 0:
                total["btts"] += 1
            if (buts_dom + buts_ext) > 2.5:
                total["over_2_5"] += 1
            if (buts_dom + buts_ext) > 1.5:
                total["over_1_5"] += 1
            if buts_encaisse == 0:
                total["clean_sheets"] += 1

            total["xg_dom"] += data.get("xg_dom") or 0 if est_domicile else 0
            total["xg_ext"] += data.get("xg_ext") or 0 if not est_domicile else 0

        def avg(val):
            return round(val / total["matchs_joues"], 2) if total["matchs_joues"] else 0.0

        values = (
            equipe, competition, saison,
            total["matchs_joues"], total["victoires"], total["nuls"], total["defaites"],
            total["buts_marques"], total["buts_encaisse"], total["difference_buts"],
            total["tirs"], total["tirs_cadres"], total["tirs_hors_cadre"], total["tirs_bloques"],
            total["tirs_dans_boite"], total["tirs_hors_boite"], total["arrets"], total["passes"],
            avg(total["passes_reussies"]), avg(total["passes_pourcent"]), avg(total["possession"]),
            total["corners"], total["cartons_jaunes"], total["cartons_rouges"],
            total["fautes"], total["hors_jeu"], avg(total["buts_marques"]),
            round(100 * total["btts"] / total["matchs_joues"], 2),
            round(100 * total["over_2_5"] / total["matchs_joues"], 2),
            round(100 * total["over_1_5"] / total["matchs_joues"], 2),
            round(100 * total["clean_sheets"] / total["matchs_joues"], 2),
            avg(total["xg_dom"]), avg(total["xg_ext"])
        )

        cursor.execute("""
            INSERT INTO stats_globales_v2 (
                equipe, competition, saison, matchs_joues, victoires, nuls, defaites,
                buts_marques, buts_encaisse, difference_buts, tirs, tirs_cadres,
                tirs_hors_cadre, tirs_bloques, tirs_dans_boite, tirs_hors_boite,
                arrets, passes, passes_reussies, passes_pourcent, possession, corners,
                cartons_jaunes, cartons_rouges, fautes, hors_jeu, moyenne_buts,
                pourcentage_BTTS, pourcentage_over_2_5, pourcentage_over_1_5,
                pourcentage_clean_sheets, moyenne_xg_dom, moyenne_xg_ext
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                      %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (equipe, competition, saison) DO UPDATE SET
                matchs_joues = EXCLUDED.matchs_joues,
                victoires = EXCLUDED.victoires,
                nuls = EXCLUDED.nuls,
                defaites = EXCLUDED.defaites,
                buts_marques = EXCLUDED.buts_marques,
                buts_encaisse = EXCLUDED.buts_encaisse,
                difference_buts = EXCLUDED.difference_buts,
                tirs = EXCLUDED.tirs,
                tirs_cadres = EXCLUDED.tirs_cadres,
                tirs_hors_cadre = EXCLUDED.tirs_hors_cadre,
                tirs_bloques = EXCLUDED.tirs_bloques,
                tirs_dans_boite = EXCLUDED.tirs_dans_boite,
                tirs_hors_boite = EXCLUDED.tirs_hors_boite,
                arrets = EXCLUDED.arrets,
                passes = EXCLUDED.passes,
                passes_reussies = EXCLUDED.passes_reussies,
                passes_pourcent = EXCLUDED.passes_pourcent,
                possession = EXCLUDED.possession,
                corners = EXCLUDED.corners,
                cartons_jaunes = EXCLUDED.cartons_jaunes,
                cartons_rouges = EXCLUDED.cartons_rouges,
                fautes = EXCLUDED.fautes,
                hors_jeu = EXCLUDED.hors_jeu,
                moyenne_buts = EXCLUDED.moyenne_buts,
                pourcentage_BTTS = EXCLUDED.pourcentage_BTTS,
                pourcentage_over_2_5 = EXCLUDED.pourcentage_over_2_5,
                pourcentage_over_1_5 = EXCLUDED.pourcentage_over_1_5,
                pourcentage_clean_sheets = EXCLUDED.pourcentage_clean_sheets,
                moyenne_xg_dom = EXCLUDED.moyenne_xg_dom,
                moyenne_xg_ext = EXCLUDED.moyenne_xg_ext
        """, values)

    conn.commit()
    print("✅ stats_globales_v2 mises à jour avec succès !")

#################################################################################################

try:
    recuperer_matchs(today, API_KEY)
    recuperer_stats_matchs(yesterday, API_KEY)
    mettre_a_jour_stats_globales(yesterday)

    telecharger_model_depuis_github()

    conn.commit()

    print("✅ Récupération des données terminée !")

    # === Chargement du modèle ML et scaler ===
    with open("model_files/model_total_buts_catboost_optuna.pkl", "rb") as f:
        model_cat = pickle.load(f)
    with open("model_files/model_total_buts_hist_gradient_boosting.pkl", "rb") as f:
        model_hgb = pickle.load(f)
    with open("model_files/model_total_buts_conformal_p25.pkl", "rb") as f:
        model_p25 = pickle.load(f)
    with open("model_files/model_total_buts_conformal_p75.pkl", "rb") as f:
        model_p75 = pickle.load(f)
    with open("model_files/scaler_total_buts.pkl", "rb") as f:
        scaler_ml = pickle.load(f)
    with open("model_files/features_list.pkl", "rb") as f:
        features = pickle.load(f)

    # === Récupération historique des anciens matchs ===
    query_hist = """
        SELECT m.date::date AS date_match, m.equipe_domicile AS dom, m.equipe_exterieur AS ext,
               s.buts_dom, s.buts_ext
        FROM matchs_v2 m
        JOIN stats_matchs_v2 s ON m.game_id = s.game_id
        WHERE s.buts_dom IS NOT NULL AND s.buts_ext IS NOT NULL
    """
    cursor = conn.cursor()
    cursor.execute(query_hist)
    rows_hist = cursor.fetchall()
    df_all = pd.DataFrame(rows_hist, columns=["date", "dom", "ext", "buts_dom", "buts_ext"])

    def get_forme(df_hist, equipe, date_ref):
        matchs = df_hist[
            ((df_hist["dom"] == equipe) | (df_hist["ext"] == equipe)) &
            (df_hist["date"] < date_ref)
        ].sort_values("date", ascending=False).head(5)

        if matchs.empty:
            return 0.0, 0.0, 0.0

        buts_marques, buts_encaissés, over25 = [], [], []
        for _, row in matchs.iterrows():
            est_dom = (equipe == row["dom"])
            buts_marques.append(row["buts_dom"] if est_dom else row["buts_ext"])
            buts_encaissés.append(row["buts_ext"] if est_dom else row["buts_dom"])
            over25.append(int(row["buts_dom"] + row["buts_ext"] > 2.5))

        return (
            sum(buts_marques)/len(buts_marques),
            sum(buts_encaissés)/len(buts_encaissés),
            sum(over25)/len(over25)
        )

    #test
    def get_matchs_jour_for_prediction():
        cursor = conn.cursor()

        # 1. Récupère les stats globales nécessaires
        query = """
            SELECT
                m.game_id,
                m.date::date AS date_match,
                m.equipe_domicile, m.equipe_exterieur,

                sg1.moyenne_buts, sg1.buts_encaisse::FLOAT / NULLIF(sg1.matchs_joues, 0), sg1.pourcentage_over_2_5,
                sg1.pourcentage_over_1_5, sg1.pourcentage_BTTS, sg1.passes_pourcent, sg1.passes_reussies,
                sg1.possession, sg1.corners, sg1.fautes, sg1.cartons_jaunes, sg1.cartons_rouges,
                sg1.moyenne_xg_dom, sg1.tirs, sg1.tirs_cadres,
                sg1.pourcentage_clean_sheets, sg2.pourcentage_clean_sheets,
                sg2.moyenne_buts, sg2.buts_encaisse::FLOAT / NULLIF(sg2.matchs_joues, 0), sg2.pourcentage_over_2_5,
                sg2.pourcentage_over_1_5, sg2.pourcentage_BTTS, sg2.passes_pourcent,
                sg2.passes_reussies, sg2.possession, sg2.corners, sg2.fautes,
                sg2.cartons_jaunes, sg2.cartons_rouges, sg2.moyenne_xg_ext, sg2.tirs, sg2.tirs_cadres
            FROM matchs_v2 m
            JOIN stats_globales_v2 sg1 ON m.equipe_domicile = sg1.equipe AND m.competition = sg1.competition AND m.saison = sg1.saison
            JOIN stats_globales_v2 sg2 ON m.equipe_exterieur = sg2.equipe AND m.competition = sg2.competition AND m.saison = sg2.saison
            WHERE DATE(m.date) = %s
        """
        cursor.execute(query, (today,))
        rows = cursor.fetchall()

        # 2. Récupère l'historique pour forme récente
        cursor.execute("""
            SELECT m.date::date AS date_match, m.equipe_domicile, m.equipe_exterieur,
                  s.buts_dom AS buts_m_dom, s.buts_ext AS buts_m_ext,
                  s.buts_dom + s.buts_ext AS total_buts
            FROM matchs_v2 m
            JOIN stats_matchs_v2 s ON m.game_id = s.game_id
            WHERE s.buts_dom IS NOT NULL AND s.buts_ext IS NOT NULL
        """)
        df_hist = pd.DataFrame(cursor.fetchall(), columns=["date_match", "equipe_domicile", "equipe_exterieur", "buts_m_dom", "buts_m_ext", "total_buts"])
        
        def calculer_forme(equipe, date_ref):
            matchs = []
            matchs = df_hist[((df_hist["equipe_domicile"] == equipe) | (df_hist["equipe_exterieur"] == equipe)) & (df_hist["date_match"] < date_ref)].sort_values("date_match", ascending=False).head(5)
            if matchs.empty:
                return 0.0, 0.0, 0.0
            buts_marques = np.mean([row["buts_m_dom"] if row["equipe_domicile"] == equipe else row["buts_m_ext"] for _, row in matchs.iterrows()])
            buts_encaisses = np.mean([row["buts_m_ext"] if row["equipe_domicile"] == equipe else row["buts_m_dom"] for _, row in matchs.iterrows()])
            over25 = np.mean([row["total_buts"] > 2.5 for _, row in matchs.iterrows()])
            return buts_marques, buts_encaisses, over25

        # Récupère les 5 derniers matchs de chaque équipe
        def enrichir_forme_complet(equipe, date_ref):
            matchs = df_hist[((df_hist["equipe_domicile"] == equipe) | (df_hist["equipe_exterieur"] == equipe)) & (df_hist["date_match"] < date_ref)].sort_values("date_match", ascending=False).head(5)

            if matchs.empty:
                return 0, 0, 0, 0, 0, 0  # marq, encaissés, over25, std_marq, std_enc, clean_sheets

            weights = np.linspace(1.0, 2.0, len(matchs))
            buts_marques, buts_encaissés, over25, clean_sheets = [], [], [], []

            for _, row in matchs.iterrows():
                est_dom = (row["equipe_domicile"] == equipe)
                bm = row["buts_m_dom"] if est_dom else row["buts_m_ext"]
                be = row["buts_m_ext"] if est_dom else row["buts_m_dom"]

                buts_marques.append(bm)
                buts_encaissés.append(be)
                over25.append(int(row["total_buts"] > 2.5))
                clean_sheets.append(int(be == 0))

            return (
                np.average(buts_marques, weights=weights),
                np.average(buts_encaissés, weights=weights),
                np.average(over25, weights=weights),
                np.std(buts_marques),
                np.std(buts_encaissés),
                np.sum(clean_sheets)
            )

        matchs = []
        seen_game_ids = set()

        for row in rows:
            (
                game_id, date_match, dom, ext,
                buts_dom, enc_dom, over25_dom, over15_dom, btts_dom, pass_pct_dom, pass_reussies_dom,
                poss_dom, corners_dom, fautes_dom, cj_dom, cr_dom, xg_dom, tirs_dom, tirs_cadres_dom,
                clean_sheets_dom, clean_sheets_ext,
                buts_ext, enc_ext, over25_ext, over15_ext, btts_ext, pass_pct_ext, pass_reussies_ext,
                poss_ext, corners_ext, fautes_ext, cj_ext, cr_ext, xg_ext, tirs_ext, tirs_cadres_ext
            ) = row

            if game_id in seen_game_ids: continue
            seen_game_ids.add(game_id)

            fdm, fde, fdo25 = calculer_forme(dom, date_match)
            fem, fee, feo25 = calculer_forme(ext, date_match)

            # Forme complète
            fdm, fde, fdo25, std_marq_dom, std_enc_dom, clean_dom = enrichir_forme_complet(dom, date_match)
            fem, fee, feo25, std_marq_ext, std_enc_ext, clean_ext = enrichir_forme_complet(ext, date_match)

            # Calculs intermédiaires
            diff_xg = to_float(xg_dom - xg_ext)
            sum_xg = to_float(xg_dom + xg_ext)
            sum_btts = to_float(btts_dom + btts_ext)
            diff_over25 = to_float(over25_dom - over25_ext)
            total_tirs = to_float(tirs_dom + tirs_ext)
            total_tirs_cadres = to_float(tirs_cadres_dom + tirs_cadres_ext)

            clean_sheets_dom = 100 - to_float(btts_dom)
            clean_sheets_ext = 100 - to_float(btts_ext)
            solidite_dom = 1 / (to_float(enc_dom) + 0.1)
            solidite_ext = 1 / (to_float(enc_ext) + 0.1)
            solidite_def_dom = to_float(clean_dom) / (to_float(fde) + 1)
            solidite_def_ext = to_float(clean_ext) / (to_float(fee) + 1)

            forme_pond_dom = 0.6 * fdm + 0.4 * fdo25
            forme_pond_ext = 0.6 * fem + 0.4 * feo25

            feature_vector = [
                to_float(buts_dom), to_float(buts_ext), to_float(enc_dom), to_float(enc_ext),
                to_float(over25_dom), to_float(over25_ext), to_float(btts_dom), to_float(btts_ext),
                to_float(xg_dom), to_float(xg_ext), diff_xg, sum_xg,
                fdm, fde, fdo25, fem, fee, feo25,
                sum_btts, diff_over25, total_tirs, total_tirs_cadres,
                clean_sheets_dom, clean_sheets_ext,
                solidite_dom, solidite_ext,
                forme_pond_dom, forme_pond_ext,
                to_float(pass_pct_dom), to_float(pass_pct_ext),
                to_float(corners_dom), to_float(corners_ext),
                to_float(fautes_dom), to_float(fautes_ext)
            ]

            # Construction du dictionnaire temporaire avec les valeurs
            features_dict = {
                "forme_dom_enc": to_float(fde),
                "forme_ext_enc": to_float(fee),
                "std_enc_dom": to_float(std_enc_dom),
                "std_enc_ext": to_float(std_enc_ext),
                "solidite_dom": to_float(solidite_dom),
                "solidite_ext": to_float(solidite_ext),
                "clean_sheets_dom": to_float(clean_sheets_dom),
                "clean_sheets_ext": to_float(clean_sheets_ext),
                "diff_xg": to_float(diff_xg),
                "sum_xg": to_float(sum_xg),
                "total_tirs": to_float(total_tirs),
                "total_tirs_cadres": to_float(total_tirs_cadres),
                "diff_over25": to_float(diff_over25),
                "sum_btts": to_float(sum_btts),
                "forme_dom_marq": to_float(fdm),
                "forme_ext_marq": to_float(fem),
                "std_marq_dom": to_float(std_marq_dom),
                "std_marq_ext": to_float(std_marq_ext),
                "solidite_def_dom": to_float(clean_dom / (fde + 1)),
                "solidite_def_ext": to_float(clean_ext / (fee + 1)),
                "clean_dom": to_float(clean_dom),
                "clean_ext": to_float(clean_ext),
                "moyenne_xg_dom": to_float(xg_dom),
                "moyenne_xg_ext": to_float(xg_ext),
                "buts_encaissés_dom": to_float(enc_dom),
                "buts_encaissés_ext": to_float(enc_ext)
            }


            matchs.append({
                "match": f"{dom} vs {ext}",
                "features": feature_vector,
                "poss": float((poss_dom + poss_ext) / 2),
                "corners": float(corners_dom + corners_ext),
                "fautes": float(fautes_dom + fautes_ext),
                "cartons": float(cj_dom + cr_dom + cj_ext + cr_ext),
                "solidite_dom": float(solidite_dom),
                "solidite_ext": float(solidite_ext),
                "clean_sheets_dom": float(clean_sheets_dom),
                "clean_sheets_ext": float(clean_sheets_ext)
            })

        cursor.close()
        return matchs

    matchs_jour = get_matchs_jour_for_prediction()

    # === Prédictions ===
    # === Ajout cluster_type (avant scaler)
    X_live = pd.DataFrame([m["features"] for m in matchs_jour], columns=features)  # 34 colonnes

    # ✅ Transformation
    X_live = pd.DataFrame([m["features"] for m in matchs_jour], columns=features)
    X_live_scaled = scaler_ml.transform(X_live)
    
    # prédiction principale
    preds_cat = model_cat.predict(X_live_scaled)
    preds_hgb = model_hgb.predict(X_live_scaled)
    # Prédiction des bornes conformal
    pred_p25 = model_p25.predict(X_live_scaled)
    pred_p75 = model_p75.predict(X_live_scaled)
    
    matchs_ouverts = []
    matchs_fermes = []
    matchs_neutres = []

    # combinaison pondérée
    pred_buts = 0.7 * preds_cat + 0.3 * preds_hgb

    for i, match in enumerate(matchs_jour):
        features_vec = match["features"]
        pred_total = pred_buts[i]
        p25 = pred_p25[i]
        p75 = pred_p75[i]
    
        # Variables heuristiques
        buts_dom, buts_ext = float(features_vec[0]), float(features_vec[1])
        over25_dom, over25_ext = float(features_vec[4]), float(features_vec[5])
        btts_dom, btts_ext = float(features_vec[6]), float(features_vec[7])
        xg_dom, xg_ext = float(features_vec[8]), float(features_vec[9])
        tirs_cadres_total = float(features_vec[21])
        fdm, fdo25 = float(features_vec[12]), float(features_vec[14])
        fem, feo25 = float(features_vec[15]), float(features_vec[17])
        forme_pond_dom = 0.6 * fdm + 0.4 * fdo25
        forme_pond_ext = 0.6 * fem + 0.4 * feo25
        solidite_dom = float(100 - match.get("clean_sheets_dom", 0))
        solidite_ext = float(100 - match.get("clean_sheets_ext", 0))
        poss = float(match.get("poss", 50))
        corners = float(match.get("corners", 8))
        fautes = float(match.get("fautes", 20))
        cartons = float(match.get("cartons", 3))
    
        score_heuristique = (
            0.10 * (buts_dom + buts_ext) +
            0.10 * (over25_dom + over25_ext) +
            0.08 * (btts_dom + btts_ext) +
            0.08 * (xg_dom + xg_ext) +
            0.05 * tirs_cadres_total +
            0.12 * forme_pond_dom + 0.12 * forme_pond_ext -
            0.10 * solidite_dom - 0.10 * solidite_ext +
            0.03 * corners + 0.03 * fautes + 0.02 * cartons + 0.05 * poss
        )

        def convertir_pred_en_score_heuristique(pred_total):
          if pred_total <= 2:
              return 60
          elif pred_total <= 3:
              return 70
          elif pred_total <= 4:
              return 80 
          elif pred_total <= 5:
              return 90 
          else:
              return 100
    
        gmos_score = compute_gmos(pred_total, p25, p75, score_heuristique)
        match["gmos_score"] = gmos_score
    
        ligne = (
            f"🔮 GMOS : {gmos_score}\n"
            f"📊 Estimé entre {int(p25)} et {int(p75)} buts\n"
            f"📈 Prédiction (CAT/LGB/XGB) : {pred_buts[i]:.2f}\n"
        )
    
        if gmos_score >= 65:
            matchs_ouverts.append((gmos_score, match["match"], ligne))
        elif gmos_score <= 50:
            matchs_fermes.append((gmos_score, match["match"], ligne))
        else:
            matchs_neutres.append((gmos_score, match["match"], ligne))

    
    # === Génération du mail GMOS ===
    mail_lines = [f"📅 Prévisions du {today}\n"]
    mail_lines.append("📈 Top 5 Matchs Ouverts (GMOS ≥ 65)\n")
    
    for idx, (_, name, ligne) in enumerate(sorted(matchs_ouverts, reverse=True)[:5], 1):
        mail_lines.append(f"{idx}️⃣ {name}\n{ligne}\n")
    if not matchs_ouverts:
        mail_lines.append("Aucun match ouvert détecté aujourd’hui ❄️\n")
    
    mail_lines.append("🔒 Top 5 Matchs Fermés (GMOS ≤ 50)\n")
    for idx, (_, name, ligne) in enumerate(sorted(matchs_fermes)[:5], 1):
        mail_lines.append(f"{idx}️⃣ {name}\n{ligne}\n")
    if not matchs_fermes:
        mail_lines.append("Aucun match fermé détecté aujourd’hui.\n")
    
    mail_lines.append("⚪ Matchs Neutres (GMOS entre 51 et 64)\n")
    for idx, (_, name, ligne) in enumerate(sorted(matchs_neutres, reverse=True), 1):
        mail_lines.append(f"{idx}️⃣ {name}\n{ligne}\n")
    if not matchs_neutres:
        mail_lines.append("Aucun match neutre aujourd’hui.\n")
    
    mail_lines.append("🔥 GMOS = le meilleur résumé de tous tes modèles 💡")
    mail_lines.append("🧠 Cluster = profil historique | GMOS = prédictions + forme actuelle")
    mail_lines.append("Suivi : https://docs.google.com/forms/d/e/1FAIpQLSdRKd8ui1gy8lNfhMYYsLesglR9JJeAI7VgqrASbr0Ocdl7Tg/viewform?usp=header")
    
    send_email(
        subject="📊 Analyse quotidienne GMOS (Top matchs ouverts/fermés/neutres)",
        body="\n".join(mail_lines),
        to_email = ["lilian.pamphile.bts@gmail.com"]
    )


# Gestion erreur
except Exception as e:
    error_message = f"❌ Erreur durant l’exécution du script main du {today} :\n\n{str(e)}"
    send_email(
        subject="❌ Échec - Script Main",
        body=error_message,
        to_email="lilian.pamphile.bts@gmail.com"
    )
