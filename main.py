# -*- coding: utf-8 -*-
import requests
import psycopg2
# Fonction de conversion sécurisée
from datetime import datetime, timedelta
import os
import pickle
import pandas as pd
import numpy as np
import shutil
import subprocess

from telegram_message import send_telegram_message

# 🔑 Clé API SportsData.io
API_KEY = os.getenv("RAPIDAPI_KEY")

today = datetime.today().date()
yesterday = today - timedelta(days=1)
annee = datetime.now().year
saison1 = annee
saison2 = annee - 1

# 🏆 Liste des compétitions à récupérer
COMPETITIONS = {
    # ——— Clubs (déjà présents) ———
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
    "Saudi Professional League": "307",

    # ——— Coupes nationales (Top Europe) ———
    "FA Cup": "45",
    "League Cup": "48",
    "DFB Pokal": "81",
    "Coppa Italia": "137",

    # ——— Sélections & grandes compétitions internationales ———
    "World Cup": "1",
    "Euro Championship": "4",
    "Euro Championship - Qualification": "960",
    "UEFA Nations League": "5",
    "UEFA Super Cup": "531",
    "FIFA Club World Cup": "15",
    "FIFA Club World Cup - Play-In": "1186",

    # ——— Amicaux ———
    "Friendlies": "10",

    # ——— Qualifications Coupe du Monde ———
    "World Cup - Qualification Africa": "29",
    "World Cup - Qualification Asia": "30",
    "World Cup - Qualification CONCACAF": "31",
    "World Cup - Qualification Europe": "32",
    "World Cup - Qualification Oceania": "33",
    "World Cup - Qualification South America": "34",
    "World Cup - Qualification Intercontinental Play-offs": "37",

    # ——— Autres tournois sélections ———
    "CONCACAF Gold Cup": "22",
    "UEFA U21 Championship": "38",
    "UEFA U21 Championship - Qualification": "850"
}

# 🔌 Connexion PostgreSQL Railway
DATABASE_URL = "postgresql://postgres:jDDqfaqpspVDBBwsqxuaiSDNXjTxjMmP@shortline.proxy.rlwy.net:36536/railway"
conn = psycopg2.connect(DATABASE_URL)
cursor = conn.cursor()

print("Fin de la défintion de variables")

# --- Défauts safe si une exception survient plus tard ---
matchs_jour = []
matchs_over, matchs_under, matchs_opps = [], [], []
probas_over25 = np.array([])
pred_p25 = np.array([])
pred_p75 = np.array([])

################################### Fontions utiles ###################################

def to_float(x):
    try:
        v = float(x)
        return v
    except:
        return np.nan

# === Helpers MarkdownV2 (monospace) ===
MD_SPECIAL = r"_*[]()~`>#+-=|{}.!"

def mdv2_escape(s: str) -> str:
    s = str(s)
    out = []
    for ch in s:
        out.append("\\" + ch if ch in MD_SPECIAL else ch)
    return "".join(out)

def num(x, default=0.0):
    """Convertit Decimal/float/int/str en float, remplace None ou NaN par default."""
    if x is None:
        return default
    try:
        v = float(x)
        if np.isnan(v):
            return default
        return v
    except Exception:
        return default


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

# --- Téléchargement des fichiers modèle/scaler depuis GitHub ---
def telecharger_model_depuis_github():
    REPO = "LilianPamphile/xgenius"
    BRANCH = "main"
    TOKEN = os.getenv("TOKEN_HUB")

    fichiers = {
        "model_files/model_total_buts_catboost_optuna.pkl": "model_files/model_total_buts_catboost_optuna.pkl",
        "model_files/model_total_buts_hist_gradient_boosting.pkl": "model_files/model_total_buts_hist_gradient_boosting.pkl",
        "model_files/model_total_buts_conformal_p25.pkl": "model_files/model_total_buts_conformal_p25.pkl",
        "model_files/model_total_buts_conformal_p75.pkl": "model_files/model_total_buts_conformal_p75.pkl",
        "model_files/scaler_total_buts.pkl": "model_files/scaler_total_buts.pkl",
        "model_files/features_list.pkl": "model_files/features_list.pkl",
        "model_files/regression_score_heuristique.pkl": "model_files/regression_score_heuristique.pkl",
        "model_files/features_list_score_heuristique.pkl": "model_files/features_list_score_heuristique.pkl",
        "model_files/model_over25_classifier.pkl": "model_files/model_over25_classifier.pkl",
        "model_files/offset_conformal.pkl": "model_files/offset_conformal.pkl",
        "model_files/mae_models.pkl": "model_files/mae_models.pkl",
    }
    for dist, local in fichiers.items():
        url = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/{dist}"
        headers = {"Authorization": f"token {TOKEN}"} if TOKEN else {}
        os.makedirs(os.path.dirname(local), exist_ok=True)
        r = requests.get(url, headers=headers)
        if r.status_code == 200:
            with open(local, "wb") as f: f.write(r.content)
            print(f"✅ {local}")
        else:
            print(f"❌ {local} ({r.status_code})")

###################################################################################################

# === 📌 1️⃣ Récupération des Matchs ===
def recuperer_matchs(date, API_KEY):
    url_base = "https://api-football-v1.p.rapidapi.com/v3/fixtures"

    headers = {
        "x-rapidapi-key": API_KEY,
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com"
    }

    saison_api = saison1
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

    headers = {
        "x-rapidapi-key": API_KEY,
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com"
    }

    print(f"📅 Stats pour {date}")

    for competition_name, competition_id in COMPETITIONS.items():
        params = {
            "league": competition_id,
            "season": saison1,
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
                extract_stat(stats_dom, 'Red Cards'), extract_stat(stats_ext, 'Red Cards'), 
                xg_dom, xg_ext
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
                "cartons_jaunes_dom, cartons_jaunes_ext, cartons_rouges_dom, cartons_rouges_ext,"
                "xg_dom, xg_ext"
                ") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
                "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
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

            xg_for = (data.get("xg_dom") or 0) if est_domicile else (data.get("xg_ext") or 0)
            xg_against = (data.get("xg_ext") or 0) if est_domicile else (data.get("xg_dom") or 0)
            total.setdefault("xg_for", 0);       total["xg_for"] += xg_for
            total.setdefault("xg_against", 0);   total["xg_against"] += xg_against

            def avg(val):
                return round(val / total["matchs_joues"], 2) if total["matchs_joues"] else 0.0
            
            avg_xg_for = avg(total.get("xg_for", 0))
            avg_xg_against = avg(total.get("xg_against", 0))

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
            avg_xg_for, avg_xg_against

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
    with open("model_files/regression_score_heuristique.pkl", "rb") as f:
        model_heuristique = pickle.load(f)
        
    # Charger les features spécifiques au modèle heuristique
    with open("model_files/features_list_score_heuristique.pkl", "rb") as f:
        features_heur = pickle.load(f)

    with open("model_files/model_over25_classifier.pkl", "rb") as f:
        model_over25 = pickle.load(f)

    # Charger le OFFSET dynamique
    with open("model_files/offset_conformal.pkl", "rb") as f:
        OFFSET = pickle.load(f)

    def get_matchs_jour_for_prediction():
        cursor = conn.cursor()

        # 1. Récupère les stats globales nécessaires
        query = """
        SELECT
          m.game_id,
          m.date::date AS date_match,
          m.competition,
          m.equipe_domicile, m.equipe_exterieur,
        
          /* ======= ÉQUIPE DOMICILE : agrégée 2 saisons (poids: saison1=2, saison2=1) ======= */
          sg1.moyenne_buts,
          sg1.encaisse_pm,
          sg1.pourcentage_over_2_5,
          sg1.pourcentage_over_1_5,
          sg1.pourcentage_btts,
          sg1.passes_pourcent,
          sg1.passes_reussies,
          sg1.possession,
          sg1.corners,
          sg1.fautes,
          sg1.cartons_jaunes,
          sg1.cartons_rouges,
          sg1.moyenne_xg_dom,
          sg1.tirs,
          sg1.tirs_cadres,
          sg1.pourcentage_clean_sheets,
          sg2.pourcentage_clean_sheets,
        
          /* ======= ÉQUIPE EXTÉRIEUR : agrégée 2 saisons (poids: saison1=2, saison2=1) ======= */
          sg2.moyenne_buts,
          sg2.encaisse_pm,
          sg2.pourcentage_over_2_5,
          sg2.pourcentage_over_1_5,
          sg2.pourcentage_btts,
          sg2.passes_pourcent,
          sg2.passes_reussies,
          sg2.possession,
          sg2.corners,
          sg2.fautes,
          sg2.cartons_jaunes,
          sg2.cartons_rouges,
          sg2.moyenne_xg_ext,
          sg2.tirs,
          sg2.tirs_cadres
        
        FROM matchs_v2 m
        
        /* ---------- Agrégat pondéré pour l'équipe domicile ---------- */
        LEFT JOIN LATERAL (
          SELECT
            SUM(w * s.moyenne_buts) / NULLIF(SUM(w), 0)                               AS moyenne_buts,
            SUM(w * s.buts_encaisse)::float / NULLIF(SUM(w * s.matchs_joues), 0)      AS encaisse_pm,
            SUM(w * s.pourcentage_over_2_5) / NULLIF(SUM(w), 0)                        AS pourcentage_over_2_5,
            SUM(w * s.pourcentage_over_1_5) / NULLIF(SUM(w), 0)                        AS pourcentage_over_1_5,
            SUM(w * s.pourcentage_btts)   / NULLIF(SUM(w), 0)                          AS pourcentage_btts,
            SUM(w * s.passes_pourcent)    / NULLIF(SUM(w), 0)                          AS passes_pourcent,
            SUM(w * s.passes_reussies)    / NULLIF(SUM(w), 0)                          AS passes_reussies,
            SUM(w * s.possession)         / NULLIF(SUM(w), 0)                          AS possession,
            SUM(w * s.corners)            / NULLIF(SUM(w), 0)                          AS corners,
            SUM(w * s.fautes)             / NULLIF(SUM(w), 0)                          AS fautes,
            SUM(w * s.cartons_jaunes)     / NULLIF(SUM(w), 0)                          AS cartons_jaunes,
            SUM(w * s.cartons_rouges)     / NULLIF(SUM(w), 0)                          AS cartons_rouges,
            SUM(w * s.moyenne_xg_dom)     / NULLIF(SUM(w), 0)                          AS moyenne_xg_dom,
            SUM(w * s.tirs)               / NULLIF(SUM(w), 0)                          AS tirs,
            SUM(w * s.tirs_cadres)        / NULLIF(SUM(w), 0)                          AS tirs_cadres,
            SUM(w * s.pourcentage_clean_sheets) / NULLIF(SUM(w), 0)                    AS pourcentage_clean_sheets
          FROM (
            SELECT g.*,
                   CASE WHEN g.saison = %s THEN 2.0
                        WHEN g.saison = %s THEN 1.0
                        ELSE 0.0 END AS w
            FROM stats_globales_v2 g
            WHERE g.equipe = m.equipe_domicile
              AND g.saison IN (%s, %s)
          ) s
        ) sg1 ON TRUE
        
        /* ---------- Agrégat pondéré pour l'équipe extérieur ---------- */
        LEFT JOIN LATERAL (
          SELECT
            SUM(w * s.moyenne_buts) / NULLIF(SUM(w), 0)                               AS moyenne_buts,
            SUM(w * s.buts_encaisse)::float / NULLIF(SUM(w * s.matchs_joues), 0)      AS encaisse_pm,
            SUM(w * s.pourcentage_over_2_5) / NULLIF(SUM(w), 0)                        AS pourcentage_over_2_5,
            SUM(w * s.pourcentage_over_1_5) / NULLIF(SUM(w), 0)                        AS pourcentage_over_1_5,
            SUM(w * s.pourcentage_btts)   / NULLIF(SUM(w), 0)                          AS pourcentage_btts,
            SUM(w * s.passes_pourcent)    / NULLIF(SUM(w), 0)                          AS passes_pourcent,
            SUM(w * s.passes_reussies)    / NULLIF(SUM(w), 0)                          AS passes_reussies,
            SUM(w * s.possession)         / NULLIF(SUM(w), 0)                          AS possession,
            SUM(w * s.corners)            / NULLIF(SUM(w), 0)                          AS corners,
            SUM(w * s.fautes)             / NULLIF(SUM(w), 0)                          AS fautes,
            SUM(w * s.cartons_jaunes)     / NULLIF(SUM(w), 0)                          AS cartons_jaunes,
            SUM(w * s.cartons_rouges)     / NULLIF(SUM(w), 0)                          AS cartons_rouges,
            SUM(w * s.moyenne_xg_ext)     / NULLIF(SUM(w), 0)                          AS moyenne_xg_ext,
            SUM(w * s.tirs)               / NULLIF(SUM(w), 0)                          AS tirs,
            SUM(w * s.tirs_cadres)        / NULLIF(SUM(w), 0)                          AS tirs_cadres,
            SUM(w * s.pourcentage_clean_sheets) / NULLIF(SUM(w), 0)                    AS pourcentage_clean_sheets
          FROM (
            SELECT g.*,
                   CASE WHEN g.saison = %s THEN 2.0
                        WHEN g.saison = %s THEN 1.0
                        ELSE 0.0 END AS w
            FROM stats_globales_v2 g
            WHERE g.equipe = m.equipe_exterieur
              AND g.saison IN (%s, %s)
          ) s
        ) sg2 ON TRUE
        
        WHERE DATE(m.date) = %s
        """

        cursor.execute(
            query,
            (
                # sg1 (domicile)
                saison1, saison2, saison1, saison2,
                # sg2 (extérieur)
                saison1, saison2, saison1, saison2,
                # filtre de date
                today,
            ),
        )
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
        
        def calculer_forme(equipe, date_ref, n=5, role=None, decay=0.85):
            q = (df_hist["date_match"] < date_ref)
            if role is None:
                q &= ((df_hist["equipe_domicile"] == equipe) | (df_hist["equipe_exterieur"] == equipe))
            elif role is True:
                q &= (df_hist["equipe_domicile"] == equipe)
            elif role is False:
                q &= (df_hist["equipe_exterieur"] == equipe)
        
            m = df_hist.loc[q].sort_values("date_match", ascending=False).head(n)
            if m.empty:
                return 0.0, 0.0, 0.0
        
            est_dom = (m["equipe_domicile"].values == equipe)
            bm = m["buts_m_dom"].values * est_dom + m["buts_m_ext"].values * (~est_dom)
            be = m["buts_m_ext"].values * est_dom + m["buts_m_dom"].values * (~est_dom)
            tb = m["total_buts"].values
        
            # Pondération temporelle
            w = decay ** np.arange(len(m))
            w /= w.sum()
        
            return (
                float(np.average(bm, weights=w)),
                float(np.average(be, weights=w)),
                float(np.average((tb > 2.5).astype(float), weights=w))
            )
        
        
        def enrichir_forme_complet(equipe, date_ref):

            matchs = df_hist[
                ((df_hist["equipe_domicile"] == equipe) | (df_hist["equipe_exterieur"] == equipe)) &
                (df_hist["date_match"] < date_ref)
            ].sort_values("date_match", ascending=False).head(5)
        
            if matchs.empty:
                return 0, 0, 0, 0, 0, 0
        
            weights = np.linspace(1.0, 2.0, len(matchs))
            buts_marques, buts_encaisses, over25, clean_sheets = [], [], [], []
        
            for _, row in matchs.iterrows():
                est_dom = (row["equipe_domicile"] == equipe)
                bm = row["buts_m_dom"] if est_dom else row["buts_m_ext"]
                be = row["buts_m_ext"] if est_dom else row["buts_m_dom"]
        
                buts_marques.append(bm)
                buts_encaisses.append(be)
                over25.append(int(row["total_buts"] > 2.5))
                clean_sheets.append(int(be == 0))
        
            return (
                np.average(buts_marques, weights=weights),
                np.average(buts_encaisses, weights=weights),
                np.average(over25, weights=weights),
                np.std(buts_marques),
                np.std(buts_encaisses),
                np.sum(clean_sheets)
            )
        
        
        matchs = []
        seen_game_ids = set()
        
        for row in rows:
            (
                game_id, date_match, competition, dom, ext,
                buts_dom, enc_dom, over25_dom, over15_dom, btts_dom, pass_pct_dom, pass_reussies_dom,
                poss_dom, corners_dom, fautes_dom, cj_dom, cr_dom, xg_dom, tirs_dom, tirs_cadres_dom,
                clean_sheets_dom, clean_sheets_ext,
                buts_ext, enc_ext, over25_ext, over15_ext, btts_ext, pass_pct_ext, pass_reussies_ext,
                poss_ext, corners_ext, fautes_ext, cj_ext, cr_ext, xg_ext, tirs_ext, tirs_cadres_ext
            ) = row
        
            if game_id in seen_game_ids:
                continue
            seen_game_ids.add(game_id)
        
            # --- Forme split domicile / extérieur ---
            bm_home, be_home, o25_home = calculer_forme(dom, date_match, role=True)
            bm_away, be_away, o25_away = calculer_forme(ext, date_match, role=False)
            
            # --- Forme complète (pour std & clean sheets réels) ---
            _, _, _, std_marq_dom, std_enc_dom, clean_dom = enrichir_forme_complet(dom, date_match)
            _, _, _, std_marq_ext, std_enc_ext, clean_ext = enrichir_forme_complet(ext, date_match)
            
            # --- Possession moyenne ---
            poss_moyenne = (num(poss_dom) + num(poss_ext)) / 2.0
            
            # --- Cartons total ---
            cartons_total = num(cj_dom) + num(cr_dom) + num(cj_ext) + num(cr_ext)
            
            # --- Construction du dictionnaire optimisé ---
            features_dict = {
                # Forme split
                "forme_home_buts_marques": to_float(bm_home),
                "forme_home_buts_encaisses": to_float(be_home),
                "forme_home_over25": to_float(o25_home),
                "forme_away_buts_marques": to_float(bm_away),
                "forme_away_buts_encaisses": to_float(be_away),
                "forme_away_over25": to_float(o25_away),
            
                # Variabilité
                "std_marq_dom": to_float(std_marq_dom),
                "std_enc_dom": to_float(std_enc_dom),
                "std_marq_ext": to_float(std_marq_ext),
                "std_enc_ext": to_float(std_enc_ext),
            
                # Clean sheets réels
                "clean_dom": to_float(clean_dom),
                "clean_ext": to_float(clean_ext),
            
                # Expected Goals bruts
                "moyenne_xg_dom": to_float(xg_dom),
                "moyenne_xg_ext": to_float(xg_ext),
            
                # Buts encaissés bruts
                "buts_encaissés_dom": to_float(enc_dom),
                "buts_encaissés_ext": to_float(enc_ext),
            
                # Possession et tirs
                "poss_moyenne": to_float(poss_moyenne),
                "tirs_dom": to_float(tirs_dom),
                "tirs_ext": to_float(tirs_ext),
                "tirs_cadres_dom": to_float(tirs_cadres_dom),
                "tirs_cadres_ext": to_float(tirs_cadres_ext),
            
                # Discipline et corners
                "corners_dom": to_float(corners_dom),
                "corners_ext": to_float(corners_ext),
                "cartons_total": to_float(cartons_total)
            }
            
            # --- Vector de features final ---
            feature_vector = [features_dict.get(f, 0.0) for f in features]
            
            # --- Stockage match ---
            matchs.append({
                "match": f"{dom} vs {ext}",
                "competition": competition,
                "features": feature_vector,
                "poss": poss_moyenne,
                "corners": num(corners_dom) + num(corners_ext),
                "fautes": num(fautes_dom) + num(fautes_ext),
                "cartons": cartons_total,
                "clean_sheets_dom": float(clean_dom),
                "clean_sheets_ext": float(clean_ext)
            })
            
                    
        cursor.close()
        return matchs


    matchs_jour = get_matchs_jour_for_prediction()

    # === Prédictions ===
    # ✅ Transformation
    X_live = pd.DataFrame([m["features"] for m in matchs_jour], columns=features)

    # 1) Tout en numérique + inf -> NaN
    X_live = X_live.apply(pd.to_numeric, errors="coerce")
    X_live.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 2) Médianes colonne par colonne
    col_medians = X_live.median(numeric_only=True)
    X_live = X_live.fillna(col_medians)
    
    # 3) Filet de sécurité (si une médiane vaut encore NaN → 0.0)
    X_live = X_live.fillna(0.0)
    
    # (optionnel) Sanity check
    if np.isnan(X_live.values).any():
        raise ValueError("Still NaN in X_live after imputation.")
    
    X_live_scaled = scaler_ml.transform(X_live)

    # prédiction principale
    preds_cat = model_cat.predict(X_live_scaled)
    preds_hgb = model_hgb.predict(X_live_scaled)
    # Prédiction des bornes conformal
    pred_p25 = model_p25.predict(X_live_scaled) - OFFSET
    pred_p75 = model_p75.predict(X_live_scaled) + OFFSET


    # Prédiction classification over 2.5
    try:
        probas_over25 = model_over25.predict_proba(X_live_scaled)[:, 1]
    except Exception as e:
        print("⚠️ Over2.5 classifier failed, fallback to 0.5:", e)
        probas_over25 = np.full(shape=(len(X_live_scaled),), fill_value=0.5, dtype=float)
    

    # === Pondération par LIGUE (biais de modèles) + micro-biais de buts (baseline ligue)
    LEAGUE_MODEL_WEIGHTS = {
        "Ligue 1": (0.60, 0.40),
        "Bundesliga": (0.40, 0.60),
        "Eredivisie": (0.45, 0.55),
        "Premier League": (0.50, 0.50),
        "Serie A": (0.50, 0.50),
        "La Liga": (0.55, 0.45),
    }
    LEAGUE_GOAL_BIAS = {
        # petit ajustement de tendance de ligue (en buts)
        "Bundesliga": +0.10,
        "Eredivisie": +0.10,
        "Premier League": +0.05,
        "Serie A": +0.05,
        "La Liga": -0.05,
        "Ligue 1": -0.10,
    }

    matchs_over, matchs_under, matchs_opps = [], [], []

    def clip01(x):
        return max(0.0, min(1.0, float(x)))

    for i, match in enumerate(matchs_jour):
        comp = match.get("competition", "")
    
        # d’abord on calcule l’incertitude
        p25, p75 = float(pred_p25[i]), float(pred_p75[i])
        incertitude = p75 - p25
        prob_over25 = float(probas_over25[i])
        prob_under25 = 1.0 - prob_over25
    
        # puis on fait la pondération adaptative Cat/HGB
        w_cat0, w_hgb0 = LEAGUE_MODEL_WEIGHTS.get(comp, (0.50, 0.50))
        if incertitude <= 1.2:
            k = 0.0
        elif incertitude >= 2.0:
            k = 1.0
        else:
            k = (incertitude - 1.2) / (2.0 - 1.2)
    
        w_cat = (1.0 - k)*w_cat0 + k*0.50
        w_hgb = 1.0 - w_cat
    
        pred_mix = w_cat * float(preds_cat[i]) + w_hgb * float(preds_hgb[i])
        pred_total = float(pred_mix) + float(LEAGUE_GOAL_BIAS.get(comp, 0.0))

        # ligne propre (non-scalée) pour features heuristiques déjà construite au-dessus
        row = X_live.iloc[i]
        enc_dom_val = float(row.get("buts_encaissés_dom", 0.0))
        enc_ext_val = float(row.get("buts_encaissés_ext", 0.0))
        solidite_dom = 1.0 / (enc_dom_val + 0.1)
        solidite_ext = 1.0 / (enc_ext_val + 0.1)
    
        corners_total = float(match.get("corners", num(row.get("corners_dom", 0.0)) + num(row.get("corners_ext", 0.0))))
        fautes_total  = float(match.get("fautes", num(row.get("fautes_dom", 0.0)) + num(row.get("fautes_ext", 0.0))))
        cartons_total = float(match.get("cartons", row.get("cartons_total", 0.0)))
        poss_moy      = float(match.get("poss", row.get("poss_moyenne", 50.0)))
    
        def getf(name, default=0.0):
            v = row.get(name, default)
            return float(v if pd.notna(v) else default)
    
        d = {
            "buts_dom": getf("forme_home_buts_marques"),
            "buts_ext": getf("forme_away_buts_marques"),
            "over25_dom": getf("forme_home_over25"),
            "over25_ext": getf("forme_away_over25"),
            "btts_dom": 0.0, "btts_ext": 0.0,
            "moyenne_xg_dom": getf("moyenne_xg_dom"),
            "moyenne_xg_ext": getf("moyenne_xg_ext"),
            "tirs_cadres_total": getf("tirs_cadres_dom") + getf("tirs_cadres_ext"),
            "forme_dom_marq": getf("forme_home_buts_marques"),
            "forme_ext_marq": getf("forme_away_buts_marques"),
            "solidite_dom": solidite_dom, "solidite_ext": solidite_ext,
            "corners": corners_total, "fautes": fautes_total,
            "cartons_total": cartons_total, "poss": poss_moy,
        }
    
        # Vérif que toutes les features attendues sont présentes
        missing_feats = [f for f in features_heur if f not in d]
        if missing_feats:
            print(f"[WARN] Features manquantes pour score_heur: {missing_feats}")
    
        # Construction dans le bon ordre
        X_input_heur = pd.DataFrame([[d.get(f, 0.0) for f in features_heur]], columns=features_heur)
    
        # Prédiction heuristique
        score_heur = float(model_heuristique.predict(X_input_heur)[0])
    
        # Contrôle bornes et clamp
        if score_heur < -0.05 or score_heur > 2:
            print(f"[WARN] score_heur aberrant ({score_heur:.2f}) pour {match.get('match', 'inconnu')}")
        score_heur = max(0.0, min(score_heur, 1.5))
        match["score_heur"] = score_heur
    
        # === Confiance composite (plus de poids à l'intervalle)
        sharp = 1.0 - min(1.0, incertitude / 3.0)
        prob_strength = abs(prob_over25 - 0.5) * 2.0
        margin_strength = min(1.0, abs(pred_total - 2.5) / 2.0)
        heur_pct = score_heur / 1.5
        agree_bits = [
            1.0 if pred_total >= 2.5 else 0.0,
            1.0 if prob_over25 >= 0.5 else 0.0,
            1.0 if heur_pct >= 0.5 else 0.0,
        ]
        agreement_ratio = sum(agree_bits) / 3.0
        consistency = abs(agreement_ratio - 0.5) * 2.0
        # → on renforce l’impact de l’intervalle
        w1, w2, w3, w4 = 0.25, 0.30, 0.25, 0.20
        confidence_score = w1*sharp + w2*prob_strength + w3*margin_strength + w4*consistency
        confidence_pct = int(round(confidence_score * 100))
    
        # Catégorie d’intervalle (réintroduit comme signal lisible)
        if incertitude < 1.2:
            bucket_ci = "sharp"
        elif incertitude <= 2.0:
            bucket_ci = "moyen"
        else:
            bucket_ci = "flou"
    
        # Bandes de proba (seulement interprétation)
        if prob_over25 >= 0.68:
            band_proba = "forte"
        elif prob_over25 >= 0.55:
            band_proba = "moyenne"
        else:
            band_proba = "faible"
    
        if confidence_score >= 0.68:
            commentaire = f"✅ Confiance élevée ({confidence_pct}%) • {bucket_ci} • proba {band_proba}"
        elif confidence_score >= 0.50:
            commentaire = f"ℹ️ Confiance modérée ({confidence_pct}%) • {bucket_ci} • proba {band_proba}"
        else:
            commentaire = f"⚠️ Confiance faible ({confidence_pct}%) • {bucket_ci} • proba {band_proba}"
    
        match["confiance"] = commentaire
        match["confidence_pct"] = confidence_pct
        match["bucket_ci"] = bucket_ci
        match["band_proba"] = band_proba
        match["pred_total"] = pred_total

        # === OpenScore / CloseScore + drivers lisibles (pour Telegram)
        def nz(x, d=0.0):
            try:
                v = float(x)
                if np.isnan(v):
                    return d
                return v
            except:
                return d
        
        # On réutilise 'row' (déjà défini), 'match', et clip01() existant
        xg_tot = nz(row.get("moyenne_xg_dom")) + nz(row.get("moyenne_xg_ext"))
        tc_tot = nz(row.get("tirs_cadres_dom")) + nz(row.get("tirs_cadres_ext"))
        fm_dom = nz(row.get("forme_home_buts_marques"))
        fm_ext = nz(row.get("forme_away_buts_marques"))
        be_dom = nz(row.get("buts_encaissés_dom"))
        be_ext = nz(row.get("buts_encaissés_ext"))
        
        # Composantes normalisées (0–1)
        SO_xg  = clip01(xg_tot / 3.5)                 # signal offensif via xG
        SO_tc  = clip01(tc_tot / 10.0)                # signal via tirs cadrés
        SO_fm  = clip01((fm_dom + fm_ext) / 3.0)      # forme offensive
        SO     = (SO_xg + SO_tc + SO_fm) / 3.0
        
        SDm    = clip01((be_dom + be_ext) / 3.0)      # faiblesse défensive (inverse de solidité)
        solid  = clip01(((1.0/(be_dom+0.1)) + (1.0/(be_ext+0.1))) / 2.0)
        
        pos_moy = nz(match.get("poss"), 50.0)         # déjà moyenne dom/ext plus haut
        rt_pos  = clip01(abs(pos_moy - 50.0) / 20.0)  # déséquilibre vs neutre 50/50
        rt_int  = clip01((corners_total/14.0 + fautes_total/30.0) / 2.0)
        RT      = (rt_pos + rt_int) / 2.0
        
        
        # Drivers compacts (on en garde 2–3 max)
        drivers = []
        if SO_xg >= 0.60: drivers.append("xG↑")
        if SO_tc >= 0.60: drivers.append("tirs cadrés↑")
        if rt_pos >= 0.60: drivers.append("déséquilibre pos↑")
        if rt_int >= 0.60: drivers.append("intensité↑")
        if SDm   >= 0.60: drivers.append("défenses friables")
        if solid >= 0.60: drivers.append("solidités↑")
        if SO    <= 0.40: drivers.append("signal offensif↓")
        if RT    <= 0.40: drivers.append("rythme↓")
        
        drivers_str = ", ".join(drivers[:3])
        
        # On enrichit le texte 'commentaire' utilisé dans les tableaux Telegram
        commentaire = f"{commentaire} • {drivers_str}".strip()
        match["drivers"]    = drivers_str
        match["confiance"]  = commentaire   # <-- pour que le CSV récupère aussi OS/CS/drivers

        # === Score global continu (remplace les règles OR)
        score_global = 0.50*prob_over25 + 0.30*min(pred_total/4.0, 1.0) + 0.20*heur_pct
        score_global -= 0.10 * max(0.0, (incertitude - 1.5))  # pénalité si intervalle large
        score_global = clip01(score_global)
        match["score_global"] = float(score_global)

        META = int(round(100 * score_global))
        match["META"] = META
    
        if score_global >= 0.68:
            matchs_over.append((
                prob_over25, match["match"], pred_total, f"{p25:.2f} – {p75:.2f}",
                commentaire, score_heur, confidence_pct, META
            ))
        elif score_global <= 0.38:
            matchs_under.append((
                prob_under25, match["match"], pred_total, f"{p25:.2f} – {p75:.2f}",
                commentaire, score_heur, confidence_pct, META
            ))
        else:
            matchs_opps.append((
                prob_over25, match["match"], pred_total, f"{p25:.2f} – {p75:.2f}",
                commentaire, score_heur, confidence_pct, META
            ))

except Exception as e:
    print("❌ Erreur pendant la génération des prédictions :", e)

def build_table(title_emoji: str, title_text: str, rows, is_under: bool = False) -> str:
    """
    Rend un bloc compact adapté mobile :
    - 2 lignes par match
    - Ligne 1 : Match + G (buts attendus)
    - Ligne 2 : O2.5 / U2.5 / XGS / Conf + OS/CS + drivers
    """
    header = f"*{mdv2_escape(title_emoji + ' ' + title_text)}*"
    if not rows:
        return f"{header}\n_Aucun match détecté._\n"

    # Tri par le dernier champ (confiance %) si présent, sinon par proba
    try:
        ordered = sorted(rows, key=lambda x: x[-1], reverse=True)
    except Exception:
        ordered = rows

    lines = [header]
    lines.append("```")  # bloc monospace pour éviter les retours hasardeux

    for prob, name, pred, intervalle, conf_txt, heur, *rest in ordered:
        prob_pct = int(round(float(prob) * 100))
        label = "Proba Under2.5" if is_under else "Proba Over2.5"

        try:
            p25_str, p75_str = [s.strip() for s in str(intervalle).split("–")]
        except Exception:
            p25_str, p75_str = "?", "?"

        META = int(rest[-1]) if rest else 0

        drivers = ""
        parts = (conf_txt or "").split("•")
        if len(parts) >= 3:
            drv = parts[-1].strip()
            drivers = ", ".join([d.strip() for d in drv.split(",")][:3])

        lines.append(f"{name}")
        lines.append(
            f"Buts attendus : {pred:.2f} ({p25_str}–{p75_str}) | META {META} | {label} : {prob_pct}%"
        )
        lines.append(f"Drivers : {drivers}")
        lines.append("")
        
    lines.append("```")
    return "\n".join(lines)

recap_md = (
    f"*{mdv2_escape('📅 Prévisions du ' + str(today))}*\n"
    f"{mdv2_escape('Over:')} *{len(matchs_over)}*  •  "
    f"{mdv2_escape('Under:')} *{len(matchs_under)}*  •  "
    f"{mdv2_escape('Opps:')} *{len(matchs_opps)}*\n"
)

sec_over  = build_table("🔥", "TOP CONFIANCE OVER",   matchs_over,  is_under=False)
sec_under = build_table("❄️", "TOP CONFIANCE UNDER",  matchs_under, is_under=True)
sec_opps  = build_table("🎯", "OPPORTUNITÉS CACHÉES", matchs_opps,  is_under=False)

messages = [recap_md, sec_over, sec_under, sec_opps]

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

for chunk in messages:
    # Telegram MarkdownV2: 4096 chars max → on reste safe à ~3800
    if len(chunk) > 3800:
        chunk = chunk[:3800] + "\n_" + mdv2_escape("(troncature…)") + "_"
    send_telegram_message(BOT_TOKEN, CHAT_ID, chunk, parse_mode="MarkdownV2")

today_str = datetime.now().strftime("%Y-%m-%d")

def run(cmd, cwd=None):
    """Exécute une commande shell et lève en cas d’erreur (log propre)."""
    print("→", " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd)

def split_home_away(s):
    parts = str(s).split(" vs ")
    return (parts[0], parts[1]) if len(parts) == 2 else (s, "")

rows_csv = []
for i, m in enumerate(matchs_jour):
    home, away = split_home_away(m["match"])
    prob_o25 = float(probas_over25[i])
    p25 = float(pred_p25[i])
    p75 = float(pred_p75[i])
    pred_b = float(m.get("pred_total", 0.0))
    heur_pct = float(m.get("score_heur", 0.0)) / 1.5  # normalisé [0,1]

    # Décision par score global (même logique que plus haut)
    incert = p75 - p25
    score_global = 0.50*prob_o25 + 0.30*min(pred_b/4.0, 1.0) + 0.20*heur_pct
    score_global -= 0.10 * max(0.0, (incert - 1.5))
    score_global = max(0.0, min(1.0, float(score_global)))
    if score_global >= 0.68:
        categorie = "Ouvert"
    elif score_global <= 0.38:
        categorie = "Fermé"
    else:
        categorie = "Neutre"

    # Buckets pour analyse hebdo
    if prob_o25 >= 0.68:
        prob_band = "forte"
    elif prob_o25 >= 0.55:
        prob_band = "moyenne"
    else:
        prob_band = "faible"
    if incert < 1.2:
        ci_bucket = "sharp"
    elif incert <= 2.0:
        ci_bucket = "moyen"
    else:
        ci_bucket = "flou"

    rows_csv.append({
        "date": today_str,
        "home": home,
        "away": away,
        "match": m["match"],
        "prediction_buts": round(pred_b, 2),
        "p25": round(p25, 2),
        "p75": round(p75, 2),
        "incertitude": round(incert, 2),
        "prob_over25": round(prob_o25, 3),
        "prob_under25": round(1.0 - prob_o25, 3),
        "confiance": m.get("confiance", ""),
        "score_heuristique": round(float(m.get("score_heur", 0.0)), 2),
        "categorie": categorie,
        "score_global": round(float(score_global), 3),
        "prob_band": prob_band,
        "ci_bucket": ci_bucket,
        "competition": m.get("competition", "")
    })

df_today = pd.DataFrame(rows_csv)

TOKEN_HUB = os.getenv("TOKEN_HUB")
if not TOKEN_HUB:
    raise ValueError("❌ Le token GitHub (TOKEN_HUB) n'est pas défini.")

# --- Git config (comme dans le train) ---
run(["git", "config", "--global", "user.email", "lilian.pamphile.bts@gmail.com"])
run(["git", "config", "--global", "user.name", "LilianPamphile"])

REPO_DIR = "main_push"
REPO_URL = f"https://{TOKEN_HUB}@github.com/LilianPamphile/xgenius.git"

# Nettoyage / clone
if os.path.exists(REPO_DIR):
    shutil.rmtree(REPO_DIR)
run(["git", "clone", REPO_URL, REPO_DIR])

# (Optionnel mais utile en CI/root)
try:
    run(["git", "config", "--global", "--add", "safe.directory", os.path.abspath(REPO_DIR)])
except subprocess.CalledProcessError:
    pass  # pas bloquant

# Chemins
suivi_path = os.path.join(REPO_DIR, "suivi_predictions")
os.makedirs(suivi_path, exist_ok=True)
csv_path = os.path.join(suivi_path, "historique_predictions.csv")

# Fusion avec l’historique s’il existe
if os.path.exists(csv_path):
    try:
        df_hist = pd.read_csv(csv_path)
    except Exception:
        df_hist = pd.DataFrame(columns=df_today.columns)
    df_combined = pd.concat([df_hist, df_today], ignore_index=True)
else:
    df_combined = df_today

# Écriture
df_combined.to_csv(csv_path, index=False)
print(f"💾 Écrit: {csv_path} ({len(df_today)} lignes ajoutées)")

# Commit & push (avec erreurs visibles)
run(["git", "add", "suivi_predictions/historique_predictions.csv"], cwd=REPO_DIR)
# Si aucune diff, 'commit' renvoie code ≠0. On gère proprement.
try:
    run(["git", "commit", "-m", f"📊 Ajout des prédictions du {today_str}"], cwd=REPO_DIR)
except subprocess.CalledProcessError:
    print("ℹ️ Aucun changement à committer (fichier identique).")

run(["git", "push", "origin", "main"], cwd=REPO_DIR)
print("✅ Suivi des prédictions mis à jour dans historique_predictions.csv.")
