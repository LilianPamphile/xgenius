# -*- coding: utf-8 -*-
import requests
import psycopg2
from datetime import datetime, timedelta
import os
import pandas as pd
import numpy as np
import shutil
import subprocess
import json
import joblib

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
    "World Cup": "1",
    "Euro Championship": "4",
    "Euro Championship - Qualification": "960",
    "UEFA Nations League": "5",
    "UEFA Super Cup": "531",
    "FIFA Club World Cup": "15",
    "FIFA Club World Cup - Play-In": "1186",
    "Friendlies": "10",
    "World Cup - Qualification Africa": "29",
    "World Cup - Qualification Europe": "32",
    "World Cup - Qualification South America": "34",
    "World Cup - Qualification Intercontinental Play-offs": "37",
}

# 🔌 Connexion PostgreSQL Railway
DATABASE_URL = "postgresql://postgres:jDDqfaqpspVDBBwsqxuaiSDNXjTxjMmP@shortline.proxy.rlwy.net:36536/railway"
conn = psycopg2.connect(DATABASE_URL)
cursor = conn.cursor()

print("Fin de la défintion de variables")

# --- Défauts safe si une exception survient plus tard ---
matchs_jour = []
matchs_low, matchs_mid, matchs_high = [], [], []

################################### Fonctions utiles ###################################

def to_float(x):
    try:
        v = float(x)
        return v
    except:
        return 0.0

MD_SPECIAL = r"_*[]()~`>#+-=|{}.!"

def mdv2_escape(s: str) -> str:
    s = str(s)
    out = []
    for ch in s:
        out.append("\\" + ch if ch in MD_SPECIAL else ch)
    return "".join(out)

def num(x, default=0.0):
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

def get_saison_api(competition_name: str) -> int:
    cname = competition_name.lower()
    if "world cup - qualification africa" in cname:
        return 2023
    elif "world cup - qualification europe" in cname:
        return 2024
    elif "world cup - qualification south america" in cname:
        return 2026
    elif cname.strip() == "world cup":
        return 2026
    else:
        return 2025  # tout le reste

# === 📌 1️⃣ Récupération des Matchs ===
def recuperer_matchs(date, API_KEY):
    url_base = "https://api-football-v1.p.rapidapi.com/v3/fixtures"
    headers = {
        "x-rapidapi-key": API_KEY,
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com"
    }

    total_matchs = 0
    print(f"📅 Récupération des matchs pour le {date}")

    for competition_name, competition_id in COMPETITIONS.items():
        saison_api_for_this = get_saison_api(competition_name)
        params = {
            "league": competition_id,
            "season": saison_api_for_this,
            "date": str(date),
            "timezone": "Europe/Paris"
        }

        r = requests.get(url_base, headers=headers, params=params)
        fixtures = r.json().get("response", [])
        matchs_inseres_competition = 0

        for match in fixtures:
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

        print(f"   ▶ {competition_name} | saison {saison_api_for_this} | {len(fixtures)} reçus | {matchs_inseres_competition} insérés")

    conn.commit()
    print(f"📊 Total : {total_matchs} matchs insérés pour {date}")

# === 📌 2️⃣ Récupération des Stats ===
def recuperer_stats_matchs(date, API_KEY):
    url_fixtures = "https://api-football-v1.p.rapidapi.com/v3/fixtures"
    url_stats = "https://api-football-v1.p.rapidapi.com/v3/fixtures/statistics"
    headers = {
        "x-rapidapi-key": API_KEY,
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com"
    }

    print(f"📅 Stats pour {date}")

    for competition_name, competition_id in COMPETITIONS.items():
        saison_api_for_this = get_saison_api(competition_name)
        params = {
            "league": competition_id,
            "season": saison_api_for_this,
            "date": str(date),
            "timezone": "Europe/Paris"
        }

        rf = requests.get(url_fixtures, headers=headers, params=params)
        fixtures = rf.json().get("response", [])
        if not fixtures:
            continue

        for match in fixtures:
            fixture_id = match["fixture"]["id"]
            equipe_dom = match["teams"]["home"]["name"]
            equipe_ext = match["teams"]["away"]["name"]

            rs = requests.get(url_stats, headers=headers, params={"fixture": fixture_id})
            stats_data = rs.json().get("response", [])
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

            cols = [
                "game_id",
                "possession_dom","possession_ext","tirs_dom","tirs_ext",
                "tirs_cadres_dom","tirs_cadres_ext","tirs_hors_cadre_dom","tirs_hors_cadre_ext",
                "tirs_bloques_dom","tirs_bloques_ext","tirs_dans_boite_dom","tirs_dans_boite_ext",
                "tirs_hors_boite_dom","tirs_hors_boite_ext","arrets_dom","arrets_ext",
                "buts_dom","buts_ext","passes_dom","passes_ext","passes_reussies_dom","passes_reussies_ext",
                "passes_pourcent_dom","passes_pourcent_ext","corners_dom","corners_ext",
                "fautes_dom","fautes_ext","hors_jeu_dom","hors_jeu_ext",
                "cartons_jaunes_dom","cartons_jaunes_ext","cartons_rouges_dom","cartons_rouges_ext",
                "xg_dom","xg_ext"
            ]
            placeholders = ", ".join(["%s"] * len(cols))

            if len(values) != len(cols):
                raise ValueError(f"Mismatch INSERT stats_matchs_v2: {len(values)} values pour {len(cols)} colonnes")

            sql_insert = f"""
                INSERT INTO stats_matchs_v2 ({", ".join(cols)})
                VALUES ({placeholders})
                ON CONFLICT (game_id) DO NOTHING
            """
            cursor.execute(sql_insert, values)

    conn.commit()
    print(f"✅ Stats enrichies insérées avec succès pour {date}")

### Mettre à jour table stats_globales
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

# === Fonctions pour le modèle vEnsemble_ADPoisson ===

def league_avg_goals(df_hist: pd.DataFrame, competition: str, date_ref, window: int = 60) -> float:
    q = (df_hist["competition"] == competition) & (df_hist["date_match"] < date_ref)
    m = df_hist.loc[q].sort_values("date_match", ascending=False).head(window)
    if len(m) == 0:
        return 2.5
    return float(m["total_buts"].mean())

def baseline_poisson_row(row: pd.Series) -> float:
    bm_home = float(row.get("forme_home_buts_marques", 0.0))
    be_home = float(row.get("forme_home_buts_encaisses", 0.0))
    bm_away = float(row.get("forme_away_buts_marques", 0.0))
    be_away = float(row.get("forme_away_buts_encaisses", 0.0))
    lam_home = 0.5 * (bm_home + be_away)
    lam_away = 0.5 * (bm_away + be_home)
    lam_total = max(0.0, lam_home + lam_away)
    return lam_total

try:
    # 1) Mise à jour des données
    recuperer_matchs(today, API_KEY)
    recuperer_stats_matchs(yesterday, API_KEY)
    mettre_a_jour_stats_globales(yesterday)
    conn.commit()
    print("✅ Récupération des données terminée !")

    # 2) Chargement des modèles et features (nouveau train)
    model_cat = joblib.load("models/model_cat_total_goals.pkl")
    model_hgb = joblib.load("models/model_hgb_total_goals.pkl")

    with open("models/FEATURES_TOTAL_BUTS.json", "r", encoding="utf-8") as f:
        FEATURES = json.load(f)

    with open("models/ensemble_weights_and_metrics.json", "r", encoding="utf-8") as f:
        conf = json.load(f)
    w_ml = float(conf["weights"]["w_ml"])
    w_poisson = float(conf["weights"]["w_poisson"])
    w_xg_exp = float(conf["weights"]["w_xg_exp"])

    def get_matchs_jour_for_prediction():
        cursor = conn.cursor()

        # 1. Stats agrégées pour les équipes du jour
        query = """
        SELECT
          m.game_id,
          m.date::date AS date_match,
          m.competition,
          m.equipe_domicile, m.equipe_exterieur,

          /* DOMICILE agrégé (2 saisons) */
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

          /* EXTERIEUR agrégé (2 saisons) */
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
                # sg1
                saison1, saison2, saison1, saison2,
                # sg2
                saison1, saison2, saison1, saison2,
                # date
                today,
            ),
        )
        rows = cursor.fetchall()

        # 2. Historique match par match (pour forme & ligue)
        cursor.execute("""
            SELECT
              m.date::date AS date_match,
              m.competition,
              m.equipe_domicile,
              m.equipe_exterieur,
              s.buts_dom AS buts_m_dom,
              s.buts_ext AS buts_m_ext,
              s.buts_dom + s.buts_ext AS total_buts
            FROM matchs_v2 m
            JOIN stats_matchs_v2 s ON m.game_id = s.game_id
            WHERE s.buts_dom IS NOT NULL AND s.buts_ext IS NOT NULL
        """)
        df_hist = pd.DataFrame(
            cursor.fetchall(),
            columns=[
                "date_match", "competition",
                "equipe_domicile", "equipe_exterieur",
                "buts_m_dom", "buts_m_ext", "total_buts"
            ],
        )

        def calculer_forme(equipe, date_ref, role: str, n=5):
            q = (df_hist["date_match"] < date_ref)
            if role == "home":
                q &= (df_hist["equipe_domicile"] == equipe)
            else:
                q &= (df_hist["equipe_exterieur"] == equipe)

            m = df_hist.loc[q].sort_values("date_match", ascending=False).head(n)
            if m.empty:
                return 0.0, 0.0

            if role == "home":
                bm = m["buts_m_dom"].values
                be = m["buts_m_ext"].values
            else:
                bm = m["buts_m_ext"].values
                be = m["buts_m_dom"].values

            return float(np.mean(bm)), float(np.mean(be))

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

            # forme attaque / défense
            bm_home, be_home = calculer_forme(dom, date_match, role="home")
            bm_away, be_away = calculer_forme(ext, date_match, role="away")

            attaque_home = bm_home
            defense_home = be_home
            attaque_away = bm_away
            defense_away = be_away

            xg_exp_total = attaque_home * defense_away + attaque_away * defense_home
            league_avg = league_avg_goals(df_hist, competition, date_match, window=60)

            features_dict = {
                "forme_home_buts_marques": to_float(bm_home),
                "forme_home_buts_encaisses": to_float(be_home),
                "forme_away_buts_marques": to_float(bm_away),
                "forme_away_buts_encaisses": to_float(be_away),
                "attaque_home": to_float(attaque_home),
                "defense_home": to_float(defense_home),
                "attaque_away": to_float(attaque_away),
                "defense_away": to_float(defense_away),
                "xg_dom": to_float(xg_dom),
                "xg_ext": to_float(xg_ext),
                "tirs_dom": to_float(tirs_dom),
                "tirs_ext": to_float(tirs_ext),
                "league_avg_goals_60d": to_float(league_avg),
                "xg_exp_total": to_float(xg_exp_total),
            }

            feature_vector = [features_dict.get(f, 0.0) for f in FEATURES]

            matchs.append({
                "match": f"{dom} vs {ext}",
                "competition": competition,
                "features": feature_vector,
                "xg_exp_total": xg_exp_total,
            })

        cursor.close()
        return matchs

    # 3) Récupération des matchs du jour + features
    matchs_jour = get_matchs_jour_for_prediction()

    if not matchs_jour:
        print("ℹ️ Aucun match à prédire aujourd'hui.")
    else:
        # X_live : même ordre de colonnes que le train
        X_live = pd.DataFrame([m["features"] for m in matchs_jour], columns=FEATURES)

        # prédictions ML
        preds_cat = model_cat.predict(X_live.values)
        preds_hgb = model_hgb.predict(X_live.values)
        pred_ml = 0.5 * (preds_cat + preds_hgb)

        # baseline Poisson + xg_exp_total
        baseline_vals = []
        xg_exp_vals = []
        for i in range(len(matchs_jour)):
            row = X_live.iloc[i]
            baseline_vals.append(baseline_poisson_row(row))
            xg_exp_vals.append(float(row.get("xg_exp_total", 0.0)))

        baseline_vals = np.array(baseline_vals)
        xg_exp_vals = np.array(xg_exp_vals)

        # ensemble final
        pred_final = w_ml * pred_ml + w_poisson * baseline_vals + w_xg_exp * xg_exp_vals

        for i, m in enumerate(matchs_jour):
            m["pred_total"] = float(pred_final[i])

        # 4) Buckets simples : Faible / Moyen / Fort
        matchs_low, matchs_mid, matchs_high = [], [], []

        for m in matchs_jour:
            p = m["pred_total"]
            name = m["match"]
            comp = m.get("competition", "")
            if p <= 2.0:
                matchs_low.append((p, name, comp))
            elif p >= 3.0:
                matchs_high.append((p, name, comp))
            else:
                matchs_mid.append((p, name, comp))

except Exception as e:
    print("❌ Erreur pendant la génération des prédictions :", e)

# === Affichage Telegram simplifié ===

def build_table(title_emoji: str, title_text: str, rows) -> str:
    header = f"*{mdv2_escape(title_emoji + ' ' + title_text)}*"
    if not rows:
        return f"{header}\n_{mdv2_escape('Aucun match détecté.')}_\n"

    ordered = sorted(rows, key=lambda x: x[0], reverse=True)
    lines = [header]

    for pred_buts, name, comp in ordered:
        line1 = mdv2_escape(f"{comp} — {name}")
        line2 = mdv2_escape(f"Buts attendus : {pred_buts:.2f}")
        lines.append(line1)
        lines.append(line2)
        lines.append("")

    return "\n".join(lines)

recap_md = (
    f"*{mdv2_escape('📅 Prévisions du ' + str(today))}*\n"
    f"{mdv2_escape('Fort buts:')} *{len(matchs_high)}*  •  "
    f"{mdv2_escape('Moyen:')} *{len(matchs_mid)}*  •  "
    f"{mdv2_escape('Faible:')} *{len(matchs_low)}*\n"
)

sec_high = build_table("🔥", "MATCHS À FORT POTENTIEL DE BUTS", matchs_high)
sec_mid  = build_table("⚖️", "MATCHS ÉQUILIBRÉS", matchs_mid)
sec_low  = build_table("❄️", "MATCHS À FAIBLE POTENTIEL", matchs_low)

messages = [recap_md, sec_high, sec_mid, sec_low]

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

for chunk in messages:
    if len(chunk) > 3800:
        chunk = chunk[:3800] + "\n_" + mdv2_escape("(troncature…)") + "_"
    if BOT_TOKEN and CHAT_ID:
        send_telegram_message(BOT_TOKEN, CHAT_ID, chunk, parse_mode="MarkdownV2")

# === Sauvegarde CSV historique_predictions (simplifié) ===

today_str = datetime.now().strftime("%Y-%m-%d")

def run(cmd, cwd=None):
    print("→", " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd)

def split_home_away(s):
    parts = str(s).split(" vs ")
    return (parts[0], parts[1]) if len(parts) == 2 else (s, "")

rows_csv = []
for m in matchs_jour:
    home, away = split_home_away(m["match"])
    pred_b = float(m.get("pred_total", 0.0))
    competition = m.get("competition", "")

    if pred_b <= 2.0:
        categorie = "Faible"
    elif pred_b >= 3.0:
        categorie = "Fort"
    else:
        categorie = "Moyen"

    rows_csv.append({
        "date": today_str,
        "home": home,
        "away": away,
        "match": m["match"],
        "competition": competition,
        "prediction_buts": round(pred_b, 2),
        "categorie": categorie,
    })

df_today = pd.DataFrame(rows_csv)

TOKEN_HUB = os.getenv("TOKEN_HUB")
if not TOKEN_HUB:
    raise ValueError("❌ Le token GitHub (TOKEN_HUB) n'est pas défini.")

run(["git", "config", "--global", "user.email", "lilian.pamphile.bts@gmail.com"])
run(["git", "config", "--global", "user.name", "LilianPamphile"])

REPO_DIR = "main_push"
REPO_URL = f"https://{TOKEN_HUB}@github.com/LilianPamphile/xgenius.git"

if os.path.exists(REPO_DIR):
    shutil.rmtree(REPO_DIR)
run(["git", "clone", REPO_URL, REPO_DIR])

try:
    run(["git", "config", "--global", "--add", "safe.directory", os.path.abspath(REPO_DIR)])
except subprocess.CalledProcessError:
    pass

suivi_path = os.path.join(REPO_DIR, "suivi_predictions")
os.makedirs(suivi_path, exist_ok=True)
csv_path = os.path.join(suivi_path, "historique_predictions.csv")

if os.path.exists(csv_path):
    try:
        df_hist = pd.read_csv(csv_path)
    except Exception:
        df_hist = pd.DataFrame(columns=df_today.columns)
    df_combined = pd.concat([df_hist, df_today], ignore_index=True)
else:
    df_combined = df_today

df_combined.to_csv(csv_path, index=False)
print(f"💾 Écrit: {csv_path} ({len(df_today)} lignes ajoutées)")

run(["git", "add", "suivi_predictions/historique_predictions.csv"], cwd=REPO_DIR)
try:
    run(["git", "commit", "-m", f"📊 Ajout des prédictions du {today_str}"], cwd=REPO_DIR)
except subprocess.CalledProcessError:
    print("ℹ️ Aucun changement à committer (fichier identique).")

run(["git", "push", "origin", "main"], cwd=REPO_DIR)
print("✅ Suivi des prédictions mis à jour dans historique_predictions.csv.")
