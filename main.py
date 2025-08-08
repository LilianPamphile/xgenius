# -*- coding: utf-8 -*-
import requests
import psycopg2
# Fonction de conversion sécurisée
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import pickle
import pandas as pd
import numpy as np
import shutil

# 🔑 Clé API SportsData.io
API_KEY = "b63f99b8e4mshb5383731d310a85p103ea1jsn47e34368f5df"

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
        v = float(x)
        return v
    except:
        return np.nan

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
def send_email_html(subject, html_body, to_email):
    from_email = "lilian.pamphile.bts@gmail.com"
    password = "fifkktsenfxsqiob"  # mot de passe d'application Gmail

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = ", ".join(to_email)

    part_html = MIMEText(html_body, "html")
    msg.attach(part_html)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(from_email, password)
            server.send_message(msg)
        print("📬 Email HTML envoyé avec succès.")
    except Exception as e:
        print("❌ Erreur lors de l'envoi de l'email HTML :", e)


def gen_table(matchs, segment):
    """
    matchs: liste de tuples (prob, name, pred, intervalle, conf, heur, s_final)
    segment: "Over" | "Under" | "Opps"
    """
    if not matchs:
        return "<p>Aucun match détecté.</p>"

    # tri par score final décroissant
    matchs_sorted = sorted(matchs, key=lambda x: x[-1], reverse=True)

    # libellé proba selon segment
    proba_col = "📊 Proba O≥2.5" if segment in ("Over", "Opps") else "🚫 Proba U≤2.0"

    rows = []
    for prob, name, pred, intervalle, conf, heur, _ in matchs_sorted:
        rows.append(f"""
        <tr>
            <td>{name}</td>
            <td>{pred:.2f}</td>
            <td>{intervalle}</td>
            <td>{int(prob*100)}%</td>
            <td>{heur:.2f}</td>
            <td>{conf}</td>
        </tr>""")

    return f"""
    <table>
        <thead>
            <tr>
                <th>Match</th>
                <th>⚽ Prédiction</th>
                <th>🔁 Intervalle</th>
                <th>{proba_col}</th>
                <th>🧠 Score</th>
                <th>Confiance</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>"""



# --- Téléchargement des fichiers modèle/scaler depuis GitHub ---
def telecharger_model_depuis_github():
    REPO = "LilianPamphile/paris-sportifs"
    BRANCH = "main"
    TOKEN = os.getenv("GITHUB_TOKEN")

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
    url_xg = "https://api-football-v1.p.rapidapi.com/v3/fixtures"  # même endpoint, xG inclus dans statistics parfois

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

    def get_matchs_jour_for_prediction():
        cursor = conn.cursor()

        # 1. Récupère les stats globales nécessaires
        query = """
        SELECT
          m.game_id,
          m.date::date AS date_match,
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
            bm = np.where(est_dom, m["buts_m_dom"].values, m["buts_m_ext"].values)
            be = np.where(est_dom, m["buts_m_ext"].values, m["buts_m_dom"].values)
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
                game_id, date_match, dom, ext,
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
            poss_moyenne = float((poss_dom + poss_ext) / 2)
            
            # --- Cartons total ---
            cartons_total = float(cj_dom + cr_dom + cj_ext + cr_ext)
            
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
                "features": feature_vector,
                "poss": poss_moyenne,
                "corners": float(corners_dom + corners_ext),
                "fautes": float(fautes_dom + fautes_ext),
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

    X_live = X_live.replace([np.inf, -np.inf], np.nan)
    for col in list(X_live.columns):
        miss = X_live[col].isna()
        if miss.any():
            X_live[col].fillna(X_live[col].median(), inplace=True)
    
    X_live_scaled = scaler_ml.transform(X_live)
    
    # prédiction principale
    preds_cat = model_cat.predict(X_live_scaled)
    preds_hgb = model_hgb.predict(X_live_scaled)
    # Prédiction des bornes conformal
    pred_p25 = model_p25.predict(X_live_scaled) - OFFSET
    pred_p75 = model_p75.predict(X_live_scaled) + OFFSET


    # Prédiction classification over 2.5
    probas_over25 = model_over25.predict_proba(X_live_scaled)[:, 1]  # probabilité que over 2.5

   # === Pondération dynamique selon MAE inverse ===
    with open("model_files/mae_models.pkl", "rb") as f:
        mae_dict = pickle.load(f)
    
    mae_cat = mae_dict["mae_cat"]
    mae_hgb = mae_dict["mae_hgb"]

    inv_total = 1 / mae_cat + 1 / mae_hgb
    weight_cat = (1 / mae_cat) / inv_total
    weight_hgb = (1 / mae_hgb) / inv_total
    pred_buts = weight_cat * preds_cat + weight_hgb * preds_hgb
    
    matchs_over, matchs_under, matchs_opps = [], [], []

    def coherence_count(pred_total, prob_over25, incertitude, score_heur):
        """Compte combien de signaux vont dans la même direction (Over vs Under)."""
        votes_over = 0
        votes_under = 0
    
        # ML (régression)
        votes_over += int(pred_total >= 2.5)
        votes_under += int(pred_total <= 2.0)
    
        # Classif Over25
        votes_over += int(prob_over25 >= 0.65)
        votes_under += int(prob_over25 <= 0.35)
    
        # Conformal (confiance)
        votes_over += int(incertitude < 1.5)   # bonus confiance pour les deux sens
        votes_under += int(incertitude < 1.5)
    
        # Heuristique
        votes_over += int(score_heur >= 0.60)
        votes_under += int(score_heur <= 0.40)
    
        return votes_over, votes_under
    
    def score_final(pred_total, prob_over25, score_heur, incertitude):
        """Score de tri interne pour l'affichage (plus haut = mieux)."""
        base = 0.4 * (pred_total / 4.0) + 0.3 * prob_over25 + 0.3 * (score_heur / 1.0)
        conf_bonus = max(0.0, 0.2 - 0.1 * max(0.0, incertitude - 1.0))  # petit bonus si intervalle serré
        return base + conf_bonus


    for i, match in enumerate(matchs_jour):
        pred_total = float(pred_buts[i])
        p25, p75 = float(pred_p25[i]), float(pred_p75[i])
        incertitude = p75 - p25
        prob_over25 = float(probas_over25[i])
        prob_under25 = 1.0 - prob_over25
    
        # ligne propre (non-scalée) pour features heuristiques déjà construite au-dessus
        row = X_live.iloc[i]
        enc_dom_val = float(row.get("buts_encaissés_dom", 0.0))
        enc_ext_val = float(row.get("buts_encaissés_ext", 0.0))
        solidite_dom = 1.0 / (enc_dom_val + 0.1)
        solidite_ext = 1.0 / (enc_ext_val + 0.1)
    
        corners_total = float(match.get("corners", (row.get("corners_dom", 0.0) + row.get("corners_ext", 0.0))))
        fautes_total  = float(match.get("fautes", 0.0))
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
            "cartons": cartons_total, "poss": poss_moy,
        }
    
        # Vérif que toutes les features attendues sont présentes
        missing_feats = [f for f in features_heur if f not in d]
        if missing_feats:
            print(f"[WARN] Features manquantes pour score_heur: {missing_feats}")
        
        # Construction dans le bon ordre
        X_input_heur = pd.DataFrame([[d.get(f, 0.0) for f in features_heur]], columns=features_heur)
        
        # Prédiction
        score_heur = float(model_heuristique.predict(X_input_heur)[0])
        
        # Contrôle des bornes
        if score_heur < -0.05 or score_heur > 2:
            print(f"[WARN] score_heur aberrant ({score_heur:.2f}) pour {match.get('match', 'inconnu')}")
        
        # Clamp pour rester dans la plage exploitable
        score_heur = max(0.0, min(score_heur, 1.5))

        match["score_heur"] = score_heur
    
        # commentaire confiance simple
        if incertitude > 2.5:
            commentaire = "⚠️ Incertitude élevée"
        elif incertitude < 1.5:
            commentaire = "✅ Confiance élevée"
        else:
            commentaire = "ℹ️ Confiance modérée"
        match["confiance"] = commentaire
    
        # --- COHÉRENCE ---
        votes_over, votes_under = coherence_count(pred_total, prob_over25, incertitude, score_heur)
    
        # --- RÈGLES OPTION 2 ---
        is_over = (pred_total >= 2.5) and (prob_over25 >= 0.65) and (incertitude < 1.5) and (score_heur >= 0.60) and (votes_over >= 3)
        is_under = (pred_total <= 2.0) and (prob_under25 >= 0.70) and (incertitude < 1.5) and (score_heur <= 0.40) and (votes_under >= 3)
    
        # opportunités cachées : zone 50–65% (ou proche seuils) + cohérence ≥3
        near_over_cut = (2.3 <= pred_total < 2.5) and (prob_over25 >= 0.55) and (votes_over >= 3)
        mid_prob = (0.50 <= prob_over25 <= 0.65) and (votes_over >= 3 or votes_under >= 3)
    
        # score de tri pour affichage
        s_final = score_final(pred_total, prob_over25, score_heur, incertitude)
    
        if is_over:
            matchs_over.append((prob_over25, match["match"], pred_total, f"{p25:.2f} – {p75:.2f}", commentaire, score_heur, s_final))
        elif is_under:
            matchs_under.append((prob_under25, match["match"], pred_total, f"{p25:.2f} – {p75:.2f}", commentaire, score_heur, s_final))
        elif near_over_cut or mid_prob:
            # stocke la proba Over pour l'affichage (plus parlant)
            matchs_opps.append((prob_over25, match["match"], pred_total, f"{p25:.2f} – {p75:.2f}", commentaire, score_heur, s_final))
                
        match["confiance"] = commentaire

    
    # === Génération du mail ===
    mail_lines = [f"📅 Prévisions du {today}\n"]

    # Over
    mail_lines.append("🔥 TOP CONFIANCE OVER (≥ 2.5, score🧠 ≥ 0.6, intervalle < 1.5)\n")
    for prob, name, pred, intervalle, conf, heur, s in sorted(matchs_over, key=lambda x: x[-1], reverse=True)[:6]:
        mail_lines.append(f"{name}\t⚽ {pred:.2f}\t🔁 {intervalle}\t📊 {int(prob*100)}%\t{conf}\t🧠 {heur:.2f}")
    if not matchs_over:
        mail_lines.append("Aucun match Over hautement fiable.\n")
    
    # Under
    mail_lines.append("\n❄️ TOP CONFIANCE UNDER (≤ 2.0, score🧠 ≤ 0.4, intervalle < 1.5)\n")
    for probU, name, pred, intervalle, conf, heur, s in sorted(matchs_under, key=lambda x: x[-1], reverse=True)[:6]:
        mail_lines.append(f"{name}\t⚽ {pred:.2f}\t🔁 {intervalle}\t🚫 {int(probU*100)}%\t{conf}\t🧠 {heur:.2f}")
    if not matchs_under:
        mail_lines.append("Aucun match Under hautement fiable.\n")
    
    # Opportunités cachées
    mail_lines.append("\n🎯 OPPORTUNITÉS CACHÉES (proba 50–65% mais signaux cohérents)\n")
    for probO, name, pred, intervalle, conf, heur, s in sorted(matchs_opps, key=lambda x: x[-1], reverse=True)[:8]:
        mail_lines.append(f"{name}\t⚽ {pred:.2f}\t🔁 {intervalle}\t📊 {int(probO*100)}%\t{conf}\t🧠 {heur:.2f}")
    if not matchs_opps:
        mail_lines.append("Aucune opportunité intermédiaire détectée.\n")
    
    mail_lines += [
        "\n🧠 Note méthodologique",
        "Sélection = ≥3 signaux alignés (ML, classif over/under, heuristique, intervalle).",
        "Intervalle = Conformal [p25 – p75] ; Confiance élevée si largeur < 1.5.",
        "Score🧠: ≥0.6 fort potentiel offensif ; ≤0.4 faible potentiel.",
    ]

    
    html_body = f"""
    <html>
    <head>
    <style>
        body {{ font-family: Arial, sans-serif; color: #333; }}
        h2 {{ color: #d9534f; margin-bottom: 6px; }}
        h3 {{ margin-top: 18px; }}
        .pill {{ display:inline-block; background:#eef; border-radius:12px; padding:2px 8px; font-size:12px; color:#445; margin-left:8px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 6px; }}
        th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
        th {{ background-color: #f7f7f7; }}
        .note {{ font-size: 12px; color: #555; margin-top: 16px; line-height: 1.4; }}
    </style>
    </head>
    <body>
    <h2>📅 Prévisions du {today}</h2>
    <div class="pill">Over: {len(matchs_over)}</div>
    <div class="pill">Under: {len(matchs_under)}</div>
    <div class="pill">Opps: {len(matchs_opps)}</div>
    
    <div class="match-section">
        <h3>🔥 TOP CONFIANCE OVER (≥ 2.5, score🧠 ≥ 0.6, intervalle &lt; 1.5)</h3>
        {gen_table(matchs_over, "Over")}
    </div>
    
    <div class="match-section">
        <h3>❄️ TOP CONFIANCE UNDER (≤ 2.0, score🧠 ≤ 0.4, intervalle &lt; 1.5)</h3>
        {gen_table(matchs_under, "Under")}
    </div>
    
    <div class="match-section">
        <h3>🎯 OPPORTUNITÉS CACHÉES (proba 50–65% mais signaux cohérents)</h3>
        {gen_table(matchs_opps, "Opps")}
    </div>
    
    <div class="note">
        <strong>🧠 Méthodo.</strong><br>
        Sélection = ≥3 signaux alignés (régression ML, classif Over/Under, heuristique, intervalle).<br>
        Intervalle = Conformal [p25–p75]; Confiance élevée si largeur &lt; 1.5.<br>
        Pondération ML: moyenne CatBoost/HGB pondérée à l’inverse des MAE.
    </div>
    </body>
    </html>
    """

    
    send_email_html(
        subject="📊 Analyse quotidienne Xgenius (HTML)",
        html_body=html_body,
        to_email=["lilian.pamphile.bts@gmail.com"]
    )


# Gestion erreur
except Exception as e:
    error_message = f"❌ Erreur durant l’exécution du script main du {today} :\n\n{str(e)}"
    send_email_html(
        subject="❌ Échec - Script Main",
        html_body=f"<pre>{error_message}</pre>",
        to_email=["lilian.pamphile.bts@gmail.com"]  # liste, pas string
    )

    

# === Sauvegarde dans un unique fichier historique CSV ===
import pandas as pd

today_str = datetime.now().strftime("%Y-%m-%d")
df_today = pd.DataFrame([
    {
        "date": today_str,
        "match": m["match"],
        "prediction_buts": round(pred_buts[i], 2),
        "p25": round(pred_p25[i], 2),
        "p75": round(pred_p75[i], 2),
        "confiance": m["confiance"],
        "score_heuristique": round(m.get("score_heur", 0), 2),
        "categorie": (
            "Ouvert" if pred_buts[i] >= 2.5 and probas_over25[i] >= 0.6 and (pred_p75[i] - pred_p25[i]) < 1.5
            else "Fermé" if pred_buts[i] <= 2.0 and (1 - probas_over25[i]) >= 0.7 and (pred_p75[i] - pred_p25[i]) < 1.5
            else "Neutre"
        )
    }
    for i, m in enumerate(matchs_jour)
])

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("❌ Le token GitHub (GITHUB_TOKEN) n'est pas défini.")

# === Clone du dépôt GitHub ===
REPO_DIR = "main_push"
REPO_URL = f"https://{GITHUB_TOKEN}@github.com/LilianPamphile/paris-sportifs.git"

if os.path.exists(REPO_DIR):
    shutil.rmtree(REPO_DIR)
os.system(f"git clone {REPO_URL} {REPO_DIR}")

# === Chargement de l'historique s’il existe ===
suivi_path = os.path.join(REPO_DIR, "suivi_predictions")
os.makedirs(suivi_path, exist_ok=True)
csv_path = os.path.join(suivi_path, "historique_predictions.csv")

if os.path.exists(csv_path):
    df_hist = pd.read_csv(csv_path)
    df_combined = pd.concat([df_hist, df_today], ignore_index=True)
else:
    df_combined = df_today

# === Écriture finale du fichier unique ===
df_combined.to_csv(csv_path, index=False)

# === Commit & Push sur GitHub ===
os.system(f"cd {REPO_DIR} && git add suivi_predictions/historique_predictions.csv")
os.system(f"cd {REPO_DIR} && git commit -m '📊 Ajout des prédictions du {today_str}'")
os.system(f"cd {REPO_DIR} && git push origin main")

print("✅ Suivi des prédictions mis à jour dans historique_predictions.csv.")
