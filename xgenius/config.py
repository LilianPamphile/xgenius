# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from zoneinfo import ZoneInfo
import os

# ============================================================
# XGENIUS CONFIG
# ============================================================
# Tu as demandé à garder l'URL BDD directement dans le code.
# Si ton repo est public, cette URL sera visible publiquement.
DATABASE_URL = "postgresql://postgres:jDDqfaqpspVDBBwsqxuaiSDNXjTxjMmP@shortline.proxy.rlwy.net:36536/railway"

API_BASE_URL = "https://api-football-v1.p.rapidapi.com/v3"
API_HOST = "api-football-v1.p.rapidapi.com"
API_KEY = os.getenv("RAPIDAPI_KEY", "")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

LOCAL_TZ = ZoneInfo("Europe/Paris")
UTC_TZ = ZoneInfo("UTC")

# GitHub Actions lance 06:17 et 07:17 UTC pour gérer heure été/hiver.
# Le code ne continue que si l'heure Paris est 08:17 le lundi/jeudi.
SCHEDULED_LOCAL_HOUR = 8
SCHEDULED_LOCAL_MINUTE = 17

# Dates suivies : lundi -> jeudi puis jeudi -> lundi.
# End exclusif côté code, affichage inclusif côté Telegram.
MONDAY_RUN_WEEKDAY = 0
THURSDAY_RUN_WEEKDAY = 3

# Seuils de volume
MIN_TRAIN_MATCHES = int(os.getenv("MIN_TRAIN_MATCHES", "120"))
MIN_VALIDATION_MATCHES = int(os.getenv("MIN_VALIDATION_MATCHES", "40"))
MAX_STATS_IMPORT_PER_RUN = int(os.getenv("MAX_STATS_IMPORT_PER_RUN", "80"))
MAX_RADAR_TOP_MATCHES = int(os.getenv("MAX_RADAR_TOP_MATCHES", "8"))
MAX_FULL_LIST_MATCHES = int(os.getenv("MAX_FULL_LIST_MATCHES", "160"))
SHOW_FULL_LIST = os.getenv("SHOW_FULL_LIST", "true").lower() in {"1", "true", "yes", "y"}

# Bootstrapping manuel : import des derniers N jours d'historique.
DEFAULT_BOOTSTRAP_DAYS = int(os.getenv("BOOTSTRAP_DAYS", "90"))

# Modèle
MODEL_FAMILY = "extratrees_poisson_v1"
MODEL_ACTIVATION_TOLERANCE = float(os.getenv("MODEL_ACTIVATION_TOLERANCE", "0.03"))
RANDOM_STATE = 42

# Compétitions suivies. On récupère les fixtures par date puis on filtre localement.
# Tu peux ajouter/retirer des ligues ici sans toucher au reste du code.
MONITORED_LEAGUES = {
    61: "Ligue 1",
    39: "Premier League",
    78: "Bundesliga",
    135: "Serie A",
    88: "Eredivisie",
    140: "La Liga",
    2: "UEFA Champions League",
    3: "UEFA Europa League",
    848: "UEFA Europa Conference League",
    79: "2. Bundesliga",
    307: "Saudi Professional League",
    1: "World Cup",
    4: "Euro Championship",
    960: "Euro Championship - Qualification",
    5: "UEFA Nations League",
    531: "UEFA Super Cup",
    15: "FIFA Club World Cup",
    1186: "FIFA Club World Cup - Play-In",
    10: "Friendlies",
    29: "World Cup - Qualification Africa",
    32: "World Cup - Qualification Europe",
    34: "World Cup - Qualification South America",
    37: "World Cup - Qualification Intercontinental Play-offs",
}

FINISHED_STATUS = {"FT", "AET", "PEN"}
LIVE_OR_NOT_STARTED_STATUS = {"NS", "TBD", "1H", "HT", "2H", "ET", "BT", "P", "SUSP", "INT", "PST"}

@dataclass(frozen=True)
class RunWindow:
    mode: str
    evaluation_label: str
    evaluation_start_local: object
    evaluation_end_local: object
    prediction_label: str
    prediction_start_local: object
    prediction_end_local: object
