# -*- coding: utf-8 -*-
"""XGenius Match Radar - version minimale.

Deux exécutions par semaine via GitHub Actions avec diffusion Telegram :
- lundi : bilan du week-end + radar lundi-mercredi ;
- jeudi : bilan lundi-mercredi + radar jeudi-dimanche.

Les prévisions 1X2 proviennent de l'endpoint /predictions d'API-Football.
Les probabilités Over 2.5 et BTTS sont calculées avec une approximation
Poisson simple à partir de la forme des cinq derniers matchs renvoyée par
le même endpoint.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import psycopg2
import requests
from psycopg2.extras import RealDictCursor
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PARIS_TZ = ZoneInfo("Europe/Paris")
API_BASE_URL = "https://api-football-v1.p.rapidapi.com/v3"
API_HOST = "api-football-v1.p.rapidapi.com"
TELEGRAM_API_BASE = "https://api.telegram.org"

# Compétitions suivies. Ajouter/supprimer simplement un identifiant ici.
# Les saisons ne sont pas codées : les matchs sont récupérés par date.
COMPETITIONS = {
    1: "FIFA World Cup",
    2: "UEFA Champions League",
    3: "UEFA Europa League",
    39: "Premier League",
    61: "Ligue 1",
    78: "Bundesliga",
    88: "Eredivisie",
    135: "Serie A",
    140: "La Liga",
    307: "Saudi Pro League",
    848: "UEFA Conference League",
}

FUTURE_STATUSES = {"NS", "TBD"}

DATABASE_URL = "postgresql://postgres:jDDqfaqpspVDBBwsqxuaiSDNXjTxjMmP@shortline.proxy.rlwy.net:36536/railway"
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "").strip()
DRY_RUN = os.getenv("DRY_RUN", "true").strip().lower() in {"1", "true", "yes", "oui"}
MAX_RADAR_MATCHES = int(os.getenv("MAX_RADAR_MATCHES", "5"))
MAX_PREDICTIONS_PER_RUN = int(os.getenv("MAX_PREDICTIONS_PER_RUN", "70"))
SHOW_ALL_MATCHES = os.getenv("SHOW_ALL_MATCHES", "true").strip().lower() in {"1", "true", "yes", "oui"}
MAX_FULL_MATCHES = int(os.getenv("MAX_FULL_MATCHES", "120"))


# ---------------------------------------------------------------------------
# Modèles de données légers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Periods:
    label: str
    future_start: date
    future_end_exclusive: date
    bilan_label: str
    bilan_start: date
    bilan_end_exclusive: date

    @property
    def future_end_inclusive(self) -> date:
        return self.future_end_exclusive - timedelta(days=1)

    @property
    def bilan_end_inclusive(self) -> date:
        return self.bilan_end_exclusive - timedelta(days=1)


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def require_settings() -> None:
    missing = []
    if not RAPIDAPI_KEY:
        missing.append("RAPIDAPI_KEY")
    if missing:
        raise RuntimeError("Variables manquantes : " + ", ".join(missing))


def daterange(start: date, end_exclusive: date) -> Iterable[date]:
    """Dates entre start inclus et end_exclusive exclu."""
    current = start
    while current < end_exclusive:
        yield current
        current += timedelta(days=1)


def parse_api_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def pct(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).replace("%", "").strip())
    except (TypeError, ValueError):
        return None


def number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def nested(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def average_available(values: Iterable[float | None], default: float) -> float:
    cleaned = [float(v) for v in values if v is not None]
    return sum(cleaned) / len(cleaned) if cleaned else default


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def poisson_signals(prediction_payload: dict[str, Any]) -> tuple[float, float, float, float]:
    """Retourne lambda domicile, lambda extérieur, P(Over 2.5), P(BTTS)."""
    home_for = number(nested(prediction_payload, "teams", "home", "last_5", "goals", "for", "average"))
    home_against = number(nested(prediction_payload, "teams", "home", "last_5", "goals", "against", "average"))
    away_for = number(nested(prediction_payload, "teams", "away", "last_5", "goals", "for", "average"))
    away_against = number(nested(prediction_payload, "teams", "away", "last_5", "goals", "against", "average"))

    lambda_home = clamp(average_available([home_for, away_against], 1.35), 0.15, 3.50)
    lambda_away = clamp(average_available([away_for, home_against], 1.10), 0.15, 3.50)
    lambda_total = lambda_home + lambda_away

    # P(total >= 3) pour une loi de Poisson de paramètre lambda_total.
    p_over_25 = 1.0 - math.exp(-lambda_total) * (
        1.0 + lambda_total + (lambda_total**2) / 2.0
    )

    # P(domicile >= 1 et extérieur >= 1), indépendance Poisson simplifiée.
    p_btts = (1.0 - math.exp(-lambda_home)) * (1.0 - math.exp(-lambda_away))

    return lambda_home, lambda_away, 100.0 * p_over_25, 100.0 * p_btts


def predicted_side(home: float, draw: float, away: float) -> str:
    return max({"HOME": home, "DRAW": draw, "AWAY": away}, key={"HOME": home, "DRAW": draw, "AWAY": away}.get)


def side_label(side: str, home_team: str, away_team: str) -> str:
    return {
        "HOME": home_team,
        "DRAW": "Match nul",
        "AWAY": away_team,
    }.get(side, side)


def profile_label(confidence: float, over_25: float, btts: float, home: float, draw: float, away: float) -> str:
    spread = max(home, draw, away) - min(home, draw, away)
    if over_25 >= 65 and btts >= 60:
        return "ouvert"
    if over_25 <= 40:
        return "fermé"
    if confidence >= 65:
        return "déséquilibré"
    if spread <= 12:
        return "très indécis"
    return "équilibré"



# ---------------------------------------------------------------------------
# API-Football
# ---------------------------------------------------------------------------

class FootballAPI:
    def __init__(self, api_key: str):
        self.session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.headers = {
            "x-rapidapi-key": api_key,
            "x-rapidapi-host": API_HOST,
        }

    def get(self, endpoint: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        response = self.session.get(
            f"{API_BASE_URL}/{endpoint.lstrip('/')}",
            headers=self.headers,
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()

        errors = payload.get("errors")
        if errors:
            raise RuntimeError(f"Erreur API-Football {endpoint}: {errors}")

        remaining = response.headers.get("x-ratelimit-requests-remaining")
        if remaining is not None:
            print(f"API-Football — requêtes restantes : {remaining}")

        return payload.get("response", [])

    def fixtures_for_date(self, day: date) -> list[dict[str, Any]]:
        fixtures = self.get(
            "fixtures",
            {"date": day.isoformat(), "timezone": "Europe/Paris"},
        )
        return [
            fixture
            for fixture in fixtures
            if int(fixture.get("league", {}).get("id", -1)) in COMPETITIONS
        ]

    def prediction(self, fixture_id: int) -> dict[str, Any] | None:
        response = self.get("predictions", {"fixture": fixture_id})
        return response[0] if response else None


# ---------------------------------------------------------------------------
# PostgreSQL
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS radar_matches (
    fixture_id BIGINT PRIMARY KEY,
    league_id INTEGER NOT NULL,
    competition TEXT NOT NULL,
    season INTEGER,
    kickoff TIMESTAMPTZ NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    status TEXT,
    home_goals INTEGER,
    away_goals INTEGER,

    predicted_result TEXT,
    home_pct DOUBLE PRECISION,
    draw_pct DOUBLE PRECISION,
    away_pct DOUBLE PRECISION,
    api_under_over TEXT,
    expected_home_goals DOUBLE PRECISION,
    expected_away_goals DOUBLE PRECISION,
    over_25_pct DOUBLE PRECISION,
    btts_pct DOUBLE PRECISION,
    confidence DOUBLE PRECISION,
    profile TEXT,
    advice TEXT,
    prediction_generated_at TIMESTAMPTZ,

    reported_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_radar_matches_kickoff
    ON radar_matches (kickoff);

CREATE INDEX IF NOT EXISTS idx_radar_matches_unreported
    ON radar_matches (reported_at)
    WHERE reported_at IS NULL;

CREATE TABLE IF NOT EXISTS radar_reports (
    report_key TEXT PRIMARY KEY,
    message_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE radar_reports
    ADD COLUMN IF NOT EXISTS message_id TEXT;
"""


class Database:
    def __init__(self, url: str):
        self.conn = psycopg2.connect(url)

    def close(self) -> None:
        self.conn.close()

    def ensure_schema(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
        self.conn.commit()

    def upsert_fixture(self, fixture: dict[str, Any]) -> None:
        fixture_info = fixture["fixture"]
        league = fixture["league"]
        teams = fixture["teams"]
        goals = fixture.get("goals", {})

        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO radar_matches (
                    fixture_id, league_id, competition, season, kickoff,
                    home_team, away_team, status, home_goals, away_goals, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (fixture_id) DO UPDATE SET
                    league_id = EXCLUDED.league_id,
                    competition = EXCLUDED.competition,
                    season = EXCLUDED.season,
                    kickoff = EXCLUDED.kickoff,
                    home_team = EXCLUDED.home_team,
                    away_team = EXCLUDED.away_team,
                    status = EXCLUDED.status,
                    home_goals = EXCLUDED.home_goals,
                    away_goals = EXCLUDED.away_goals,
                    updated_at = NOW()
                """,
                (
                    int(fixture_info["id"]),
                    int(league["id"]),
                    league.get("name") or COMPETITIONS.get(int(league["id"]), "Compétition"),
                    league.get("season"),
                    parse_api_datetime(fixture_info["date"]),
                    teams["home"]["name"],
                    teams["away"]["name"],
                    fixture_info.get("status", {}).get("short"),
                    goals.get("home"),
                    goals.get("away"),
                ),
            )

    def upsert_prediction(self, fixture_id: int, payload: dict[str, Any]) -> bool:
        prediction = payload.get("predictions", {})
        percents = prediction.get("percent", {})

        home_pct = pct(percents.get("home"))
        draw_pct = pct(percents.get("draw"))
        away_pct = pct(percents.get("away"))
        if home_pct is None or draw_pct is None or away_pct is None:
            return False

        expected_home, expected_away, over_25, btts = poisson_signals(payload)
        result = predicted_side(home_pct, draw_pct, away_pct)
        confidence = max(home_pct, draw_pct, away_pct)
        profile = profile_label(confidence, over_25, btts, home_pct, draw_pct, away_pct)

        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE radar_matches
                SET predicted_result = %s,
                    home_pct = %s,
                    draw_pct = %s,
                    away_pct = %s,
                    api_under_over = %s,
                    expected_home_goals = %s,
                    expected_away_goals = %s,
                    over_25_pct = %s,
                    btts_pct = %s,
                    confidence = %s,
                    profile = %s,
                    advice = %s,
                    prediction_generated_at = NOW(),
                    updated_at = NOW()
                WHERE fixture_id = %s
                """,
                (
                    result,
                    home_pct,
                    draw_pct,
                    away_pct,
                    prediction.get("under_over"),
                    expected_home,
                    expected_away,
                    over_25,
                    btts,
                    confidence,
                    profile,
                    prediction.get("advice"),
                    fixture_id,
                ),
            )
        return True

    def commit(self) -> None:
        self.conn.commit()

    def rollback(self) -> None:
        self.conn.rollback()

    def report_exists(self, report_key: str) -> bool:
        with self.conn.cursor() as cur:
            cur.execute("SELECT 1 FROM radar_reports WHERE report_key = %s", (report_key,))
            return cur.fetchone() is not None

    def save_report(self, report_key: str, message_id: str | None) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO radar_reports (report_key, message_id)
                VALUES (%s, %s)
                ON CONFLICT (report_key) DO NOTHING
                """,
                (report_key, message_id),
            )
        self.conn.commit()

    def complete_bilan(
        self,
        report_key: str,
        message_id: str | None,
        fixture_ids: list[int],
    ) -> None:
        """Enregistre le bilan et marque les matchs dans une seule transaction."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO radar_reports (report_key, message_id)
                VALUES (%s, %s)
                ON CONFLICT (report_key) DO NOTHING
                """,
                (report_key, message_id),
            )
            cur.execute(
                "UPDATE radar_matches SET reported_at = NOW() WHERE fixture_id = ANY(%s)",
                (fixture_ids,),
            )
        self.conn.commit()

    def unreported_completed(self, start: date, end_exclusive: date) -> list[dict[str, Any]]:
        start_dt = datetime.combine(start, datetime.min.time(), tzinfo=PARIS_TZ)
        end_dt = datetime.combine(end_exclusive, datetime.min.time(), tzinfo=PARIS_TZ)
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT *
                FROM radar_matches
                WHERE reported_at IS NULL
                  AND predicted_result IS NOT NULL
                  AND status IN ('FT', 'AET', 'PEN')
                  AND home_goals IS NOT NULL
                  AND away_goals IS NOT NULL
                  AND kickoff >= %s
                  AND kickoff < %s
                ORDER BY kickoff
                """,
                (start_dt, end_dt),
            )
            return list(cur.fetchall())

    def future_predictions(self, start: date, end_exclusive: date) -> list[dict[str, Any]]:
        start_dt = datetime.combine(start, datetime.min.time(), tzinfo=PARIS_TZ)
        end_dt = datetime.combine(end_exclusive, datetime.min.time(), tzinfo=PARIS_TZ)
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT *
                FROM radar_matches
                WHERE kickoff >= %s
                  AND kickoff < %s
                  AND kickoff > NOW()
                  AND status IN ('NS', 'TBD')
                  AND predicted_result IS NOT NULL
                ORDER BY kickoff
                """,
                (start_dt, end_dt),
            )
            return list(cur.fetchall())


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

class TelegramClient:
    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    def _require_settings(self) -> None:
        if not self.bot_token or not self.chat_id:
            raise RuntimeError(
                "Identifiants Telegram manquants : "
                "TELEGRAM_BOT_TOKEN et TELEGRAM_CHAT_ID."
            )

    def send_messages(self, messages: list[str]) -> str | None:
        cleaned = [message.strip() for message in messages if message.strip()]
        if not cleaned:
            return None

        if DRY_RUN:
            print("\n===== DRY RUN TELEGRAM =====")
            for index, message in enumerate(cleaned, start=1):
                print(f"\n--- Message {index}/{len(cleaned)} ---\n{message}")
            print("\n============================\n")
            return None

        self._require_settings()
        first_message_id: str | None = None
        url = f"{TELEGRAM_API_BASE}/bot{self.bot_token}/sendMessage"

        for message in cleaned:
            # Telegram limite un message texte à 4096 caractères.
            chunks = [message[i:i + 4000] for i in range(0, len(message), 4000)]
            for chunk in chunks:
                response = requests.post(
                    url,
                    json={
                        "chat_id": self.chat_id,
                        "text": chunk,
                        "disable_web_page_preview": True,
                    },
                    timeout=30,
                )
                if response.status_code >= 300:
                    raise RuntimeError(
                        f"Erreur Telegram {response.status_code}: {response.text}"
                    )
                payload = response.json()
                message_id = str(payload["result"]["message_id"])
                first_message_id = first_message_id or message_id

        return first_message_id


# ---------------------------------------------------------------------------
# Rapports
# ---------------------------------------------------------------------------

def actual_result(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "HOME"
    if home_goals < away_goals:
        return "AWAY"
    return "DRAW"


def build_bilan(rows: list[dict[str, Any]], label: str) -> str:
    total = len(rows)
    result_ok = 0
    over_ok = 0
    btts_ok = 0
    goal_errors: list[float] = []

    for row in rows:
        home_goals = int(row["home_goals"])
        away_goals = int(row["away_goals"])
        total_goals = home_goals + away_goals

        result_ok += int(row["predicted_result"] == actual_result(home_goals, away_goals))
        over_ok += int((float(row["over_25_pct"]) >= 50) == (total_goals >= 3))
        btts_ok += int((float(row["btts_pct"]) >= 50) == (home_goals > 0 and away_goals > 0))
        expected_total = float(row["expected_home_goals"]) + float(row["expected_away_goals"])
        goal_errors.append(abs(expected_total - total_goals))

    mae = sum(goal_errors) / len(goal_errors)
    return (
        f"📊 XGenius — {label}\n"
        f"{total} matchs évalués\n"
        f"1X2 : {result_ok}/{total} ({100 * result_ok / total:.0f} %)\n"
        f"Over/Under 2,5 : {over_ok}/{total} ({100 * over_ok / total:.0f} %)\n"
        f"BTTS : {btts_ok}/{total} ({100 * btts_ok / total:.0f} %)\n"
        f"Erreur moyenne sur les buts : {mae:.2f}"
    )


def choose_radar_matches(rows: list[dict[str, Any]], maximum: int) -> list[tuple[str, str, dict[str, Any]]]:
    if not rows:
        return []

    choices: list[tuple[str, str, dict[str, Any]]] = []
    used: set[int] = set()

    selectors = [
        ("🎯", "Signal 1X2 le plus net", lambda r: float(r["confidence"]), True),
        ("🔥", "Potentiel offensif", lambda r: float(r["over_25_pct"]), True),
        ("🤝", "BTTS à surveiller", lambda r: float(r["btts_pct"]), True),
        (
            "⚖️",
            "Match le plus indécis",
            lambda r: max(float(r["home_pct"]), float(r["draw_pct"]), float(r["away_pct"]))
            - min(float(r["home_pct"]), float(r["draw_pct"]), float(r["away_pct"])),
            False,
        ),
        ("🧊", "Match potentiellement fermé", lambda r: float(r["over_25_pct"]), False),
    ]

    for emoji, title, key, reverse in selectors:
        candidates = [row for row in rows if int(row["fixture_id"]) not in used]
        if not candidates:
            break
        selected = sorted(candidates, key=key, reverse=reverse)[0]
        choices.append((emoji, title, selected))
        used.add(int(selected["fixture_id"]))
        if len(choices) >= maximum:
            return choices

    # Complète avec les meilleures confiances si MAX_RADAR_MATCHES > 5.
    for row in sorted(rows, key=lambda r: float(r["confidence"]), reverse=True):
        if int(row["fixture_id"]) in used:
            continue
        choices.append(("🔎", "Autre match à suivre", row))
        used.add(int(row["fixture_id"]))
        if len(choices) >= maximum:
            break

    return choices


def format_match_post(emoji: str, title: str, row: dict[str, Any]) -> str:
    kickoff = row["kickoff"].astimezone(PARIS_TZ)
    weekdays = ["lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim."]
    kickoff_text = f"{weekdays[kickoff.weekday()]} {kickoff.strftime('%d/%m à %H:%M')}"
    signal = side_label(row["predicted_result"], row["home_team"], row["away_team"])

    return (
        f"{emoji} {title}\n"
        f"{row['home_team']} – {row['away_team']}\n"
        f"🗓 {kickoff_text}\n"
        f"1X2 : {row['home_pct']:.0f}% | {row['draw_pct']:.0f}% | {row['away_pct']:.0f}%\n"
        f"Buts : {row['expected_home_goals']:.1f}-{row['expected_away_goals']:.1f} | "
        f"+2,5 {row['over_25_pct']:.0f}% | BTTS {row['btts_pct']:.0f}%\n"
        f"Signal : {signal} ({row['confidence']:.0f}%) | Profil : {row['profile']}"
    )


def format_compact_match(row: dict[str, Any]) -> str:
    kickoff = row["kickoff"].astimezone(PARIS_TZ)
    signal = side_label(row["predicted_result"], row["home_team"], row["away_team"])
    return (
        f"• {kickoff.strftime('%d/%m %H:%M')} — {row['home_team']} – {row['away_team']}\n"
        f"  1X2 {row['home_pct']:.0f}/{row['draw_pct']:.0f}/{row['away_pct']:.0f} | "
        f"+2,5 {row['over_25_pct']:.0f}% | BTTS {row['btts_pct']:.0f}% | "
        f"{signal} ({row['confidence']:.0f}%)"
    )


def build_all_matches_messages(rows: list[dict[str, Any]], periods: Periods) -> list[str]:
    if not SHOW_ALL_MATCHES or not rows:
        return []

    ordered = sorted(rows, key=lambda row: row["kickoff"])[:MAX_FULL_MATCHES]
    header = (
        f"📋 Tous les matchs analysés — {periods.label}\n"
        f"Du {periods.future_start.strftime('%d/%m')} au {periods.future_end_inclusive.strftime('%d/%m')}\n"
        f"{len(ordered)} matchs listés sur {len(rows)}."
    )

    messages: list[str] = []
    current_lines = [header]
    current_len = len(header)

    for row in ordered:
        line = format_compact_match(row)
        # Telegram accepte 4096 caractères ; on garde une marge.
        if current_len + len(line) + 2 > 3600:
            messages.append("\n".join(current_lines))
            current_lines = ["📋 Suite des matchs analysés"]
            current_len = len(current_lines[0])
        current_lines.append(line)
        current_len += len(line) + 1

    if current_lines:
        messages.append("\n".join(current_lines))

    if len(rows) > len(ordered):
        messages.append(f"ℹ️ Liste limitée à {MAX_FULL_MATCHES} matchs. Augmente MAX_FULL_MATCHES si besoin.")

    return messages


def build_radar_messages(rows: list[dict[str, Any]], periods: Periods) -> list[str]:
    selected = choose_radar_matches(rows, MAX_RADAR_MATCHES)
    if not selected:
        return []

    root = (
        f"⚽ XGenius — {periods.label}\n"
        f"Du {periods.future_start.strftime('%d/%m')} au {periods.future_end_inclusive.strftime('%d/%m')}\n"
        f"{len(rows)} matchs analysés, {len(selected)} tops affichés.\n"
        "1X2, buts, BTTS et confiance"
    )
    messages = [root] + [format_match_post(emoji, title, row) for emoji, title, row in selected]
    messages.extend(build_all_matches_messages(rows, periods))
    return messages


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def resolve_mode(requested: str) -> str:
    if requested in {"monday", "thursday"}:
        return requested
    weekday = datetime.now(PARIS_TZ).weekday()
    if weekday == 0:
        return "monday"
    if weekday == 3:
        return "thursday"
    raise RuntimeError("Le mode auto ne s'exécute que le lundi ou le jeudi.")


def anchor_for_mode(mode: str, reference: date) -> date:
    """Calcule la date d'ancrage du traitement.

    - monday couvre lundi → jeudi exclu.
      Si on lance manuellement lundi, mardi ou mercredi, on garde le lundi
      de la semaine en cours. Sinon on prépare le lundi suivant.

    - thursday couvre jeudi → lundi exclu.
      Si on lance manuellement jeudi, vendredi, samedi ou dimanche, on garde
      le jeudi de la semaine en cours. Sinon on prépare le jeudi suivant.
    """
    weekday = reference.weekday()

    if mode == "monday":
        if 0 <= weekday <= 2:
            return reference - timedelta(days=weekday)
        return reference + timedelta(days=(7 - weekday) % 7)

    if 3 <= weekday <= 6:
        return reference - timedelta(days=weekday - 3)
    return reference + timedelta(days=3 - weekday)


def periods_for(mode: str, today: date) -> Periods:
    if mode == "monday":
        anchor = anchor_for_mode(mode, today)
        return Periods(
            label="Radar lundi → mercredi",
            future_start=anchor,
            future_end_exclusive=anchor + timedelta(days=3),
            bilan_label="Bilan du week-end",
            bilan_start=anchor - timedelta(days=3),
            bilan_end_exclusive=anchor,
        )

    anchor = anchor_for_mode(mode, today)
    return Periods(
        label="Radar jeudi → dimanche",
        future_start=anchor,
        future_end_exclusive=anchor + timedelta(days=4),
        bilan_label="Bilan lundi → mercredi",
        bilan_start=anchor - timedelta(days=3),
        bilan_end_exclusive=anchor,
    )


def collect_dates(api: FootballAPI, db: Database, days: Iterable[date]) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    for day in days:
        print(f"Récupération des matchs du {day.isoformat()}…")
        fixtures = api.fixtures_for_date(day)
        for fixture in fixtures:
            db.upsert_fixture(fixture)
        db.commit()
        collected.extend(fixtures)
        print(f"{len(fixtures)} matchs suivis.")
    return collected


def publish_bilan(db: Database, telegram: TelegramClient, mode: str, periods: Periods) -> None:
    rows = db.unreported_completed(periods.bilan_start, periods.bilan_end_exclusive)
    if not rows:
        print("Aucun nouveau match terminé à intégrer au bilan pour cette période.")
        return

    report_key = (
        f"bilan:{mode}:"
        f"{periods.bilan_start.isoformat()}:{periods.bilan_end_exclusive.isoformat()}"
    )
    if db.report_exists(report_key):
        print("Bilan déjà publié pour cette période.")
        return

    label = (
        f"{periods.bilan_label} "
        f"({periods.bilan_start.strftime('%d/%m')} → "
        f"{periods.bilan_end_inclusive.strftime('%d/%m')})"
    )
    message = build_bilan(rows, label)
    message_id = telegram.send_messages([message])

    if not DRY_RUN:
        db.complete_bilan(
            report_key,
            message_id,
            [int(row["fixture_id"]) for row in rows],
        )


def generate_predictions(
    api: FootballAPI,
    db: Database,
    fixtures: list[dict[str, Any]],
) -> int:
    now = datetime.now(PARIS_TZ)
    future = []
    seen: set[int] = set()

    for fixture in fixtures:
        fixture_id = int(fixture["fixture"]["id"])
        if fixture_id in seen:
            continue
        seen.add(fixture_id)
        status = fixture["fixture"].get("status", {}).get("short")
        kickoff = parse_api_datetime(fixture["fixture"]["date"])
        if status in FUTURE_STATUSES and kickoff > now:
            future.append(fixture)

    future.sort(key=lambda fixture: parse_api_datetime(fixture["fixture"]["date"]))
    future = future[:MAX_PREDICTIONS_PER_RUN]

    saved = 0
    for index, fixture in enumerate(future, start=1):
        fixture_id = int(fixture["fixture"]["id"])
        print(f"Prédiction {index}/{len(future)} — fixture {fixture_id}")
        try:
            payload = api.prediction(fixture_id)
            if payload and db.upsert_prediction(fixture_id, payload):
                saved += 1
            else:
                print("Prédiction indisponible pour ce match.")
            db.commit()
        except Exception as exc:  # on continue pour ne pas perdre tout le batch
            db.rollback()
            print(f"Prédiction ignorée pour {fixture_id}: {exc}")

    return saved


def publish_radar(
    db: Database,
    telegram: TelegramClient,
    mode: str,
    today: date,
    periods: Periods,
) -> None:
    report_key = (
        f"radar:{mode}:"
        f"{periods.future_start.isoformat()}:{periods.future_end_exclusive.isoformat()}"
    )
    if db.report_exists(report_key):
        print("Radar déjà publié pour cette exécution.")
        return

    rows = db.future_predictions(periods.future_start, periods.future_end_exclusive)
    messages = build_radar_messages(rows, periods)
    if not messages:
        print("Aucune prédiction disponible pour le radar.")
        return

    message_id = telegram.send_messages(messages)
    if not DRY_RUN:
        db.save_report(report_key, message_id)


def run(mode: str) -> None:
    require_settings()
    today = datetime.now(PARIS_TZ).date()
    periods = periods_for(mode, today)
    print(
        "Période bilan : "
        f"{periods.bilan_start.isoformat()} → "
        f"{periods.bilan_end_exclusive.isoformat()} exclu"
    )
    print(
        "Période radar : "
        f"{periods.future_start.isoformat()} → "
        f"{periods.future_end_exclusive.isoformat()} exclu"
    )

    api = FootballAPI(RAPIDAPI_KEY)
    db = Database(DATABASE_URL)
    telegram = TelegramClient()

    try:
        db.ensure_schema()

        past_fixtures = collect_dates(
            api,
            db,
            daterange(periods.bilan_start, periods.bilan_end_exclusive),
        )
        print(f"Historique actualisé : {len(past_fixtures)} matchs récupérés.")
        publish_bilan(db, telegram, mode, periods)

        future_fixtures = collect_dates(
            api,
            db,
            daterange(periods.future_start, periods.future_end_exclusive),
        )
        saved = generate_predictions(api, db, future_fixtures)
        print(f"{saved} prédictions enregistrées ou actualisées.")
        publish_radar(db, telegram, mode, today, periods)

    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGenius Match Radar")
    parser.add_argument(
        "--mode",
        choices=("auto", "monday", "thursday"),
        default="auto",
        help="Mode d'exécution. 'auto' dépend du jour courant.",
    )
    args = parser.parse_args()

    try:
        run(resolve_mode(args.mode))
    except Exception as error:
        print(f"ERREUR XGENIUS : {error}", file=sys.stderr)
        raise
