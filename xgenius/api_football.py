# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo
import time

import requests

from .config import API_BASE_URL, API_HOST, API_KEY, LOCAL_TZ, MONITORED_LEAGUES, UTC_TZ


class ApiFootballError(RuntimeError):
    pass


class ApiFootballClient:
    def __init__(self, api_key: str = API_KEY, timeout: int = 30, sleep_seconds: float = 0.2):
        if not api_key:
            raise ApiFootballError("RAPIDAPI_KEY manquant dans les secrets GitHub.")
        self.session = requests.Session()
        self.headers = {
            "x-rapidapi-key": api_key,
            "x-rapidapi-host": API_HOST,
        }
        self.timeout = timeout
        self.sleep_seconds = sleep_seconds

    def _get(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{API_BASE_URL}/{endpoint.lstrip('/')}"
        r = self.session.get(url, headers=self.headers, params=params, timeout=self.timeout)
        if r.status_code >= 400:
            raise ApiFootballError(f"API-Football HTTP {r.status_code} sur {endpoint}: {r.text[:300]}")
        data = r.json()
        if data.get("errors"):
            # L'API renvoie parfois errors={} vide. On ne bloque que si non vide.
            if isinstance(data["errors"], dict) and len(data["errors"]) > 0:
                raise ApiFootballError(f"API-Football errors sur {endpoint}: {data['errors']}")
            if isinstance(data["errors"], list) and data["errors"]:
                raise ApiFootballError(f"API-Football errors sur {endpoint}: {data['errors']}")
        time.sleep(self.sleep_seconds)
        return data

    def fixtures_by_date(self, d: date) -> List[Dict[str, Any]]:
        data = self._get("fixtures", {"date": d.isoformat(), "timezone": "Europe/Paris"})
        raw = data.get("response", []) or []
        out: List[Dict[str, Any]] = []
        for item in raw:
            league_id = item.get("league", {}).get("id")
            if league_id not in MONITORED_LEAGUES:
                continue
            parsed = parse_fixture(item)
            if parsed:
                out.append(parsed)
        return out

    def fixtures_between_dates(self, start_local: date, end_local_exclusive: date) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        d = start_local
        while d < end_local_exclusive:
            out.extend(self.fixtures_by_date(d))
            d += timedelta(days=1)
        # dédoublonnage par fixture_id
        uniq = {int(f["fixture_id"]): f for f in out}
        return list(uniq.values())

    def statistics_for_fixture(self, fixture_id: int, home_team_id: Optional[int] = None, away_team_id: Optional[int] = None) -> List[Dict[str, Any]]:
        data = self._get("fixtures/statistics", {"fixture": fixture_id})
        raw = data.get("response", []) or []
        rows: List[Dict[str, Any]] = []
        for team_block in raw:
            team = team_block.get("team", {}) or {}
            team_id = team.get("id")
            if team_id is None:
                continue
            is_home = bool(home_team_id is not None and int(team_id) == int(home_team_id))
            rows.append(parse_team_stats(fixture_id, team_block, is_home=is_home))
        return rows


def parse_fixture(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    fixture = item.get("fixture", {}) or {}
    league = item.get("league", {}) or {}
    teams = item.get("teams", {}) or {}
    goals = item.get("goals", {}) or {}
    status = fixture.get("status", {}) or {}

    fid = fixture.get("id")
    raw_date = fixture.get("date")
    home = teams.get("home", {}) or {}
    away = teams.get("away", {}) or {}
    league_id = league.get("id")
    if not fid or not raw_date or not home.get("name") or not away.get("name") or not league_id:
        return None

    kickoff = datetime.fromisoformat(str(raw_date).replace("Z", "+00:00"))
    if kickoff.tzinfo is None:
        kickoff = kickoff.replace(tzinfo=UTC_TZ)
    kickoff_utc = kickoff.astimezone(UTC_TZ)

    return {
        "fixture_id": int(fid),
        "league_id": int(league_id),
        "league_name": league.get("name") or MONITORED_LEAGUES.get(int(league_id), str(league_id)),
        "season": league.get("season"),
        "kickoff_utc": kickoff_utc,
        "status_short": status.get("short"),
        "status_long": status.get("long"),
        "elapsed": status.get("elapsed"),
        "home_team_id": home.get("id"),
        "home_team_name": home.get("name"),
        "away_team_id": away.get("id"),
        "away_team_name": away.get("name"),
        "home_goals": goals.get("home"),
        "away_goals": goals.get("away"),
    }


def _stat_value(stats: List[Dict[str, Any]], names: Iterable[str]) -> Optional[float]:
    name_set = {n.lower() for n in names}
    for s in stats:
        typ = str(s.get("type", "")).strip().lower()
        if typ not in name_set:
            continue
        val = s.get("value")
        if val is None:
            return None
        if isinstance(val, str):
            val = val.strip().replace("%", "")
            if val == "":
                return None
        try:
            return float(val)
        except Exception:
            return None
    return None


def parse_team_stats(fixture_id: int, block: Dict[str, Any], is_home: bool) -> Dict[str, Any]:
    team = block.get("team", {}) or {}
    stats = block.get("statistics", []) or []
    return {
        "fixture_id": int(fixture_id),
        "team_id": int(team.get("id")),
        "team_name": team.get("name") or str(team.get("id")),
        "is_home": is_home,
        "shots": _stat_value(stats, ["Total Shots"]),
        "shots_on_goal": _stat_value(stats, ["Shots on Goal"]),
        "shots_off_goal": _stat_value(stats, ["Shots off Goal"]),
        "blocked_shots": _stat_value(stats, ["Blocked Shots"]),
        "shots_inside_box": _stat_value(stats, ["Shots insidebox", "Shots inside box"]),
        "shots_outside_box": _stat_value(stats, ["Shots outsidebox", "Shots outside box"]),
        "fouls": _stat_value(stats, ["Fouls"]),
        "corners": _stat_value(stats, ["Corner Kicks"]),
        "offsides": _stat_value(stats, ["Offsides"]),
        "possession": _stat_value(stats, ["Ball Possession"]),
        "yellow_cards": _stat_value(stats, ["Yellow Cards"]),
        "red_cards": _stat_value(stats, ["Red Cards"]),
        "goalkeeper_saves": _stat_value(stats, ["Goalkeeper Saves"]),
        "total_passes": _stat_value(stats, ["Total passes", "Total Passes"]),
        "passes_accurate": _stat_value(stats, ["Passes accurate", "Passes Accurate"]),
        "passes_pct": _stat_value(stats, ["Passes %"]),
        "xg": _stat_value(stats, ["expected goals", "Expected Goals", "xG"]),
    }
