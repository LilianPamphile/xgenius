# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Tuple

import psycopg2
from psycopg2.extras import Json, RealDictCursor, execute_values

from .config import DATABASE_URL


@contextmanager
def get_conn():
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ai_fixtures (
                fixture_id BIGINT PRIMARY KEY,
                league_id INTEGER NOT NULL,
                league_name TEXT NOT NULL,
                season INTEGER,
                kickoff_utc TIMESTAMPTZ NOT NULL,
                status_short TEXT,
                status_long TEXT,
                elapsed INTEGER,
                home_team_id BIGINT,
                home_team_name TEXT NOT NULL,
                away_team_id BIGINT,
                away_team_name TEXT NOT NULL,
                home_goals INTEGER,
                away_goals INTEGER,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ai_fixtures_kickoff ON ai_fixtures(kickoff_utc);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ai_fixtures_league ON ai_fixtures(league_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ai_fixtures_status ON ai_fixtures(status_short);")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ai_team_match_stats (
                fixture_id BIGINT NOT NULL REFERENCES ai_fixtures(fixture_id) ON DELETE CASCADE,
                team_id BIGINT NOT NULL,
                team_name TEXT NOT NULL,
                is_home BOOLEAN NOT NULL,
                shots NUMERIC,
                shots_on_goal NUMERIC,
                shots_off_goal NUMERIC,
                blocked_shots NUMERIC,
                shots_inside_box NUMERIC,
                shots_outside_box NUMERIC,
                fouls NUMERIC,
                corners NUMERIC,
                offsides NUMERIC,
                possession NUMERIC,
                yellow_cards NUMERIC,
                red_cards NUMERIC,
                goalkeeper_saves NUMERIC,
                total_passes NUMERIC,
                passes_accurate NUMERIC,
                passes_pct NUMERIC,
                xg NUMERIC,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (fixture_id, team_id)
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ai_stats_team ON ai_team_match_stats(team_id);")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ai_predictions (
                fixture_id BIGINT PRIMARY KEY REFERENCES ai_fixtures(fixture_id) ON DELETE CASCADE,
                generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                model_version TEXT NOT NULL,
                model_kind TEXT NOT NULL,
                lambda_home NUMERIC NOT NULL,
                lambda_away NUMERIC NOT NULL,
                p_home NUMERIC NOT NULL,
                p_draw NUMERIC NOT NULL,
                p_away NUMERIC NOT NULL,
                p_over25 NUMERIC NOT NULL,
                p_btts NUMERIC NOT NULL,
                confidence NUMERIC NOT NULL,
                profile TEXT NOT NULL,
                signal TEXT NOT NULL,
                features JSONB,
                evaluated BOOLEAN NOT NULL DEFAULT FALSE,
                evaluation JSONB,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ai_predictions_generated ON ai_predictions(generated_at);")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ai_model_runs (
                model_version TEXT PRIMARY KEY,
                trained_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                model_family TEXT NOT NULL,
                n_samples INTEGER NOT NULL,
                metrics JSONB NOT NULL,
                is_active BOOLEAN NOT NULL DEFAULT FALSE,
                artifact BYTEA,
                notes TEXT
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ai_model_runs_active ON ai_model_runs(is_active, trained_at DESC);")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ai_reports (
                report_key TEXT PRIMARY KEY,
                mode TEXT NOT NULL,
                report_type TEXT NOT NULL,
                period_start DATE NOT NULL,
                period_end DATE NOT NULL,
                sent_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                dry_run BOOLEAN NOT NULL DEFAULT FALSE,
                content_preview TEXT
            );
            """
        )


def upsert_fixtures(conn, fixtures: List[Dict[str, Any]]) -> int:
    if not fixtures:
        return 0
    values = []
    for f in fixtures:
        values.append((
            f["fixture_id"], f["league_id"], f["league_name"], f.get("season"), f["kickoff_utc"],
            f.get("status_short"), f.get("status_long"), f.get("elapsed"),
            f.get("home_team_id"), f["home_team_name"], f.get("away_team_id"), f["away_team_name"],
            f.get("home_goals"), f.get("away_goals"),
        ))
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO ai_fixtures (
                fixture_id, league_id, league_name, season, kickoff_utc,
                status_short, status_long, elapsed,
                home_team_id, home_team_name, away_team_id, away_team_name,
                home_goals, away_goals
            ) VALUES %s
            ON CONFLICT (fixture_id) DO UPDATE SET
                league_id = EXCLUDED.league_id,
                league_name = EXCLUDED.league_name,
                season = EXCLUDED.season,
                kickoff_utc = EXCLUDED.kickoff_utc,
                status_short = EXCLUDED.status_short,
                status_long = EXCLUDED.status_long,
                elapsed = EXCLUDED.elapsed,
                home_team_id = EXCLUDED.home_team_id,
                home_team_name = EXCLUDED.home_team_name,
                away_team_id = EXCLUDED.away_team_id,
                away_team_name = EXCLUDED.away_team_name,
                home_goals = EXCLUDED.home_goals,
                away_goals = EXCLUDED.away_goals,
                updated_at = NOW()
            """,
            values,
        )
    return len(values)


def upsert_team_stats(conn, rows: List[Dict[str, Any]]) -> int:
    if not rows:
        return 0
    cols = [
        "fixture_id", "team_id", "team_name", "is_home", "shots", "shots_on_goal", "shots_off_goal",
        "blocked_shots", "shots_inside_box", "shots_outside_box", "fouls", "corners", "offsides",
        "possession", "yellow_cards", "red_cards", "goalkeeper_saves", "total_passes", "passes_accurate",
        "passes_pct", "xg",
    ]
    values = [tuple(r.get(c) for c in cols) for r in rows]
    with conn.cursor() as cur:
        execute_values(
            cur,
            f"""
            INSERT INTO ai_team_match_stats ({', '.join(cols)}) VALUES %s
            ON CONFLICT (fixture_id, team_id) DO UPDATE SET
                team_name = EXCLUDED.team_name,
                is_home = EXCLUDED.is_home,
                shots = EXCLUDED.shots,
                shots_on_goal = EXCLUDED.shots_on_goal,
                shots_off_goal = EXCLUDED.shots_off_goal,
                blocked_shots = EXCLUDED.blocked_shots,
                shots_inside_box = EXCLUDED.shots_inside_box,
                shots_outside_box = EXCLUDED.shots_outside_box,
                fouls = EXCLUDED.fouls,
                corners = EXCLUDED.corners,
                offsides = EXCLUDED.offsides,
                possession = EXCLUDED.possession,
                yellow_cards = EXCLUDED.yellow_cards,
                red_cards = EXCLUDED.red_cards,
                goalkeeper_saves = EXCLUDED.goalkeeper_saves,
                total_passes = EXCLUDED.total_passes,
                passes_accurate = EXCLUDED.passes_accurate,
                passes_pct = EXCLUDED.passes_pct,
                xg = EXCLUDED.xg,
                updated_at = NOW()
            """,
            values,
        )
    return len(values)


def fetch_fixtures_between(conn, start_utc, end_utc, completed_only: bool = False, future_only: bool = False) -> List[Dict[str, Any]]:
    where = ["kickoff_utc >= %s", "kickoff_utc < %s"]
    params: List[Any] = [start_utc, end_utc]
    if completed_only:
        where.append("status_short IN ('FT','AET','PEN')")
        where.append("home_goals IS NOT NULL")
        where.append("away_goals IS NOT NULL")
    if future_only:
        where.append("status_short NOT IN ('FT','AET','PEN','CANC','ABD')")
    sql = "SELECT * FROM ai_fixtures WHERE " + " AND ".join(where) + " ORDER BY kickoff_utc ASC"
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]


def fetch_all_completed_fixtures(conn) -> List[Dict[str, Any]]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT *
            FROM ai_fixtures
            WHERE status_short IN ('FT','AET','PEN')
              AND home_goals IS NOT NULL
              AND away_goals IS NOT NULL
            ORDER BY kickoff_utc ASC
            """
        )
        return [dict(r) for r in cur.fetchall()]


def fetch_all_team_stats(conn) -> List[Dict[str, Any]]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM ai_team_match_stats")
        return [dict(r) for r in cur.fetchall()]


def fetch_finished_without_stats(conn, limit: int) -> List[Dict[str, Any]]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT f.*
            FROM ai_fixtures f
            LEFT JOIN ai_team_match_stats s ON s.fixture_id = f.fixture_id
            WHERE f.status_short IN ('FT','AET','PEN')
              AND f.home_goals IS NOT NULL
              AND f.away_goals IS NOT NULL
              AND s.fixture_id IS NULL
            ORDER BY f.kickoff_utc DESC
            LIMIT %s
            """,
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]


def upsert_prediction(conn, row: Dict[str, Any]) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO ai_predictions (
                fixture_id, generated_at, model_version, model_kind,
                lambda_home, lambda_away, p_home, p_draw, p_away, p_over25, p_btts,
                confidence, profile, signal, features, evaluated, updated_at
            ) VALUES (
                %(fixture_id)s, NOW(), %(model_version)s, %(model_kind)s,
                %(lambda_home)s, %(lambda_away)s, %(p_home)s, %(p_draw)s, %(p_away)s, %(p_over25)s, %(p_btts)s,
                %(confidence)s, %(profile)s, %(signal)s, %(features)s, FALSE, NOW()
            )
            ON CONFLICT (fixture_id) DO UPDATE SET
                generated_at = NOW(),
                model_version = EXCLUDED.model_version,
                model_kind = EXCLUDED.model_kind,
                lambda_home = EXCLUDED.lambda_home,
                lambda_away = EXCLUDED.lambda_away,
                p_home = EXCLUDED.p_home,
                p_draw = EXCLUDED.p_draw,
                p_away = EXCLUDED.p_away,
                p_over25 = EXCLUDED.p_over25,
                p_btts = EXCLUDED.p_btts,
                confidence = EXCLUDED.confidence,
                profile = EXCLUDED.profile,
                signal = EXCLUDED.signal,
                features = EXCLUDED.features,
                evaluated = FALSE,
                evaluation = NULL,
                updated_at = NOW()
            """,
            {**row, "features": Json(row.get("features", {}))},
        )


def fetch_predictions_for_fixtures(conn, fixture_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    if not fixture_ids:
        return {}
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM ai_predictions WHERE fixture_id = ANY(%s)", (fixture_ids,))
        return {int(r["fixture_id"]): dict(r) for r in cur.fetchall()}


def fetch_predictions_between(conn, start_utc, end_utc) -> List[Dict[str, Any]]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT p.*, f.kickoff_utc, f.league_name, f.home_team_name, f.away_team_name,
                   f.home_goals, f.away_goals, f.status_short
            FROM ai_predictions p
            JOIN ai_fixtures f ON f.fixture_id = p.fixture_id
            WHERE f.kickoff_utc >= %s AND f.kickoff_utc < %s
            ORDER BY f.kickoff_utc ASC
            """,
            (start_utc, end_utc),
        )
        return [dict(r) for r in cur.fetchall()]


def save_model_run(conn, model_version: str, model_family: str, n_samples: int, metrics: Dict[str, Any], artifact: bytes, activate: bool, notes: str = "") -> None:
    with conn.cursor() as cur:
        if activate:
            cur.execute("UPDATE ai_model_runs SET is_active = FALSE WHERE is_active = TRUE")
        cur.execute(
            """
            INSERT INTO ai_model_runs (model_version, model_family, n_samples, metrics, artifact, is_active, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (model_version) DO UPDATE SET
                n_samples = EXCLUDED.n_samples,
                metrics = EXCLUDED.metrics,
                artifact = EXCLUDED.artifact,
                is_active = EXCLUDED.is_active,
                notes = EXCLUDED.notes,
                trained_at = NOW()
            """,
            (model_version, model_family, n_samples, Json(metrics), psycopg2.Binary(artifact), activate, notes),
        )


def fetch_active_model(conn) -> Optional[Dict[str, Any]]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT * FROM ai_model_runs
            WHERE is_active = TRUE AND artifact IS NOT NULL
            ORDER BY trained_at DESC
            LIMIT 1
            """
        )
        r = cur.fetchone()
        return dict(r) if r else None


def fetch_latest_model_metrics(conn) -> Optional[Dict[str, Any]]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM ai_model_runs WHERE is_active = TRUE ORDER BY trained_at DESC LIMIT 1")
        r = cur.fetchone()
        return dict(r) if r else None


def report_exists(conn, report_key: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM ai_reports WHERE report_key = %s", (report_key,))
        return cur.fetchone() is not None


def save_report(conn, report_key: str, mode: str, report_type: str, period_start, period_end, dry_run: bool, content_preview: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO ai_reports (report_key, mode, report_type, period_start, period_end, dry_run, content_preview)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (report_key) DO NOTHING
            """,
            (report_key, mode, report_type, period_start, period_end, dry_run, content_preview[:1000]),
        )
