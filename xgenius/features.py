# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "league_id",
    "league_avg_goals_60",
    "league_over25_rate_60",
    "league_btts_rate_60",
    "league_samples_60",
    "home_goals_for_5",
    "home_goals_against_5",
    "home_points_5",
    "home_over25_rate_5",
    "home_btts_rate_5",
    "home_samples_5",
    "away_goals_for_5",
    "away_goals_against_5",
    "away_points_5",
    "away_over25_rate_5",
    "away_btts_rate_5",
    "away_samples_5",
    "home_home_goals_for_5",
    "home_home_goals_against_5",
    "home_home_points_5",
    "home_home_samples_5",
    "away_away_goals_for_5",
    "away_away_goals_against_5",
    "away_away_points_5",
    "away_away_samples_5",
    "home_shots_for_5",
    "home_sot_for_5",
    "home_xg_for_5",
    "home_xg_against_5",
    "away_shots_for_5",
    "away_sot_for_5",
    "away_xg_for_5",
    "away_xg_against_5",
    "attack_balance",
    "defense_balance",
    "form_balance",
    "expected_base_home",
    "expected_base_away",
]


def fixtures_to_df(fixtures: List[Dict[str, Any]]) -> pd.DataFrame:
    if not fixtures:
        return pd.DataFrame()
    df = pd.DataFrame(fixtures)
    df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], utc=True)
    for c in ["fixture_id", "league_id", "home_team_id", "away_team_id", "home_goals", "away_goals"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("kickoff_utc").reset_index(drop=True)


def stats_to_df(stats: List[Dict[str, Any]]) -> pd.DataFrame:
    if not stats:
        return pd.DataFrame()
    df = pd.DataFrame(stats)
    for c in df.columns:
        if c not in {"team_name", "updated_at"}:
            df[c] = pd.to_numeric(df[c], errors="coerce") if c != "is_home" else df[c]
    return df


def _safe_mean(values, default=0.0) -> float:
    arr = pd.Series(values).dropna()
    if arr.empty:
        return float(default)
    return float(arr.mean())


def _rate(values, default=0.0) -> float:
    arr = pd.Series(values).dropna()
    if arr.empty:
        return float(default)
    return float(arr.mean())


def _team_matches_before(hist: pd.DataFrame, team_id: int, date_ref, n: int = 5, role: Optional[str] = None) -> pd.DataFrame:
    if hist.empty or pd.isna(team_id):
        return pd.DataFrame()
    q = hist["kickoff_utc"] < date_ref
    if role == "home":
        q &= hist["home_team_id"] == team_id
    elif role == "away":
        q &= hist["away_team_id"] == team_id
    else:
        q &= (hist["home_team_id"] == team_id) | (hist["away_team_id"] == team_id)
    return hist.loc[q].sort_values("kickoff_utc", ascending=False).head(n).copy()


def _league_matches_before(hist: pd.DataFrame, league_id: int, date_ref, n: int = 60) -> pd.DataFrame:
    if hist.empty:
        return pd.DataFrame()
    q = (hist["league_id"] == league_id) & (hist["kickoff_utc"] < date_ref)
    return hist.loc[q].sort_values("kickoff_utc", ascending=False).head(n).copy()


def _team_form_features(matches: pd.DataFrame, team_id: int) -> Dict[str, float]:
    if matches.empty:
        return {
            "goals_for": 0.0, "goals_against": 0.0, "points": 0.0,
            "over25_rate": 0.0, "btts_rate": 0.0, "samples": 0.0,
        }
    gf, ga, pts, over25, btts = [], [], [], [], []
    for _, r in matches.iterrows():
        is_home = int(r["home_team_id"]) == int(team_id)
        my_goals = float(r["home_goals"] if is_home else r["away_goals"])
        opp_goals = float(r["away_goals"] if is_home else r["home_goals"])
        gf.append(my_goals)
        ga.append(opp_goals)
        pts.append(3.0 if my_goals > opp_goals else 1.0 if my_goals == opp_goals else 0.0)
        over25.append(1.0 if (my_goals + opp_goals) > 2.5 else 0.0)
        btts.append(1.0 if my_goals > 0 and opp_goals > 0 else 0.0)
    return {
        "goals_for": _safe_mean(gf),
        "goals_against": _safe_mean(ga),
        "points": _safe_mean(pts),
        "over25_rate": _rate(over25),
        "btts_rate": _rate(btts),
        "samples": float(len(matches)),
    }


def _team_stat_features(matches: pd.DataFrame, stats_df: pd.DataFrame, team_id: int, opponent_id: int) -> Dict[str, float]:
    base = {"shots_for": 0.0, "sot_for": 0.0, "xg_for": 0.0, "xg_against": 0.0}
    if matches.empty or stats_df.empty or pd.isna(team_id):
        return base
    fixture_ids = matches["fixture_id"].astype(int).tolist()
    s_for = stats_df[(stats_df["fixture_id"].isin(fixture_ids)) & (stats_df["team_id"] == team_id)]
    if not s_for.empty:
        base["shots_for"] = _safe_mean(s_for.get("shots", pd.Series(dtype=float)))
        base["sot_for"] = _safe_mean(s_for.get("shots_on_goal", pd.Series(dtype=float)))
        base["xg_for"] = _safe_mean(s_for.get("xg", pd.Series(dtype=float)))
    xga = []
    for _, m in matches.iterrows():
        opp = int(m["away_team_id"]) if int(m["home_team_id"]) == int(team_id) else int(m["home_team_id"])
        row = stats_df[(stats_df["fixture_id"] == int(m["fixture_id"])) & (stats_df["team_id"] == opp)]
        if not row.empty and "xg" in row:
            val = row.iloc[0].get("xg")
            if pd.notna(val):
                xga.append(float(val))
    base["xg_against"] = _safe_mean(xga)
    return base


def _league_features(matches: pd.DataFrame) -> Dict[str, float]:
    if matches.empty:
        return {"avg_goals": 2.50, "over25_rate": 0.50, "btts_rate": 0.50, "samples": 0.0}
    total = matches["home_goals"].astype(float) + matches["away_goals"].astype(float)
    return {
        "avg_goals": float(total.mean()),
        "over25_rate": float((total > 2.5).mean()),
        "btts_rate": float(((matches["home_goals"] > 0) & (matches["away_goals"] > 0)).mean()),
        "samples": float(len(matches)),
    }


def build_features_for_fixture(fixture: Dict[str, Any], hist_df: pd.DataFrame, stats_df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    if hist_df is None:
        hist_df = pd.DataFrame()
    if stats_df is None:
        stats_df = pd.DataFrame()

    date_ref = pd.to_datetime(fixture["kickoff_utc"], utc=True)
    league_id = int(fixture["league_id"])
    home_id = int(fixture["home_team_id"]) if fixture.get("home_team_id") is not None else -1
    away_id = int(fixture["away_team_id"]) if fixture.get("away_team_id") is not None else -2

    home_matches = _team_matches_before(hist_df, home_id, date_ref, 5)
    away_matches = _team_matches_before(hist_df, away_id, date_ref, 5)
    home_home_matches = _team_matches_before(hist_df, home_id, date_ref, 5, role="home")
    away_away_matches = _team_matches_before(hist_df, away_id, date_ref, 5, role="away")
    league_matches = _league_matches_before(hist_df, league_id, date_ref, 60)

    hf = _team_form_features(home_matches, home_id)
    af = _team_form_features(away_matches, away_id)
    hhf = _team_form_features(home_home_matches, home_id)
    aaf = _team_form_features(away_away_matches, away_id)
    lf = _league_features(league_matches)

    hs = _team_stat_features(home_matches, stats_df, home_id, away_id)
    aas = _team_stat_features(away_matches, stats_df, away_id, home_id)

    # Baseline lissée : forme + ligue, utile en fallback et comme feature.
    league_half = lf["avg_goals"] / 2.0
    base_home = 0.40 * hf["goals_for"] + 0.35 * af["goals_against"] + 0.25 * league_half + 0.15
    base_away = 0.40 * af["goals_for"] + 0.35 * hf["goals_against"] + 0.25 * league_half
    base_home = float(np.clip(base_home, 0.05, 5.5))
    base_away = float(np.clip(base_away, 0.05, 5.5))

    features = {
        "league_id": float(league_id),
        "league_avg_goals_60": lf["avg_goals"],
        "league_over25_rate_60": lf["over25_rate"],
        "league_btts_rate_60": lf["btts_rate"],
        "league_samples_60": lf["samples"],
        "home_goals_for_5": hf["goals_for"],
        "home_goals_against_5": hf["goals_against"],
        "home_points_5": hf["points"],
        "home_over25_rate_5": hf["over25_rate"],
        "home_btts_rate_5": hf["btts_rate"],
        "home_samples_5": hf["samples"],
        "away_goals_for_5": af["goals_for"],
        "away_goals_against_5": af["goals_against"],
        "away_points_5": af["points"],
        "away_over25_rate_5": af["over25_rate"],
        "away_btts_rate_5": af["btts_rate"],
        "away_samples_5": af["samples"],
        "home_home_goals_for_5": hhf["goals_for"],
        "home_home_goals_against_5": hhf["goals_against"],
        "home_home_points_5": hhf["points"],
        "home_home_samples_5": hhf["samples"],
        "away_away_goals_for_5": aaf["goals_for"],
        "away_away_goals_against_5": aaf["goals_against"],
        "away_away_points_5": aaf["points"],
        "away_away_samples_5": aaf["samples"],
        "home_shots_for_5": hs["shots_for"],
        "home_sot_for_5": hs["sot_for"],
        "home_xg_for_5": hs["xg_for"],
        "home_xg_against_5": hs["xg_against"],
        "away_shots_for_5": aas["shots_for"],
        "away_sot_for_5": aas["sot_for"],
        "away_xg_for_5": aas["xg_for"],
        "away_xg_against_5": aas["xg_against"],
        "attack_balance": hf["goals_for"] - af["goals_for"],
        "defense_balance": af["goals_against"] - hf["goals_against"],
        "form_balance": hf["points"] - af["points"],
        "expected_base_home": base_home,
        "expected_base_away": base_away,
    }
    return {k: float(features.get(k, 0.0) or 0.0) for k in FEATURE_COLUMNS}


def features_to_matrix(feature_rows: List[Dict[str, float]]) -> np.ndarray:
    return np.array([[float(r.get(c, 0.0) or 0.0) for c in FEATURE_COLUMNS] for r in feature_rows], dtype=float)


def build_training_dataset(fixtures: List[Dict[str, Any]], stats: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    hist_df = fixtures_to_df(fixtures)
    stats_df = stats_to_df(stats)
    if hist_df.empty:
        return pd.DataFrame(), np.empty((0, len(FEATURE_COLUMNS))), np.array([]), np.array([])

    rows = []
    feature_rows = []
    y_home, y_away = [], []
    completed = hist_df.dropna(subset=["home_goals", "away_goals"]).sort_values("kickoff_utc")
    for _, r in completed.iterrows():
        f = r.to_dict()
        feats = build_features_for_fixture(f, hist_df, stats_df)
        feature_rows.append(feats)
        y_home.append(float(r["home_goals"]))
        y_away.append(float(r["away_goals"]))
        rows.append({
            "fixture_id": int(r["fixture_id"]),
            "kickoff_utc": r["kickoff_utc"],
            "home_team_name": r.get("home_team_name"),
            "away_team_name": r.get("away_team_name"),
            "league_name": r.get("league_name"),
        })
    meta = pd.DataFrame(rows)
    X = features_to_matrix(feature_rows)
    return meta, X, np.array(y_home, dtype=float), np.array(y_away, dtype=float)
