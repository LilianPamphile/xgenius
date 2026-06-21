# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import json

from .api_football import ApiFootballClient, ApiFootballError
from .config import DEFAULT_BOOTSTRAP_DAYS, FINISHED_STATUS, LOCAL_TZ, MAX_STATS_IMPORT_PER_RUN, MODEL_FAMILY
from .db import (
    fetch_active_model,
    fetch_all_completed_fixtures,
    fetch_all_team_stats,
    fetch_finished_without_stats,
    fetch_fixtures_between,
    fetch_latest_model_metrics,
    fetch_predictions_between,
    get_conn,
    init_db,
    report_exists,
    save_model_run,
    save_report,
    upsert_fixtures,
    upsert_prediction,
    upsert_team_stats,
    fetch_predictions_for_fixtures,
)
from .evaluation import evaluate_predictions
from .features import fixtures_to_df, stats_to_df, build_features_for_fixture
from .modeling import (
    deserialize_bundle,
    prediction_from_feature,
    serialize_bundle,
    should_activate_candidate,
    train_candidate,
)
from .reporting import build_evaluation_report, build_radar_report
from .telegram import send_telegram_message
from .time_windows import build_run_window, to_utc


def _date_range_local(start_local, end_local_exclusive):
    d = start_local.date()
    end = end_local_exclusive.date()
    while d < end:
        yield d
        d = d + timedelta(days=1)


def ingest_fixtures_for_window(conn, client: ApiFootballClient, start_local, end_local_exclusive) -> int:
    fixtures = client.fixtures_between_dates(start_local.date(), end_local_exclusive.date())
    return upsert_fixtures(conn, fixtures)


def refresh_results_for_window(conn, client: ApiFootballClient, start_local, end_local_exclusive) -> int:
    # Même endpoint que calendrier : il contient les statuts et les scores mis à jour.
    return ingest_fixtures_for_window(conn, client, start_local, end_local_exclusive)


def ingest_missing_stats(conn, client: ApiFootballClient, limit: int = MAX_STATS_IMPORT_PER_RUN) -> int:
    fixtures = fetch_finished_without_stats(conn, limit=limit)
    inserted = 0
    for f in fixtures:
        try:
            rows = client.statistics_for_fixture(
                int(f["fixture_id"]),
                home_team_id=f.get("home_team_id"),
                away_team_id=f.get("away_team_id"),
            )
            inserted += upsert_team_stats(conn, rows)
        except Exception as e:
            print(f"Stats ignorées pour fixture {f.get('fixture_id')} : {e}")
    return inserted


def bootstrap_history(days: int = DEFAULT_BOOTSTRAP_DAYS, dry_run: bool = False) -> None:
    now = datetime.now(LOCAL_TZ)
    start = now - timedelta(days=days)
    end = now + timedelta(days=1)
    client = ApiFootballClient()
    with get_conn() as conn:
        init_db(conn)
        total = refresh_results_for_window(conn, client, start, end)
        stats_count = ingest_missing_stats(conn, client, limit=MAX_STATS_IMPORT_PER_RUN * 3)
        msg = f"🧱 Bootstrap XGenius IA\nPériode : {start.date()} → {end.date()}\nFixtures importées/actualisées : {total}\nStats importées : {stats_count}"
        send_telegram_message(msg, dry_run=dry_run)


def train_self_learning_model(conn) -> Dict[str, Any]:
    fixtures = fetch_all_completed_fixtures(conn)
    stats = fetch_all_team_stats(conn)
    candidate = train_candidate(fixtures, stats)
    if candidate is None:
        return {
            "trained": False,
            "activated": False,
            "message": f"Pas assez de matchs terminés pour entraîner le modèle. Historique actuel : {len(fixtures)} matchs.",
            "n_samples": len(fixtures),
            "metrics": None,
            "model_version": None,
        }

    active = fetch_latest_model_metrics(conn)
    activate, reason = should_activate_candidate(candidate, active)
    artifact = serialize_bundle(candidate)
    save_model_run(
        conn,
        model_version=candidate.model_version,
        model_family=MODEL_FAMILY,
        n_samples=candidate.n_samples,
        metrics=candidate.metrics,
        artifact=artifact,
        activate=activate,
        notes=reason,
    )
    return {
        "trained": True,
        "activated": activate,
        "message": f"Nouveau modèle entraîné : {candidate.model_version[:32]} | Activé : {'oui' if activate else 'non'} | {reason}",
        "n_samples": candidate.n_samples,
        "metrics": candidate.metrics,
        "model_version": candidate.model_version,
    }


def load_active_bundle(conn):
    active = fetch_active_model(conn)
    if not active:
        return None, None
    artifact = active.get("artifact")
    if artifact is None:
        return None, active
    try:
        # psycopg2 peut retourner memoryview pour BYTEA.
        if isinstance(artifact, memoryview):
            artifact = artifact.tobytes()
        return deserialize_bundle(bytes(artifact)), active
    except Exception as e:
        print(f"Impossible de charger le modèle actif, fallback baseline : {e}")
        return None, active


def predict_window(conn, start_local, end_local_exclusive) -> List[Dict[str, Any]]:
    start_utc = to_utc(start_local)
    end_utc = to_utc(end_local_exclusive)
    future_fixtures = fetch_fixtures_between(conn, start_utc, end_utc, future_only=True)
    completed = fetch_all_completed_fixtures(conn)
    stats = fetch_all_team_stats(conn)
    hist_df = fixtures_to_df(completed)
    stats_df = stats_to_df(stats)
    bundle, active_row = load_active_bundle(conn)

    predictions: List[Dict[str, Any]] = []
    for f in future_fixtures:
        feature = build_features_for_fixture(f, hist_df, stats_df)
        pred = prediction_from_feature(feature, f["home_team_name"], f["away_team_name"], bundle)
        row = {"fixture_id": int(f["fixture_id"]), **pred}
        upsert_prediction(conn, row)
        predictions.append({**f, **pred})
    return predictions


def evaluate_window(conn, start_local, end_local_exclusive) -> Dict[str, Any]:
    rows = fetch_predictions_between(conn, to_utc(start_local), to_utc(end_local_exclusive))
    return evaluate_predictions(rows)


def _report_key(prefix: str, mode: str, start_local, end_local_exclusive) -> str:
    return f"{prefix}:{mode}:{start_local.date()}:{end_local_exclusive.date()}"


def run_mode(mode: str, dry_run: bool = False, force: bool = False) -> None:
    window = build_run_window(mode)
    client = ApiFootballClient()
    with get_conn() as conn:
        init_db(conn)

        # 1) Résultats + stats post-match
        updated_eval = refresh_results_for_window(conn, client, window.evaluation_start_local, window.evaluation_end_local)
        updated_pred = ingest_fixtures_for_window(conn, client, window.prediction_start_local, window.prediction_end_local)
        stats_count = ingest_missing_stats(conn, client, MAX_STATS_IMPORT_PER_RUN)

        # 2) Évaluation + apprentissage automatique
        evaluation = evaluate_window(conn, window.evaluation_start_local, window.evaluation_end_local)
        train_summary = train_self_learning_model(conn)
        active_info = fetch_latest_model_metrics(conn)

        # 3) Prédictions IA de la période à venir
        predictions = predict_window(conn, window.prediction_start_local, window.prediction_end_local)
        pred_ids = [int(p["fixture_id"]) for p in predictions]
        # Les prédictions viennent d'être upsert, on prend les valeurs complètes stockées si besoin.

        # 4) Telegram bilan
        eval_key = _report_key("eval", window.mode, window.evaluation_start_local, window.evaluation_end_local)
        eval_message = build_evaluation_report(
            window.mode,
            window.evaluation_label,
            window.evaluation_start_local,
            window.evaluation_end_local,
            evaluation,
            active_info,
            train_summary,
        )
        eval_message += f"\n\n🔄 Données actualisées : {updated_eval + updated_pred} fixtures, {stats_count} lignes stats."
        if dry_run or force or not report_exists(conn, eval_key):
            send_telegram_message(eval_message, dry_run=dry_run)
            if not dry_run:
                save_report(conn, eval_key, window.mode, "evaluation", window.evaluation_start_local.date(), window.evaluation_end_local.date(), dry_run, eval_message)
        else:
            print(f"Bilan déjà envoyé : {eval_key}")

        # 5) Telegram radar
        radar_key = _report_key("radar", window.mode, window.prediction_start_local, window.prediction_end_local)
        _, active_row = load_active_bundle(conn)
        radar_blocks = build_radar_report(
            window.prediction_label,
            window.prediction_start_local,
            window.prediction_end_local,
            predictions,
            active_row,
        )
        if dry_run or force or not report_exists(conn, radar_key):
            for msg in radar_blocks:
                send_telegram_message(msg, dry_run=dry_run)
            if not dry_run:
                save_report(conn, radar_key, window.mode, "radar", window.prediction_start_local.date(), window.prediction_end_local.date(), dry_run, "\n---\n".join(radar_blocks))
        else:
            print(f"Radar déjà envoyé : {radar_key}")


def status_report(dry_run: bool = False) -> None:
    with get_conn() as conn:
        init_db(conn)
        active = fetch_latest_model_metrics(conn)
        completed = fetch_all_completed_fixtures(conn)
        stats = fetch_all_team_stats(conn)
        msg = ["🩺 XGenius IA — statut"]
        msg.append(f"Matchs terminés en base : {len(completed)}")
        msg.append(f"Lignes stats en base : {len(stats)}")
        if active:
            msg.append(f"Modèle actif : {active.get('model_version')}")
            msg.append(f"Samples : {active.get('n_samples')}")
            msg.append(f"Metrics : {active.get('metrics')}")
        else:
            msg.append("Aucun modèle actif pour l'instant. Baseline dynamique utilisée.")
        send_telegram_message("\n".join(msg), dry_run=dry_run)
