# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo
from typing import Optional

from .config import LOCAL_TZ, UTC_TZ, MONDAY_RUN_WEEKDAY, THURSDAY_RUN_WEEKDAY, RunWindow, SCHEDULED_LOCAL_HOUR, SCHEDULED_LOCAL_MINUTE


def start_of_day_local(dt: datetime) -> datetime:
    local = dt.astimezone(LOCAL_TZ)
    return datetime.combine(local.date(), time.min, tzinfo=LOCAL_TZ)


def next_weekday_start(now: datetime, weekday: int) -> datetime:
    base = start_of_day_local(now)
    delta = (weekday - base.weekday()) % 7
    return base + timedelta(days=delta)


def previous_weekday_start(now: datetime, weekday: int) -> datetime:
    base = start_of_day_local(now)
    delta = (base.weekday() - weekday) % 7
    return base - timedelta(days=delta)


def resolve_mode(mode: str, now: Optional[datetime] = None) -> str:
    now = now or datetime.now(LOCAL_TZ)
    local = now.astimezone(LOCAL_TZ)
    if mode != "auto":
        return mode
    if local.weekday() == MONDAY_RUN_WEEKDAY:
        return "monday"
    if local.weekday() == THURSDAY_RUN_WEEKDAY:
        return "thursday"
    return "none"


def should_run_scheduled(now: Optional[datetime] = None) -> bool:
    now = now or datetime.now(LOCAL_TZ)
    local = now.astimezone(LOCAL_TZ)
    return (
        local.weekday() in {MONDAY_RUN_WEEKDAY, THURSDAY_RUN_WEEKDAY}
        and local.hour == SCHEDULED_LOCAL_HOUR
        and abs(local.minute - SCHEDULED_LOCAL_MINUTE) <= 10
    )


def build_run_window(mode: str, now: Optional[datetime] = None) -> RunWindow:
    now = now or datetime.now(LOCAL_TZ)
    mode = resolve_mode(mode, now)
    if mode not in {"monday", "thursday"}:
        raise ValueError(f"Mode invalide pour build_run_window: {mode}")

    if mode == "monday":
        monday = next_weekday_start(now, MONDAY_RUN_WEEKDAY)
        # Si on est mardi/mercredi avec exécution manuelle monday, on garde le lundi courant passé.
        if now.astimezone(LOCAL_TZ).weekday() in {1, 2}:
            monday = previous_weekday_start(now, MONDAY_RUN_WEEKDAY)
        friday_prev = monday - timedelta(days=3)
        thursday = monday + timedelta(days=3)
        return RunWindow(
            mode="monday",
            evaluation_label="bilan du week-end",
            evaluation_start_local=friday_prev,
            evaluation_end_local=monday,
            prediction_label="radar lundi → mercredi",
            prediction_start_local=monday,
            prediction_end_local=thursday,
        )

    thursday = next_weekday_start(now, THURSDAY_RUN_WEEKDAY)
    # Si on est vendredi/samedi/dimanche avec exécution manuelle thursday, on garde le jeudi courant passé.
    if now.astimezone(LOCAL_TZ).weekday() in {4, 5, 6}:
        thursday = previous_weekday_start(now, THURSDAY_RUN_WEEKDAY)
    monday = thursday - timedelta(days=3)
    next_monday = thursday + timedelta(days=4)
    return RunWindow(
        mode="thursday",
        evaluation_label="bilan lundi → mercredi",
        evaluation_start_local=monday,
        evaluation_end_local=thursday,
        prediction_label="radar jeudi → dimanche",
        prediction_start_local=thursday,
        prediction_end_local=next_monday,
    )


def to_utc(dt: datetime) -> datetime:
    return dt.astimezone(UTC_TZ)


def display_date_range(start_local: datetime, end_local_exclusive: datetime) -> str:
    end_inclusive = end_local_exclusive - timedelta(days=1)
    return f"du {start_local.strftime('%d/%m')} au {end_inclusive.strftime('%d/%m')}"
