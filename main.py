# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from datetime import datetime

from xgenius.config import LOCAL_TZ
from xgenius.jobs import bootstrap_history, run_mode, status_report
from xgenius.time_windows import resolve_mode, should_run_scheduled


def str_to_bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "yes", "y", "on"}


def main() -> None:
    parser = argparse.ArgumentParser(description="XGenius IA auto-apprenante")
    parser.add_argument("--mode", choices=["auto", "monday", "thursday", "bootstrap", "status"], default="auto")
    parser.add_argument("--dry-run", default="false", help="true/false : affiche sans envoyer Telegram")
    parser.add_argument("--force", default="false", help="true/false : renvoie les rapports même s'ils existent déjà")
    parser.add_argument("--bootstrap-days", type=int, default=90, help="Nombre de jours à importer en mode bootstrap")
    args = parser.parse_args()

    dry_run = str_to_bool(args.dry_run)
    force = str_to_bool(args.force)
    now = datetime.now(LOCAL_TZ)

    if args.mode == "bootstrap":
        bootstrap_history(days=args.bootstrap_days, dry_run=dry_run)
        return

    if args.mode == "status":
        status_report(dry_run=dry_run)
        return

    mode = resolve_mode(args.mode, now)
    if mode == "none":
        print(f"Aucune exécution prévue aujourd'hui ({now.isoformat()}).")
        return

    # En mode auto déclenché par cron, on bloque hors créneau exact pour gérer heure été/hiver.
    if args.mode == "auto" and not should_run_scheduled(now):
        print(f"Cron ignoré : heure locale actuelle {now.isoformat()} hors créneau XGenius.")
        return

    run_mode(mode=mode, dry_run=dry_run, force=force)


if __name__ == "__main__":
    main()
