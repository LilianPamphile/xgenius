# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from .config import LOCAL_TZ, MAX_RADAR_TOP_MATCHES, MAX_FULL_LIST_MATCHES, SHOW_FULL_LIST
from .time_windows import display_date_range


def pct(x: Any) -> str:
    try:
        return f"{int(round(float(x) * 100))}%"
    except Exception:
        return "-"


def fmt_dt(dt) -> str:
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
    return dt.astimezone(LOCAL_TZ).strftime("%a %d/%m %H:%M")


def build_evaluation_report(mode: str, label: str, start_local, end_local, evaluation: Dict[str, Any], model_info: Optional[Dict[str, Any]], train_summary: Optional[Dict[str, Any]]) -> str:
    lines = []
    lines.append(f"📊 XGenius IA — {label}")
    lines.append(display_date_range(start_local, end_local))
    lines.append("")
    if evaluation["n"] == 0:
        lines.append("Aucune prédiction terminée à évaluer sur cette période.")
    else:
        lines.append(f"Matchs évalués : {evaluation['n']}")
        lines.append(f"1X2 correct : {pct(evaluation['accuracy_1x2'])}")
        lines.append(f"Over 2.5 correct : {pct(evaluation['accuracy_over25'])}")
        lines.append(f"BTTS correct : {pct(evaluation['accuracy_btts'])}")
        lines.append(f"Erreur moyenne buts : {evaluation['mae_total_goals']}")
        good = [r for r in evaluation.get("rows", []) if r["pred_result"] == r["true_result"]][:3]
        if good:
            lines.append("")
            lines.append("✅ Exemples validés")
            for r in good:
                lines.append(f"• {r['match']} : {r['score']}")
    lines.append("")
    if train_summary:
        lines.append("🤖 Apprentissage")
        lines.append(train_summary.get("message", "Entraînement non effectué."))
        if train_summary.get("metrics"):
            m = train_summary["metrics"]
            lines.append(f"MAE total validation : {m.get('mae_total')}")
            lines.append(f"Accuracy 1X2 validation : {pct(m.get('accuracy_1x2'))}")
            lines.append(f"Matchs appris : {train_summary.get('n_samples')}")
    elif model_info:
        metrics = model_info.get("metrics") or {}
        lines.append("🤖 Modèle actif")
        lines.append(str(model_info.get("model_version", "modèle inconnu"))[:80])
        if metrics:
            lines.append(f"MAE total validation : {metrics.get('mae_total')}")
            lines.append(f"Matchs appris : {model_info.get('n_samples')}")
    return "\n".join(lines).strip()


def sort_for_top(preds: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    return sorted(preds, key=lambda r: float(r.get(key, 0) or 0), reverse=True)


def format_match_line(r: Dict[str, Any], compact: bool = False) -> str:
    head = f"{fmt_dt(r['kickoff_utc'])} — {r['home_team_name']} – {r['away_team_name']}"
    probs = f"1X2 {pct(r['p_home'])}/{pct(r['p_draw'])}/{pct(r['p_away'])}"
    buts = f"Buts {float(r['lambda_home']):.1f}-{float(r['lambda_away']):.1f} | +2,5 {pct(r['p_over25'])} | BTTS {pct(r['p_btts'])}"
    sig = f"Signal : {r['signal']} | Conf. {float(r['confidence']):.0f}/100 | {r['profile']}"
    if compact:
        return f"• {head}\n  {probs} | +2,5 {pct(r['p_over25'])} | BTTS {pct(r['p_btts'])} | {r['signal']}"
    return f"{head}\n{probs}\n{buts}\n{sig}"


def build_radar_report(label: str, start_local, end_local, predictions: List[Dict[str, Any]], model_info: Optional[Dict[str, Any]]) -> List[str]:
    preds = predictions[:]
    title = f"🤖 XGenius IA — {label}"
    header = [title, display_date_range(start_local, end_local), f"{len(preds)} matchs analysés."]
    if model_info:
        header.append(f"Modèle : {str(model_info.get('model_version', 'baseline'))[:32]}")
    else:
        header.append("Modèle : baseline dynamique, en attente d'historique suffisant")
    header.append("")

    blocks = ["\n".join(header).strip()]
    if not preds:
        blocks[0] += "\nAucun match détecté sur la période."
        return blocks

    categories = [
        ("🔥 Potentiel offensif", sort_for_top(preds, "p_over25")[:MAX_RADAR_TOP_MATCHES]),
        ("🤝 BTTS à surveiller", sort_for_top(preds, "p_btts")[:MAX_RADAR_TOP_MATCHES]),
        ("🎯 Signal 1X2 le plus net", sorted(preds, key=lambda r: max(float(r["p_home"]), float(r["p_draw"]), float(r["p_away"])), reverse=True)[:MAX_RADAR_TOP_MATCHES]),
        ("🧊 Matchs potentiellement fermés", sorted(preds, key=lambda r: float(r.get("p_over25", 0)))[:MAX_RADAR_TOP_MATCHES]),
    ]
    for title, rows in categories:
        lines = [title]
        for r in rows[:3]:
            lines.append(format_match_line(r, compact=True))
        blocks.append("\n".join(lines))

    if SHOW_FULL_LIST:
        sorted_all = sorted(preds, key=lambda r: r["kickoff_utc"])[:MAX_FULL_LIST_MATCHES]
        lines = [f"📋 Tous les matchs analysés ({len(sorted_all)}/{len(preds)})"]
        for r in sorted_all:
            lines.append(format_match_line(r, compact=True))
        # découpage simple tous les 25 matchs pour Telegram
        chunk, count = [], 0
        for line in lines:
            chunk.append(line)
            if line.startswith("• "):
                count += 1
            if count >= 25:
                blocks.append("\n".join(chunk))
                chunk, count = ["📋 Suite des matchs analysés"], 0
        if len(chunk) > 1:
            blocks.append("\n".join(chunk))
    return blocks
