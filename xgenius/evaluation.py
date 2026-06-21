# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List
import numpy as np


def evaluate_predictions(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    evaluated = []
    for r in rows:
        if r.get("home_goals") is None or r.get("away_goals") is None:
            continue
        if r.get("status_short") not in {"FT", "AET", "PEN"}:
            continue
        evaluated.append(r)
    if not evaluated:
        return {
            "n": 0,
            "accuracy_1x2": None,
            "accuracy_over25": None,
            "accuracy_btts": None,
            "mae_total_goals": None,
            "rows": [],
        }

    ok_1x2 = ok_over = ok_btts = 0
    abs_err_total = []
    details = []
    for r in evaluated:
        hg = int(r["home_goals"])
        ag = int(r["away_goals"])
        true_result = "H" if hg > ag else "A" if ag > hg else "D"
        probs = {"H": float(r["p_home"]), "D": float(r["p_draw"]), "A": float(r["p_away"])}
        pred_result = max(probs, key=probs.get)
        pred_over = float(r["p_over25"]) >= 0.5
        true_over = (hg + ag) > 2.5
        pred_btts = float(r["p_btts"]) >= 0.5
        true_btts = hg > 0 and ag > 0
        ok_1x2 += int(pred_result == true_result)
        ok_over += int(pred_over == true_over)
        ok_btts += int(pred_btts == true_btts)
        pred_total = float(r["lambda_home"]) + float(r["lambda_away"])
        abs_err_total.append(abs(pred_total - (hg + ag)))
        details.append({
            "match": f"{r['home_team_name']} – {r['away_team_name']}",
            "score": f"{hg}-{ag}",
            "pred_result": pred_result,
            "true_result": true_result,
            "over_ok": pred_over == true_over,
            "btts_ok": pred_btts == true_btts,
            "model_version": r.get("model_version"),
        })

    n = len(evaluated)
    return {
        "n": n,
        "accuracy_1x2": round(ok_1x2 / n, 3),
        "accuracy_over25": round(ok_over / n, 3),
        "accuracy_btts": round(ok_btts / n, 3),
        "mae_total_goals": round(float(np.mean(abs_err_total)), 3),
        "rows": details,
    }
