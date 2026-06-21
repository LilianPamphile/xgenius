# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple
import math
import uuid

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, brier_score_loss

from .config import MODEL_FAMILY, RANDOM_STATE, MIN_TRAIN_MATCHES, MIN_VALIDATION_MATCHES, MODEL_ACTIVATION_TOLERANCE
from .features import FEATURE_COLUMNS, build_training_dataset, features_to_matrix

MAX_GOALS = 10


def poisson_pmf(lam: float, k: int) -> float:
    lam = float(np.clip(lam, 0.05, 8.0))
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def derive_probabilities(lambda_home: float, lambda_away: float) -> Dict[str, float]:
    lh = float(np.clip(lambda_home, 0.05, 6.0))
    la = float(np.clip(lambda_away, 0.05, 6.0))
    p_home = p_draw = p_away = p_over25 = p_btts = 0.0
    for hg in range(MAX_GOALS + 1):
        ph = poisson_pmf(lh, hg)
        for ag in range(MAX_GOALS + 1):
            pa = poisson_pmf(la, ag)
            p = ph * pa
            if hg > ag:
                p_home += p
            elif hg == ag:
                p_draw += p
            else:
                p_away += p
            if hg + ag > 2.5:
                p_over25 += p
            if hg > 0 and ag > 0:
                p_btts += p
    total_1x2 = p_home + p_draw + p_away
    if total_1x2 > 0:
        p_home, p_draw, p_away = p_home / total_1x2, p_draw / total_1x2, p_away / total_1x2
    return {
        "p_home": round(float(p_home), 4),
        "p_draw": round(float(p_draw), 4),
        "p_away": round(float(p_away), 4),
        "p_over25": round(float(p_over25), 4),
        "p_btts": round(float(p_btts), 4),
    }


def classify_profile(lambda_home: float, lambda_away: float, probs: Dict[str, float]) -> str:
    total = lambda_home + lambda_away
    top = max(probs["p_home"], probs["p_draw"], probs["p_away"])
    spread = top - min(probs["p_home"], probs["p_draw"], probs["p_away"])
    if probs["p_over25"] >= 0.62 and probs["p_btts"] >= 0.55:
        return "ouvert"
    if total <= 2.10 and probs["p_over25"] <= 0.42:
        return "fermé"
    if spread >= 0.30:
        return "déséquilibré"
    if top <= 0.42:
        return "indécis"
    return "équilibré"


def main_signal(probs: Dict[str, float], home_name: str, away_name: str) -> str:
    labels = [(probs["p_home"], home_name), (probs["p_draw"], "Match nul"), (probs["p_away"], away_name)]
    labels.sort(reverse=True, key=lambda x: x[0])
    top_p, top_label = labels[0]
    if top_label != "Match nul" and top_p < 0.48:
        if labels[1][1] == "Match nul":
            return f"{top_label} ou N"
        return f"{top_label} ou {labels[1][1]}"
    return f"{top_label} ({int(round(top_p * 100))}%)"


def confidence_score(probs: Dict[str, float], features: Dict[str, float], model_kind: str) -> float:
    top = max(probs["p_home"], probs["p_draw"], probs["p_away"])
    margin = top - sorted([probs["p_home"], probs["p_draw"], probs["p_away"]])[-2]
    samples = min(1.0, (features.get("home_samples_5", 0) + features.get("away_samples_5", 0)) / 10.0)
    league_samples = min(1.0, features.get("league_samples_60", 0) / 60.0)
    model_bonus = 0.12 if model_kind == "ml" else 0.0
    score = 100 * (0.30 + 0.35 * margin + 0.15 * samples + 0.08 * league_samples + model_bonus)
    return round(float(np.clip(score, 20, 92)), 1)


def baseline_predict(features: Dict[str, float]) -> Tuple[float, float]:
    lh = float(features.get("expected_base_home", 1.25) or 1.25)
    la = float(features.get("expected_base_away", 1.10) or 1.10)
    return float(np.clip(lh, 0.05, 5.5)), float(np.clip(la, 0.05, 5.5))


@dataclass
class ModelBundle:
    model_version: str
    trained_at: str
    features: List[str]
    home_model: Any
    away_model: Any
    metrics: Dict[str, Any]
    n_samples: int

    def predict_lambdas(self, feature: Dict[str, float]) -> Tuple[float, float]:
        X = features_to_matrix([feature])
        lh = float(self.home_model.predict(X)[0])
        la = float(self.away_model.predict(X)[0])
        # Lissage avec baseline pour éviter les sorties absurdes sur équipes sans historique.
        bh, ba = baseline_predict(feature)
        sample_factor = min(1.0, (feature.get("home_samples_5", 0) + feature.get("away_samples_5", 0)) / 10.0)
        lh = 0.75 * lh + 0.25 * bh if sample_factor >= 0.6 else 0.45 * lh + 0.55 * bh
        la = 0.75 * la + 0.25 * ba if sample_factor >= 0.6 else 0.45 * la + 0.55 * ba
        return float(np.clip(lh, 0.05, 5.5)), float(np.clip(la, 0.05, 5.5))


def serialize_bundle(bundle: ModelBundle) -> bytes:
    bio = BytesIO()
    joblib.dump(bundle, bio)
    return bio.getvalue()


def deserialize_bundle(data: bytes) -> ModelBundle:
    return joblib.load(BytesIO(data))


def _build_regressor() -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=220,
        max_depth=9,
        min_samples_leaf=3,
        max_features=0.85,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def _prediction_metrics(yh_true: np.ndarray, ya_true: np.ndarray, lh: np.ndarray, la: np.ndarray) -> Dict[str, float]:
    total_true = yh_true + ya_true
    total_pred = lh + la
    mae_home = float(mean_absolute_error(yh_true, lh))
    mae_away = float(mean_absolute_error(ya_true, la))
    mae_total = float(mean_absolute_error(total_true, total_pred))
    rmse_total = float(np.sqrt(mean_squared_error(total_true, total_pred)))

    pred_result = np.where(lh > la + 0.05, 1, np.where(la > lh + 0.05, 2, 0))
    true_result = np.where(yh_true > ya_true, 1, np.where(ya_true > yh_true, 2, 0))
    acc_1x2 = float(accuracy_score(true_result, pred_result))

    p_over = []
    p_btts = []
    for a, b in zip(lh, la):
        p = derive_probabilities(float(a), float(b))
        p_over.append(p["p_over25"])
        p_btts.append(p["p_btts"])
    over_true = (total_true > 2.5).astype(int)
    btts_true = ((yh_true > 0) & (ya_true > 0)).astype(int)
    over_pred = (np.array(p_over) >= 0.5).astype(int)
    btts_pred = (np.array(p_btts) >= 0.5).astype(int)

    return {
        "mae_home": round(mae_home, 4),
        "mae_away": round(mae_away, 4),
        "mae_total": round(mae_total, 4),
        "rmse_total": round(rmse_total, 4),
        "accuracy_1x2": round(float(accuracy_score(true_result, pred_result)), 4),
        "accuracy_over25": round(float(accuracy_score(over_true, over_pred)), 4),
        "accuracy_btts": round(float(accuracy_score(btts_true, btts_pred)), 4),
        "n_validation": int(len(yh_true)),
    }


def train_candidate(fixtures: List[Dict[str, Any]], stats: List[Dict[str, Any]]) -> Optional[ModelBundle]:
    meta, X, y_home, y_away = build_training_dataset(fixtures, stats)
    n = len(y_home)
    if n < MIN_TRAIN_MATCHES:
        return None

    val_n = max(MIN_VALIDATION_MATCHES, int(n * 0.20))
    val_n = min(val_n, max(1, n // 3))
    train_n = n - val_n
    if train_n < 50:
        return None

    X_train, X_val = X[:train_n], X[train_n:]
    yh_train, yh_val = y_home[:train_n], y_home[train_n:]
    ya_train, ya_val = y_away[:train_n], y_away[train_n:]

    home_model = _build_regressor()
    away_model = _build_regressor()
    home_model.fit(X_train, yh_train)
    away_model.fit(X_train, ya_train)

    lh_val = np.clip(home_model.predict(X_val), 0.05, 5.5)
    la_val = np.clip(away_model.predict(X_val), 0.05, 5.5)
    metrics = _prediction_metrics(yh_val, ya_val, lh_val, la_val)

    # Entraînement final sur tout l'historique connu.
    final_home = _build_regressor()
    final_away = _build_regressor()
    final_home.fit(X, y_home)
    final_away.fit(X, y_away)

    version = f"{MODEL_FAMILY}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    return ModelBundle(
        model_version=version,
        trained_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        features=FEATURE_COLUMNS,
        home_model=final_home,
        away_model=final_away,
        metrics=metrics,
        n_samples=n,
    )


def should_activate_candidate(candidate: ModelBundle, active_metrics: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
    if not active_metrics:
        return True, "aucun modèle actif"
    old = active_metrics.get("metrics") or {}
    if hasattr(old, "copy"):
        old_mae = old.get("mae_total")
    else:
        old_mae = None
    new_mae = candidate.metrics.get("mae_total")
    if old_mae is None or new_mae is None:
        return True, "métriques précédentes indisponibles"
    # Activation si meilleur, ou quasiment équivalent avec plus de données.
    if new_mae <= float(old_mae) + MODEL_ACTIVATION_TOLERANCE:
        return True, f"nouveau MAE {new_mae} <= ancien MAE {old_mae} + tolérance"
    return False, f"nouveau modèle moins bon ({new_mae} > {old_mae})"


def prediction_from_feature(feature: Dict[str, float], home_name: str, away_name: str, bundle: Optional[ModelBundle]) -> Dict[str, Any]:
    if bundle is None:
        lh, la = baseline_predict(feature)
        model_version = "baseline_dynamic"
        model_kind = "baseline"
    else:
        lh, la = bundle.predict_lambdas(feature)
        model_version = bundle.model_version
        model_kind = "ml"
    probs = derive_probabilities(lh, la)
    profile = classify_profile(lh, la, probs)
    signal = main_signal(probs, home_name, away_name)
    conf = confidence_score(probs, feature, model_kind)
    return {
        "model_version": model_version,
        "model_kind": model_kind,
        "lambda_home": round(lh, 3),
        "lambda_away": round(la, 3),
        "p_home": probs["p_home"],
        "p_draw": probs["p_draw"],
        "p_away": probs["p_away"],
        "p_over25": probs["p_over25"],
        "p_btts": probs["p_btts"],
        "confidence": conf,
        "profile": profile,
        "signal": signal,
        "features": feature,
    }
