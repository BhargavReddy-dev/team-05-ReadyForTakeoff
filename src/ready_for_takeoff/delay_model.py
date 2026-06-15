from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "delayed_15"

LEAKAGE_COLUMNS = {
    "dep_delay_min",
    "arr_delay_min",
    "delayed_15",
    "cancelled",
    "delay_cause",
    "late_aircraft_delay_min",
    "weather_delay_min",
}

NUMERIC_FEATURES = [
    "day_of_week",
    "dep_hour",
    "distance_km",
    "temp_c",
    "wind_speed_kmh",
    "wind_gust_kmh",
    "precip_mm",
    "snowfall_cm",
    "cloud_cover_pct",
    "weather_code",
    "inbound_arr_delay_min",
    "inbound_missing",
    "is_early_bank",
    "is_evening_bank",
]

CATEGORICAL_FEATURES = [
    "carrier",
    "origin",
    "dest",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

IDENTITY_COLUMNS = [
    "flight_id",
    "date",
    "carrier",
    "flight_number",
    "tail_number",
    "origin",
    "dest",
    "sched_dep_local",
]


@dataclass(frozen=True)
class TrainResult:
    pipeline: Pipeline
    metrics: dict
    predictions: pd.DataFrame


def load_flights(path: str | Path) -> pd.DataFrame:
    """Load the starter CSV with stable dtypes for identifiers."""
    return pd.read_csv(
        path,
        dtype={
            "flight_id": "string",
            "carrier": "string",
            "tail_number": "string",
            "origin": "string",
            "dest": "string",
            "sched_dep_local": "string",
            "scheduled_arr_local": "string",
        },
    )


def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build only pre-flight, weather, and permitted cascade features."""
    out = df.copy()
    scheduled_at = pd.to_datetime(
        out["date"].astype(str) + " " + out["sched_dep_local"].astype(str),
        errors="coerce",
    )
    out["_scheduled_at"] = scheduled_at
    out["_original_order"] = np.arange(len(out))

    if "inbound_arr_delay_min" not in out.columns:
        if "arr_delay_min" in out.columns:
            sortable = out.sort_values(["tail_number", "_scheduled_at", "_original_order"])
            inbound = sortable.groupby("tail_number", dropna=False)["arr_delay_min"].shift(1)
            out.loc[sortable.index, "inbound_arr_delay_min"] = inbound
        else:
            out["inbound_arr_delay_min"] = np.nan

    out["inbound_missing"] = out["inbound_arr_delay_min"].isna().astype(int)
    out["inbound_arr_delay_min"] = out["inbound_arr_delay_min"].fillna(0).clip(lower=0)

    out["is_early_bank"] = out["dep_hour"].between(5, 8).astype(int)
    out["is_evening_bank"] = out["dep_hour"].between(16, 21).astype(int)
    return out.drop(columns=["_scheduled_at", "_original_order"])


def modeling_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return leak-free features and binary target, dropping cancelled/missing labels."""
    missing = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    usable = df[df[TARGET_COLUMN].notna()].copy()
    if "cancelled" in usable.columns:
        usable = usable[usable["cancelled"].fillna(0).astype(int) == 0]

    overlap = sorted(set(FEATURE_COLUMNS).intersection(LEAKAGE_COLUMNS))
    if overlap:
        raise ValueError(f"Feature list contains leakage columns: {overlap}")

    X = usable[FEATURE_COLUMNS]
    y = usable[TARGET_COLUMN].astype(int)
    return X, y


def split_train_test(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Use the last days as validation to mimic forward-looking operations."""
    labeled = df[df[TARGET_COLUMN].notna()].copy()
    if "cancelled" in labeled.columns:
        labeled = labeled[labeled["cancelled"].fillna(0).astype(int) == 0]

    dates = pd.Series(pd.to_datetime(labeled["date"], errors="coerce")).dropna().sort_values().unique()
    if len(dates) < 2:
        rng = np.random.default_rng(42)
        mask = rng.random(len(labeled)) < 0.8
        return labeled.index[mask].to_numpy(), labeled.index[~mask].to_numpy()

    holdout_days = max(1, int(np.ceil(len(dates) * 0.2)))
    cutoff = dates[-holdout_days]
    train_idx = labeled.index[pd.to_datetime(labeled["date"], errors="coerce") < cutoff].to_numpy()
    test_idx = labeled.index[pd.to_datetime(labeled["date"], errors="coerce") >= cutoff].to_numpy()
    if len(train_idx) == 0 or len(test_idx) == 0:
        split_at = int(len(labeled) * 0.8)
        ordered = labeled.sort_values("date").index.to_numpy()
        return ordered[:split_at], ordered[split_at:]
    return train_idx, test_idx


def make_pipeline() -> Pipeline:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, NUMERIC_FEATURES),
            ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )
    classifier = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def train_logistic_regression(df: pd.DataFrame, threshold: float = 0.5) -> TrainResult:
    featured = add_model_features(df)
    train_idx, test_idx = split_train_test(featured)
    train_df = featured.loc[train_idx]
    test_df = featured.loc[test_idx]

    X_train, y_train = modeling_frame(train_df)
    X_test, y_test = modeling_frame(test_df)

    pipeline = make_pipeline()
    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predicted = (probabilities >= threshold).astype(int)

    metrics = classification_metrics(y_test, probabilities, predicted, threshold)
    metrics.update(
        {
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "features": FEATURE_COLUMNS,
            "leakage_columns_excluded": sorted(LEAKAGE_COLUMNS),
            "validation_strategy": "forward split by date; last 20% of days held out",
        }
    )

    predictions = score_dataframe(test_df, pipeline, threshold, include_actuals=True)
    return TrainResult(pipeline=pipeline, metrics=metrics, predictions=predictions)


def classification_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    predicted: np.ndarray,
    threshold: float,
) -> dict:
    cm = confusion_matrix(y_true, predicted, labels=[0, 1])
    metrics = {
        "threshold": threshold,
        "base_delay_rate": float(np.mean(y_true)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
    }
    if len(set(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))
        metrics["average_precision"] = float(average_precision_score(y_true, probabilities))
    return metrics


def score_dataframe(
    df: pd.DataFrame,
    pipeline: Pipeline,
    threshold: float = 0.5,
    include_actuals: bool = False,
) -> pd.DataFrame:
    featured = add_model_features(df)
    X = featured[FEATURE_COLUMNS]
    probabilities = pipeline.predict_proba(X)[:, 1]
    predicted = (probabilities >= threshold).astype(int)
    explanations = explain_rows(pipeline, X, top_n=3)

    result = featured[[col for col in IDENTITY_COLUMNS if col in featured.columns]].copy()
    result["delay_probability"] = np.round(probabilities, 4)
    result["risk_band"] = [risk_band(p) for p in probabilities]
    result["predicted_delayed_15"] = predicted
    result["top_reason"] = [items[0]["reason"] if items else "No dominant risk signal" for items in explanations]
    result["reason_details"] = [format_reasons(items) for items in explanations]
    result["recommended_action"] = [
        recommend_action(items[0]["feature"] if items else "", prob)
        for items, prob in zip(explanations, probabilities)
    ]

    if include_actuals:
        for col in ["delayed_15", "dep_delay_min", "delay_cause"]:
            if col in featured.columns:
                result[f"actual_{col}"] = featured[col].values
    return result.sort_values("delay_probability", ascending=False)


def explain_rows(pipeline: Pipeline, X: pd.DataFrame, top_n: int = 3) -> list[list[dict]]:
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    transformed = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()
    contributions = transformed.multiply(classifier.coef_[0]) if hasattr(transformed, "multiply") else transformed * classifier.coef_[0]
    if hasattr(contributions, "tocsr"):
        contributions = contributions.tocsr()

    rows: list[list[dict]] = []
    for i in range(X.shape[0]):
        row_contrib = np.asarray(contributions[i].todense()).ravel() if hasattr(contributions[i], "todense") else np.asarray(contributions[i]).ravel()
        positive = np.where(row_contrib > 0)[0]
        ranked = positive[np.argsort(row_contrib[positive])[::-1]][:top_n]
        rows.append(
            [
                {
                    "feature": readable_feature(feature_names[j]),
                    "reason": reason_label(feature_names[j]),
                    "contribution": float(row_contrib[j]),
                }
                for j in ranked
            ]
        )
    return rows


def readable_feature(feature_name: str) -> str:
    name = feature_name.replace("numeric__", "").replace("categorical__", "")
    return name.replace("_", " ")


def reason_label(feature_name: str) -> str:
    name = feature_name.replace("numeric__", "").replace("categorical__", "")
    if "inbound_arr_delay_min" in name:
        return "Late inbound aircraft cascade"
    if any(key in name for key in ["wind", "precip", "snowfall", "weather_code", "cloud_cover", "temp_c"]):
        return "Weather pressure at origin"
    if "dep_hour" in name or "early_bank" in name or "evening_bank" in name:
        return "Congested departure bank"
    if name.startswith("origin_"):
        return f"Origin airport pattern: {name.split('_', 1)[1]}"
    if name.startswith("carrier_"):
        return f"Carrier operating pattern: {name.split('_', 1)[1]}"
    if name.startswith("dest_"):
        return f"Destination/network pattern: {name.split('_', 1)[1]}"
    if "distance_km" in name:
        return "Longer sector exposure"
    return readable_feature(feature_name).title()


def recommend_action(top_feature: str, probability: float) -> str:
    if probability < 0.35:
        return "Monitor only; no proactive intervention recommended."
    feature = top_feature.lower()
    if "inbound" in feature:
        return "Check inbound aircraft status, protect the turn, and consider an aircraft swap if the delay grows."
    if any(key in feature for key in ["wind", "precip", "snow", "weather", "cloud", "temp"]):
        return "Brief the hub desk on weather exposure and prepare ground handling or de-icing mitigation."
    if "hour" in feature or "bank" in feature or "origin" in feature:
        return "Coordinate gate, ramp, and ATC constraints before the departure bank peaks."
    if "carrier" in feature:
        return "Review carrier-controlled turnaround tasks and crew readiness."
    return "Review the flight with the duty controller and prepare a targeted recovery option."


def risk_band(probability: float) -> str:
    if probability >= 0.75:
        return "high"
    if probability >= 0.5:
        return "elevated"
    if probability >= 0.35:
        return "watch"
    return "low"


def format_reasons(items: Iterable[dict]) -> str:
    return "; ".join(f"{item['reason']} ({item['contribution']:.2f})" for item in items)


def save_model(path: str | Path, pipeline: Pipeline, threshold: float, metrics: dict) -> None:
    bundle = {
        "pipeline": pipeline,
        "threshold": threshold,
        "features": FEATURE_COLUMNS,
        "leakage_columns_excluded": sorted(LEAKAGE_COLUMNS),
        "metrics": metrics,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_model(path: str | Path) -> dict:
    return joblib.load(path)


def write_json(path: str | Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
