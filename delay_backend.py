from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "flights_weather_sample.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "saved_model" / "delay_risk_logistic_regression_pipeline.joblib"
DEFAULT_MODEL_INFO_PATH = PROJECT_ROOT / "saved_model" / "model_info.json"


def risk_band(probability):
    if probability >= 0.65:
        return "High"
    if probability >= 0.35:
        return "Medium"
    return "Low"


def _parse_schedule(df):
    df["sched_dep_dt"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["sched_dep_local"].astype(str),
        errors="coerce",
    )
    df["scheduled_arr_dt"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["scheduled_arr_local"].astype(str),
        errors="coerce",
    )

    overnight_mask = df["scheduled_arr_dt"] < df["sched_dep_dt"]
    df.loc[overnight_mask, "scheduled_arr_dt"] += pd.Timedelta(days=1)
    return df


def engineer_features(raw_df):
    df = raw_df.copy()
    df = _parse_schedule(df)

    df = df.sort_values(["tail_number", "sched_dep_dt"]).reset_index(drop=True)

    tail_group = df.groupby("tail_number")
    df["prev_flight_id"] = tail_group["flight_id"].shift(1)
    df["prev_origin"] = tail_group["origin"].shift(1)
    df["prev_dest"] = tail_group["dest"].shift(1)
    df["prev_sched_dep_dt"] = tail_group["sched_dep_dt"].shift(1)
    df["prev_scheduled_arr_dt"] = tail_group["scheduled_arr_dt"].shift(1)
    df["prev_arr_delay_min"] = tail_group["arr_delay_min"].shift(1)
    df["prev_dep_delay_min"] = tail_group["dep_delay_min"].shift(1)

    df["prev_dest_equals_current_origin"] = (df["prev_dest"] == df["origin"]).astype(int)
    df["turnaround_buffer_min"] = (
        (df["sched_dep_dt"] - df["prev_scheduled_arr_dt"]).dt.total_seconds() / 60
    )
    df.loc[df["prev_dest_equals_current_origin"] == 0, "turnaround_buffer_min"] = np.nan

    df["prev_arr_delay_min_filled"] = df["prev_arr_delay_min"].fillna(0)
    df["prev_dep_delay_min_filled"] = df["prev_dep_delay_min"].fillna(0)
    df["prev_arr_delayed_15"] = (df["prev_arr_delay_min_filled"] >= 15).astype(int)
    df["prev_arr_heavily_delayed_30"] = (df["prev_arr_delay_min_filled"] >= 30).astype(int)
    df["recovery_gap_min"] = df["turnaround_buffer_min"] - df["prev_arr_delay_min_filled"]

    df["snow_flag"] = (df["snowfall_cm"] > 0).astype(int)
    df["rain_or_precip_flag"] = (df["precip_mm"] > 0).astype(int)
    df["high_wind_flag"] = (df["wind_gust_kmh"] >= 40).astype(int)
    df["fog_flag"] = df["weather_code"].isin([45, 48]).astype(int)
    df["snow_weather_code_flag"] = df["weather_code"].between(71, 77).astype(int)
    df["rain_weather_code_flag"] = df["weather_code"].between(51, 67).astype(int)
    df["storm_flag"] = (df["weather_code"] == 95).astype(int)

    df["weather_severity_score"] = (
        15 * df["snow_flag"]
        + 10 * df["rain_or_precip_flag"]
        + 10 * df["high_wind_flag"]
        + 10 * df["fog_flag"]
        + 15 * df["snow_weather_code_flag"]
        + 10 * df["rain_weather_code_flag"]
        + 20 * df["storm_flag"]
    )

    df["morning_bank_flag"] = df["dep_hour"].isin([6, 7, 8, 9]).astype(int)
    df["evening_bank_flag"] = df["dep_hour"].isin([16, 17, 18, 19, 20]).astype(int)
    df["origin_hourly_flight_count"] = (
        df.groupby(["origin", "date", "dep_hour"])["flight_id"].transform("count")
    )

    return df.sort_values("sched_dep_dt").reset_index(drop=True)


def explain_flight(row):
    reasons = []

    if row["prev_arr_delay_min_filled"] >= 30:
        reasons.append(
            f"Strong cascade risk: previous aircraft arrival delay was {row['prev_arr_delay_min_filled']:.0f} min."
        )
    elif row["prev_arr_delay_min_filled"] >= 15:
        reasons.append(
            f"Moderate cascade risk: previous aircraft arrival delay was {row['prev_arr_delay_min_filled']:.0f} min."
        )

    if pd.notna(row["turnaround_buffer_min"]):
        if row["turnaround_buffer_min"] < 30:
            reasons.append(
                f"Short turnaround buffer: only {row['turnaround_buffer_min']:.0f} min available."
            )
        if pd.notna(row["recovery_gap_min"]) and row["recovery_gap_min"] < 0:
            reasons.append(
                f"Negative recovery gap: previous delay exceeds turnaround buffer by {abs(row['recovery_gap_min']):.0f} min."
            )

    if row["snowfall_cm"] > 0:
        reasons.append(f"Snow at origin: {row['snowfall_cm']:.2f} cm during departure hour.")
    if row["precip_mm"] > 0:
        reasons.append(f"Precipitation at origin: {row['precip_mm']:.2f} mm.")
    if row["wind_gust_kmh"] >= 40:
        reasons.append(f"High wind gusts: {row['wind_gust_kmh']:.0f} km/h.")
    if row["fog_flag"] == 1:
        reasons.append("Fog condition indicated by weather code.")

    if row["evening_bank_flag"] == 1:
        reasons.append("Departure is during evening congestion-prone bank.")
    if row["morning_bank_flag"] == 1:
        reasons.append("Departure is during morning congestion-prone bank.")
    if row["origin_hourly_flight_count"] >= 40:
        reasons.append(
            f"High airport departure load: {row['origin_hourly_flight_count']} flights scheduled from {row['origin']} in this hour."
        )

    if not reasons:
        reasons.append(
            "No single strong driver found; risk may come from weaker combined schedule/weather factors."
        )

    return reasons


def main_reason_category(row):
    if row["prev_arr_delay_min_filled"] >= 15:
        return "Cascade / late inbound aircraft"

    if (
        row["snowfall_cm"] > 0
        or row["precip_mm"] > 0
        or row["wind_gust_kmh"] >= 40
        or row["fog_flag"] == 1
    ):
        return "Weather pressure"

    if (
        row["morning_bank_flag"] == 1
        or row["evening_bank_flag"] == 1
        or row["origin_hourly_flight_count"] >= 40
    ):
        return "Congestion / schedule pressure"

    return "General operational risk"


def recommend_action(row):
    reason = row["main_reason"]

    if reason == "Cascade / late inbound aircraft":
        return (
            "Monitor inbound tail, alert gate team, prepare fast-turnaround support, "
            "and check backup aircraft availability."
        )
    if reason == "Weather pressure":
        return (
            "Alert ground operations, monitor airport-wide weather impact, "
            "and prepare extra turnaround buffer."
        )
    if reason == "Congestion / schedule pressure":
        return (
            "Prioritize high-connection flights and coordinate pushback or gate sequencing "
            "with operations control."
        )
    return "Keep under monitoring; no immediate strong intervention trigger found."


def score_flights(
    data_path=DEFAULT_DATA_PATH,
    model_path=DEFAULT_MODEL_PATH,
    model_info_path=DEFAULT_MODEL_INFO_PATH,
    include_cancelled=False,
    schedule_date=None,
    sort_by="risk",
):
    data_path = Path(data_path)
    model_path = Path(model_path)
    model_info_path = Path(model_info_path)

    raw_df = pd.read_csv(data_path)
    model = joblib.load(model_path)
    with model_info_path.open("r", encoding="utf-8") as model_info_file:
        model_info = json.load(model_info_file)

    scored = engineer_features(raw_df)

    if schedule_date == "last":
        last_date = scored["sched_dep_dt"].dt.date.max()
        scored = scored[scored["sched_dep_dt"].dt.date == last_date].copy()
    elif schedule_date is not None:
        selected_date = pd.to_datetime(schedule_date).date()
        scored = scored[scored["sched_dep_dt"].dt.date == selected_date].copy()

    if "cancelled" in scored.columns and not include_cancelled:
        scored = scored[scored["cancelled"].fillna(0) == 0].copy()

    missing_features = [feature for feature in model_info["features"] if feature not in scored.columns]
    if missing_features:
        raise ValueError(f"Missing model feature columns: {', '.join(missing_features)}")

    feature_frame = scored[model_info["features"]].copy()
    scored["delay_probability"] = model.predict_proba(feature_frame)[:, 1]
    scored["predicted_delayed_15"] = model.predict(feature_frame)
    scored["risk_band"] = scored["delay_probability"].apply(risk_band)
    scored["main_reason"] = scored.apply(main_reason_category, axis=1)
    scored["explanation_list"] = scored.apply(explain_flight, axis=1)
    scored["explanation"] = scored["explanation_list"].apply(lambda reasons: " | ".join(reasons))
    scored["recommended_action"] = scored.apply(recommend_action, axis=1)

    if sort_by == "schedule":
        sort_columns = ["sched_dep_dt", "origin", "dest", "flight_number"]
        return scored.sort_values(sort_columns).reset_index(drop=True)

    return scored.sort_values("delay_probability", ascending=False).reset_index(drop=True)


def score_last_day_schedule(
    data_path=DEFAULT_DATA_PATH,
    model_path=DEFAULT_MODEL_PATH,
    model_info_path=DEFAULT_MODEL_INFO_PATH,
):
    return score_flights(
        data_path=data_path,
        model_path=model_path,
        model_info_path=model_info_path,
        include_cancelled=True,
        schedule_date="last",
        sort_by="schedule",
    )
