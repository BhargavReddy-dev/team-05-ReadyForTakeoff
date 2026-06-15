#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from ready_for_takeoff.delay_model import load_flights, load_model, score_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score flights with the trained delay-risk logistic regression model.")
    parser.add_argument("--model", default="models/delay_logreg.joblib", help="Saved model bundle from train_model.py.")
    parser.add_argument("--input", default="data/flights_weather_sample.csv", help="Flights/weather CSV to score.")
    parser.add_argument("--output", default="models/scored_flights.csv", help="Output CSV with risk, reasons, and actions.")
    parser.add_argument("--include-actuals", action="store_true", help="Include answer-key columns if they exist, for validation only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_model(args.model)
    df = load_flights(args.input)
    scored = score_dataframe(
        df,
        bundle["pipeline"],
        threshold=bundle.get("threshold", 0.5),
        include_actuals=args.include_actuals,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output, index=False)
    print(f"Saved scored flights: {args.output}")


if __name__ == "__main__":
    main()

