#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from ready_for_takeoff.delay_model import (
    load_flights,
    save_model,
    train_logistic_regression,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Ready for Takeoff logistic regression delay model.")
    parser.add_argument("--data", default="data/flights_weather_sample.csv", help="Input flights/weather CSV.")
    parser.add_argument("--model-out", default="models/delay_logreg.joblib", help="Path for the saved model bundle.")
    parser.add_argument("--metrics-out", default="models/metrics.json", help="Path for validation metrics JSON.")
    parser.add_argument("--predictions-out", default="models/validation_predictions.csv", help="Path for scored validation flights.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Delay probability threshold for delayed/not-delayed labels.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_flights(args.data)
    result = train_logistic_regression(df, threshold=args.threshold)

    save_model(args.model_out, result.pipeline, args.threshold, result.metrics)
    write_json(args.metrics_out, result.metrics)
    Path(args.predictions_out).parent.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(args.predictions_out, index=False)

    print(f"Saved model: {args.model_out}")
    print(f"Saved metrics: {args.metrics_out}")
    print(f"Saved validation predictions: {args.predictions_out}")
    print(
        "Validation: "
        f"ROC AUC={result.metrics.get('roc_auc', float('nan')):.3f}, "
        f"F1={result.metrics['f1']:.3f}, "
        f"precision={result.metrics['precision']:.3f}, "
        f"recall={result.metrics['recall']:.3f}"
    )


if __name__ == "__main__":
    main()

