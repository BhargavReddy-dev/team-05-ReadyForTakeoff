# team-05-ReadyForTakeoff
Git repository for team-05(Team Fox) on the challenge 'Ready for Takeoff' by DATAbility

Challenge    - DATAbility - Ready for Takeoff
               Build a delay risk tool to forecast flight delays, quantify the risk, 
Team no.     - 05
Team Name    - Team Fox
Team Members - Harshal Hisoriya
               Yash Thummar
               Tanay Patel
               Bhargava Reddy

## Logistic regression delay-risk model

This repo contains a leak-free logistic regression model for predicting whether a
flight is likely to depart 15+ minutes late. It follows the starter-kit rules:

- Target: `delayed_15`
- Allowed signals: schedule fields, route/airport fields, origin weather, and a
  derived inbound-aircraft cascade signal from `tail_number`
- Excluded answer-key columns: `dep_delay_min`, `arr_delay_min`, `delayed_15`,
  `cancelled`, `delay_cause`, `late_aircraft_delay_min`, and
  `weather_delay_min`
- Validation: forward split by date, holding out the last 20% of days

The model output is built for an operations controller: each scored flight gets a
delay probability, risk band, top reason signals, and a recommended action.

## Project layout

```text
data/flights_weather_sample.csv        Provided real flights plus weather sample
src/ready_for_takeoff/delay_model.py   Feature engineering, training, scoring
scripts/train_model.py                 Train and save the logistic regression model
scripts/predict_delay.py               Score flights with the saved model
models/metrics.json                    Validation metrics from the latest run
models/validation_predictions.csv      Held-out validation scores
models/scored_flights.csv              Full sample scores for demo use
tests/test_delay_model.py              Leakage and cascade feature tests
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If the dependencies are already installed, set `PYTHONPATH=src` and run the
commands directly.

## Train

```bash
PYTHONPATH=src python3 scripts/train_model.py
```

This writes:

- `models/delay_logreg.joblib`
- `models/metrics.json`
- `models/validation_predictions.csv`

Latest validation result on the provided sample:

```text
ROC AUC: 0.668
Average precision: 0.524
Accuracy: 0.654
Precision: 0.493
Recall: 0.493
F1: 0.493
```

## Score flights

```bash
PYTHONPATH=src python3 scripts/predict_delay.py \
  --model models/delay_logreg.joblib \
  --input data/flights_weather_sample.csv \
  --output models/scored_flights.csv \
  --include-actuals
```

`--include-actuals` is for validation and demos only. Do not treat the answer-key
columns as live predictive inputs.

## Example output columns

```text
flight_id, date, carrier, flight_number, origin, dest, sched_dep_local,
delay_probability, risk_band, predicted_delayed_15, top_reason,
reason_details, recommended_action
```

Example actions include checking the inbound aircraft, preparing de-icing or
weather mitigation, coordinating gate/ramp/ATC constraints, or monitoring low
risk flights without intervention.
