# Model Card: Ready for Takeoff Logistic Regression

## Purpose

Predict whether a scheduled flight is likely to depart 15 or more minutes late,
then translate the score into a reason and an operations-controller action.

## User

The intended user is an airline operations controller, dispatcher, or hub
coordinator. The model is designed to support triage, not to automate final
operational decisions.

## Data

The training data is `data/flights_weather_sample.csv`, a real one-week sample
of departures from Chicago O'Hare and Denver joined to historical origin
weather.

## Target

`delayed_15`, where 1 means `dep_delay_min >= 15`.

Cancelled flights are excluded from training because they do not have a stable
departure-delay target in the starter data.

## Features

The model uses only information known before departure:

- Schedule and route: day of week, departure hour, distance, carrier, origin,
  destination
- Weather: temperature, sustained wind, gusts, precipitation, snowfall, cloud
  cover, weather code
- Operations cascade: previous leg arrival delay for the same `tail_number`
- Bank indicators: early and evening departure banks

## Leakage Controls

These columns are excluded from training:

- `dep_delay_min`
- `arr_delay_min`
- `delayed_15`
- `cancelled`
- `delay_cause`
- `late_aircraft_delay_min`
- `weather_delay_min`

The only use of `arr_delay_min` is to build `inbound_arr_delay_min` from the
previous leg of the same aircraft, which is the permitted cascade signal in the
starter-kit rules.

## Validation

Validation uses a forward date split: train on earlier days and evaluate on the
last 20% of days. This is closer to the operational setting than a random split.

Latest metrics from `models/metrics.json`:

- ROC AUC: 0.668
- Average precision: 0.524
- Accuracy: 0.654
- Precision: 0.493
- Recall: 0.493
- F1: 0.493

## Limitations

The sample covers only two hubs and one winter week. The model does not include
crew duty limits, aircraft type, live ATC slots, passenger connections, or full
multi-leg network state. Treat high-risk predictions as a triage queue that
needs controller review.
