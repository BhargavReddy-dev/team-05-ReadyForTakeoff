DATAbility: Explainable Flight Delay Risk Assistant for Airline Operations Controllers

## Team Information

Team Number: 05
Team Name : Team Fox

Team Members:
* [Harshal Hisoriya]
* [Tanay Patel]
* [Yash Thummar]
* [Bhargava Reddy]

---

## Challenge:

Challenge Name: DATAbility | Ready for Takeoff

We are tackling the problem of flight delay prediction and decision support for airline operations teams. The goal is not only to predict whether a flight may be delayed, but also to explain why the delay risk exists and recommend feasible solution for it.

---

## Problem:

Every day, airline operations teams must decide which flights are at risk of delay and what actions should be taken before the delay spreads through the network.

The biggest operational problem is not only one delayed flight. The real damage happens when one delayed aircraft affects its next leg, then the next leg, and eventually causes missed connections, crew issues, gate conflicts, and network-wide disruption.

Today, operations controllers often rely on scattered dashboards, manual monitoring, and experience-based judgment. Existing tools may show flight status or delay numbers, but they often do not clearly explain the reason behind the risk. A black-box risk score is difficult for an operator to trust and act on.

---

## Customer

Our primary customer is: Airline Operations Controller / Duty Controller / OCC Dispatcher

The user works inside the airline operations control center and makes real-time decisions such as:

* Which flights need close monitoring
* Whether to prepare faster turnaround support
* Whether to alert ground operations
* Whether to check backup aircraft availability
* Whether to prioritize certain flights because of cascade risk
* Whether weather or congestion may affect departure reliability

We are not building this tool for passengers. We are building it for the people responsible for keeping the airline schedule running.

---

## Validated Problem

The provided challenge material identifies flight delays and cancellations as a major operational and financial problem for airlines. It also emphasizes that delays often spread through the aircraft network. A late aircraft can delay the next flight, which can then delay later flights.

The provided dataset also supports this problem because it contains:

* Real scheduled flights
* Aircraft tail numbers
* Previous flights Departure and arrival delay
* Weather conditions at the origin airport
* Day and Time

The key insight is that late inbound aircraft, weather pressure, and airport congestion together create operational delay risk. Weather alone is not the full explanation. It often acts as a catalyst that triggers congestion and cascade delays.

---

## Solution

Our solution is a flight delay risk assistant called: Delay Risk Model watchlist which informs whether a flight is likely to be delayed by 15 minutes or more. More importantly, it explains the reason behind the risk, severity and gives a recommended action to user.
---

## Value Proposition

Delay Risk Model watchlist helps airline operation teams to move from reactive delay management to proactive decision support.

### What makes it valuable?

1. **Cascade-aware decision support**
   The model uses aircraft tail-number chaining to estimate whether the previous flight can affect the current one.

2. **Weather-aware but not weather-only**
   Weather is treated as an operational pressure signal, not as the only cause of delay.

3. **Actionable recommendations**
   The tool recommends what the controller should do, such as monitoring the inbound aircraft, alerting ground operations, or preparing fast turnaround support.

4. **Built for airline operations, not passengers**
   The interface is designed for a duty controller or OCC dispatcher who needs to make fast operational decisions.

