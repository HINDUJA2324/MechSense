# MechSense - Final CSV Validation + Forecast Fix

This build fixes the uploaded-CSV workflow and machine-level consistency problems.

## Fixed behavior

- Invalid CSVs are rejected before inference and the previous active dataset is not replaced.
- Industrial CSVs require a real machine identifier and either a cycle column or a timestamp.
- Every machine/engine must contain at least 30 ordered readings for the trained LSTM and time-series forecast.
- `UDI` is treated as a row identifier when a real `Machine_ID`/machine ID exists; it is not incorrectly treated as a machine identifier.
- Sensor values are checked against the distribution used to train the Industrial LSTM to catch unit/scale mismatches.
- Industrial failure-mode fields must be valid binary values when supplied.
- One uploaded machine receives exactly one LSTM status. Dashboard counts, machine details, alerts, and machine pages use that same status.
- Uploaded forecast data is always taken from the selected machine only; fleet-wide fallback data is never used.
- Forecasts are bounded to the selected machine's recent operating range and the trained model's expected range to prevent unrealistic exploding values.
- Forecast results expose increasing/decreasing/stable trend direction and detected historical deviations.
- The forecast machine selector is server-populated and does not wait for `/api/machines`, preventing an infinite `Loading machines...` state.
- `/api/forecasts` provides a batch forecast result for every valid uploaded machine.

## Important CSV rule

A file with one row per machine is not a time series. It is rejected for the LSTM/forecast workflow because there is no history to learn from.

For the Industrial LSTM in this build, each machine needs at least 30 readings.

## Run

Activate the project's Python environment, install `requirements.txt`, then run:

```powershell
python app.py
```

Open the Flask URL shown in the terminal, normally `http://127.0.0.1:5000`.
