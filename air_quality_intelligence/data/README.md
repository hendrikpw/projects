# Data handling

The application requests live hourly forecasts at runtime and does not commit a
large raw dataset to the repository.

If the endpoint is temporarily unavailable, `src/data.py` creates a deterministic
synthetic dataset for interface continuity. The dashboard labels this mode
prominently. Synthetic values are not Open-Meteo/CAMS observations or forecasts.

| Field | Meaning | Unit |
|---|---|---|
| `european_aqi` | Consolidated European Air Quality Index | index |
| `pm2_5` | Particulate matter below 2.5 µm | µg/m³ |
| `pm10` | Particulate matter below 10 µm | µg/m³ |
| `nitrogen_dioxide` | Near-surface NO₂ | µg/m³ |
| `ozone` | Near-surface O₃ | µg/m³ |

Source: https://open-meteo.com/en/docs/air-quality-api
