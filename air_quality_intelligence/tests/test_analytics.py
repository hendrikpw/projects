import pandas as pd

from air_quality_intelligence.src.analytics import aqi_band, city_summary, prepare_frame


def _fixture() -> pd.DataFrame:
    return pd.DataFrame({
        "time": pd.to_datetime(["2026-07-24 09:00", "2026-07-24 10:00"]),
        "city": ["Stuttgart", "Stuttgart"], "european_aqi": [18.0, 68.0],
        "pm2_5": [8.0, 32.0], "pm10": [15.0, 48.0],
        "nitrogen_dioxide": [25.0, 90.0], "ozone": [40.0, 80.0],
    })


def test_aqi_band_boundaries() -> None:
    assert aqi_band(20) == "Good"
    assert aqi_band(40) == "Fair"
    assert aqi_band(60) == "Moderate"
    assert aqi_band(80) == "Poor"
    assert aqi_band(101) == "Extremely poor"


def test_prepare_frame_adds_diagnostics() -> None:
    result = prepare_frame(_fixture())
    assert {"date", "hour", "aqi_band", "dominant_pollutant"}.issubset(result.columns)
    assert result["aqi_band"].tolist() == ["Good", "Poor"]


def test_city_summary_counts_poor_hours() -> None:
    result = city_summary(prepare_frame(_fixture()))
    assert result.loc[0, "poor_hours"] == 1
    assert result.loc[0, "average_aqi"] == 43.0
