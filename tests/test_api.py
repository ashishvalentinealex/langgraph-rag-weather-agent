"""Unit tests for the weather helper functions.

These tests validate city extraction, geocoding, weather summarization,
and ensure that API logic can be mocked for offline testing.
"""

import pytest
from src.weather import extract_city_name, summarise_onecall


def test_extract_city_name_variants():
    assert extract_city_name("What is the weather in Paris today?") == "Paris"
    assert extract_city_name("tell me the weather in New York now") == "New York"
    assert extract_city_name("Hyderabad weather now") == "Hyderabad"
    # Should fall back to default when no city found
    assert extract_city_name("What's up?") == "Hyderabad"


def test_summarise_onecall():
    """Test that summarise_onecall produces human-readable output."""
    mock_json = {
        "current": {
            "temp": 29.2,
            "feels_like": 31.5,
            "humidity": 61,
            "wind_speed": 5.14,
            "weather": [{"description": "scattered clouds"}],
        }
    }
    summary = summarise_onecall(mock_json, "Hyderabad")
    assert "Hyderabad" in summary
    assert "29.2" in summary
    assert "scattered" in summary.lower()
    assert "humidity" in summary


def test_fetch_weather_for_city_monkeypatch(monkeypatch):
    """Mock the geocode and weather APIs to test end-to-end summarization."""
    from src import weather

    # Mock geocoding to return fixed lat/lon
    monkeypatch.setattr(weather, "geocode_city", lambda city, country_hint="IN": (17.36, 78.47))

    # Mock the One Call API response
    mock_weather_json = {
        "current": {
            "temp": 25.0,
            "feels_like": 26.0,
            "humidity": 70,
            "wind_speed": 4.5,
            "weather": [{"description": "clear sky"}],
        }
    }
    monkeypatch.setattr(weather, "fetch_weather_onecall", lambda lat, lon: mock_weather_json)

    result = weather.fetch_weather_for_city("what is the weather in hyderabad?")
    assert "Hyderabad" in result
    assert "clear" in result.lower()
    assert "humidity" in result
