"""
Weather utilities for the LangGraph agent.
Fetches live data from OpenWeatherMap using Geocoding and One Call APIs.
"""

import os
import re
import requests
from dotenv import load_dotenv

load_dotenv()


# -----------------------------------------------------
#  City extraction helper
# -----------------------------------------------------
def extract_city_name(user_query: str, default_city: str = "Hyderabad") -> str:
    text = user_query.lower()
    # capture up to the first non-letter word after “in”
    m = re.search(r"in ([a-zA-Z\s]+?)(?:\s+(?:today|now|tomorrow|weather|forecast))?[\s\?\,\.]*$", text)
    if m:
        city = m.group(1).strip().title()
        return city
    m = re.search(r"([a-zA-Z\s]+?)\s+weather", text)
    if m:
        city = m.group(1).strip().title()
        return city
    return default_city


# -----------------------------------------------------
#  OpenWeather API helpers
# -----------------------------------------------------
def geocode_city(city_name: str, country_hint: str = "IN"):
    """Use OpenWeather geocoding API to get lat/lon for a given city."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = f"https://api.openweathermap.org/geo/1.0/direct?q={city_name},{country_hint}&limit=1&appid={api_key}"
    print("DEBUG geocode url:", url)
    resp = requests.get(url, timeout=10)
    data = resp.json()
    print("DEBUG geocode response:", data)
    return data


def fetch_weather_onecall(lat: float, lon: float):
    """Fetch current weather data using the One Call 3.0 API."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = (
        f"https://api.openweathermap.org/data/3.0/onecall?"
        f"lat={lat}&lon={lon}&exclude=minutely,hourly,daily,alerts&units=metric&appid={api_key}"
    )
    print("DEBUG weather url:", url)
    resp = requests.get(url, timeout=10)
    data = resp.json()
    print("DEBUG weather response:", data)
    return data


def summarise_onecall(weather_json: dict, city_name: str) -> str:
    """Generate a readable weather summary from One Call data."""
    try:
        current = weather_json["current"]
        desc = current["weather"][0]["description"].capitalize()
        temp = current["temp"]
        feels = current["feels_like"]
        hum = current["humidity"]
        wind = current["wind_speed"]
        return (
            f"Current weather in {city_name}: {temp:.1f}°C (feels like {feels:.1f}°C), "
            f"humidity {hum}%, {desc}, wind {wind} m/s."
        )
    except Exception as e:
        return f"Sorry, I couldn’t parse the weather data correctly: {e}"


def fetch_weather_for_city(user_query: str, default_city="Hyderabad", country_hint="IN"):
    """
    Main entry: extract city from user query, geocode it, fetch weather,
    and return a natural-language summary.
    """
    city = extract_city_name(user_query, default_city)
    print("DEBUG extracted city:", city)

    data = geocode_city(city, country_hint)
    if not data or len(data) == 0:
        raise ValueError(f"Could not geocode city: {city},{country_hint}")
    

    if isinstance(data, tuple) and len(data) == 2:
        lat, lon = data
    elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        lat = data[0].get("lat")
        lon = data[0].get("lon")
    else:
        raise ValueError(f"Unexpected geocode response format: {data}")


    weather = fetch_weather_onecall(lat, lon)
    return summarise_onecall(weather, city)
