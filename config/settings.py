import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.json")

# OpenWeather API Key
OPENWEATHER_API_KEY = os.getenv("0772b676ef44deb64c03aac472b12b48")

# NewsAPI API Key
NEWSAPI_API_KEY = os.getenv("3a38ee3e0b504ec68460b7ef5a78bc93")