import requests
import datetime
import time
import pyttsx3
from transformers import pipeline
from config.settings import OPENWEATHER_API_KEY, NEWSAPI_API_KEY

def get_weather(city):
    """Fetch weather information for a given city."""
    if not OPENWEATHER_API_KEY:
        return "OpenWeather API key is not set."
    
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = base_url + "q=" + city + "&appid=" + OPENWEATHER_API_KEY + "&units=metric"
    response = requests.get(complete_url)
    weather_data = response.json()
    if weather_data["cod"] != "404":
        main = weather_data["main"]
        temperature = main["temp"]
        humidity = main["humidity"]
        weather_description = weather_data["weather"][0]["description"]
        weather_report = f"Temperature: {temperature}°C\nHumidity: {humidity}%\nDescription: {weather_description}"
        return weather_report
    else:
        return "City not found."

def get_news():
    """Fetch top news headlines."""
    if not NEWSAPI_API_KEY:
        return "NewsAPI API key is not set."
    
    base_url = "https://newsapi.org/v2/top-headlines?"
    complete_url = base_url + "country=us&apiKey=" + NEWSAPI_API_KEY
    response = requests.get(complete_url)
    news_data = response.json()
    headlines = [article["title"] for article in news_data["articles"][:5]]
    news_report = "\n".join(headlines)
    return news_report

def set_reminder(reminder_text, reminder_time):
    """Set a reminder."""
    current_time = datetime.datetime.now()
    reminder_time = datetime.datetime.strptime(reminder_time, "%Y-%m-%d %H:%M:%S")
    time_diff = (reminder_time - current_time).total_seconds()
    if time_diff > 0:
        engine = pyttsx3.init()
        engine.say(f"Reminder set for {reminder_time}")
        engine.runAndWait()
        time.sleep(time_diff)
        engine.say(f"Reminder: {reminder_text}")
        engine.runAndWait()
    else:
        engine = pyttsx3.init()
        engine.say("The reminder time has already passed.")
        engine.runAndWait()

def chat_with_jarvis(query):
    """Chat with JARVIS using a conversational AI model."""
    chatbot = pipeline("text-generation", model="microsoft/DialoGPT-medium")
    response = chatbot(query, max_length=50, num_return_sequences=1)
    return response[0]['generated_text']