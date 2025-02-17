import pyttsx3
from core.database import collect_data
from core.ml_models import train_model, predict_response
from core.reportes import get_weather, get_news, set_reminder, chat_with_jarvis

# Initialize the text-to-speech engine
engine = pyttsx3.init()

def speak(text):
    """Convert text to speech."""
    engine.say(text)
    engine.runAndWait()

def start_cli():
    """Start the command-line interface."""
    speak("Hi, I am Jarvis. How can I assist you today?")
    model = train_model()

    while True:
        command = input("You: ").lower()
        if 'weather' in command:
            city = input("Enter city name: ")
            weather_report = get_weather(city)
            speak(weather_report)
            print(f"JARVIS: {weather_report}")
            collect_data(command, weather_report)
        elif 'news' in command:
            news_report = get_news()
            speak(news_report)
            print(f"JARVIS: {news_report}")
            collect_data(command, news_report)
        elif 'reminder' in command:
            reminder_text = input("Enter reminder text: ")
            reminder_time = input("Enter reminder time (YYYY-MM-DD HH:MM:SS): ")
            set_reminder(reminder_text, reminder_time)
            collect_data(command, f"Reminder set for {reminder_time}")
        elif 'exit' in command or 'quit' in command:
            speak("Goodbye!")
            print("JARVIS: Goodbye!")
            break
        else:
            response = predict_response(model, command)
            if response == "I don't know the answer to that yet.":
                response = chat_with_jarvis(command)
            speak(response)
            print(f"JARVIS: {response}")
            collect_data(command, response)