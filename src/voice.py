import speech_recognition as sr
import pyttsx3
import os
import vosk
from dotenv import load_dotenv
from vosk import Model, KaldiRecognizer

def cargar_modelo_vosk():
    model_path = os.path.join("models/vosk-model-es-0.42")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo Vosk no encontrado en {model_path}")
    return Model(model_path)

# Uso
try:
    model = cargar_modelo_vosk()
except Exception as e:
    print(f"Error al cargar el modelo: {e}")

load_dotenv(os.path.join("config", ".env"))

# Configuración de síntesis de voz
engine = pyttsx3.init()
engine.setProperty("rate", 150)  # Velocidad del habla
model = Model("modelo-es")  # Descargar modelo en español

def hablar(texto):
    try:
        engine.say(texto)
        engine.runAndWait()
    except Exception as e:
        print(f"Error en síntesis de voz: {e}")

def escuchar():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        try:
            print("Escuchando...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=5)
            texto = recognizer.recognize_google(audio, language="es-ES")
            return texto.lower()
        except sr.UnknownValueError:
            print("No entendí el comando")
            return ""
        except sr.RequestError:
            print("Error de conexión con el servicio de voz")
            return ""
        except Exception as e:
            print(f"Error inesperado: {e}")
            return ""