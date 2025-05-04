"""
Módulo para manejar la entrada y salida de voz
"""
import speech_recognition as sr
import pyttsx3
import logging
import os
import vosk
from dotenv import load_dotenv
from vosk import Model, KaldiRecognizer

logger = logging.getLogger(__name__)

# Iniciar motor de voz
engine = pyttsx3.init()


engine.setProperty("rate", 150)  # Velocidad del habla
model = Model("modelo-es")  # Descargar modelo en español

def hablar(texto):
    """"
    Convierte texto a voz
    
    Args:
        texto (str): El texto que se convertirá a voz
    """
    try:
        logger.info(f"Hablando: {texto}")
        engine.say(texto)
        engine.runAndWait()
    except Exception as e:
        logger.error(f"Error al hablar: {e}")
        print(f"Error al hablar: {e}")

def escuchar(timeout=5, phrase_time_limit=5):
    """
    Escucha el micrófono y devuelve el texto reconocido
    
    Args:
        timeout (int): Tiempo máximo de espera para comenzar a hablar
        phrase_time_limit (int): Tiempo máximo para una frase

    Returns:
        str or none: El texto reconocido o None si hay un error
        """
    recognizer = sr.Recognizer()


    try:
        with sr.Microphone() as source:
            print("Escuchando...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)

        print("Procesando")
        texto = recognizer.recognize_google(audio, language="es-ES")
        print(f"Has dicho {texto}")
        return texto.lower()
    
    except sr.WaitTimeoutError:
            print("Tiempo de espera agotado. Por favor, habla después de escuchar 'Escuchando...'")
    except sr.UnknownValueError:
            print("No he podido entender lo que has dicho")
    except sr.RequestError as e:
            print(f"Error con el servicio de reconocimiento de voz: {e}")
    except Exception as e:
            logger.error(f"Error inesperado en reconocimiento de voz: {e}")
            print(f"Error: {e}")

    return None

# Función falback por si no hay micrófono o hay problemas
def entrada_texto():
    """
    Solicita entrada de texto como alternativa a la voz
      
    Returns:
        str: El texto ingresado por el usuario
    """
    return input("Escribe tu comando: ").lower()