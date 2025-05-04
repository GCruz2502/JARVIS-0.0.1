"""
Módulo para funcionalidades externas: clima, noticias, recordatorios y chat
"""
import requests
import datetime
import threading
import logging
from config.settings import OPENWEATHER_API_KEY, NEWSAPI_API_KEY

logger = logging.getLogger(__name__)

# Intentar importar transformers, pero tener una alternativa si no está disponible
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("Transformers no instalado. Chat con IA no estará disponible.")
    TRANSFORMERS_AVAILABLE = False

def get_weather(city):
    """
    Obtiene información del clima para una ciudad
    
    Args:
        city (str): Nombre de la ciudad
        
    Returns:
        str: Reporte del clima o mensaje de error
    """
    if not OPENWEATHER_API_KEY:
        logger.error("OpenWeather API key no configurada")
        return "OpenWeather API key no configurada."
    
    try:
        base_url = "http://api.openweathermap.org/data/2.5/weather?"
        complete_url = f"{base_url}q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(complete_url, timeout=10)
        weather_data = response.json()
        
        if weather_data.get("cod") != "404":
            main = weather_data.get("main", {})
            temperature = main.get("temp", "N/A")
            humidity = main.get("humidity", "N/A")
            weather_description = weather_data.get("weather", [{}])[0].get("description", "N/A")
            
            weather_report = f"Temperatura: {temperature}°C\nHumedad: {humidity}%\nDescripción: {weather_description}"
            logger.info(f"Clima obtenido para {city}")
            return weather_report
        else:
            logger.warning(f"Ciudad no encontrada: {city}")
            return "Ciudad no encontrada."
            
    except requests.RequestException as e:
        logger.error(f"Error al obtener clima: {e}")
        return f"Error al obtener el clima: {e}"
    except Exception as e:
        logger.error(f"Error inesperado en get_weather: {e}")
        return "Ocurrió un error al procesar la información del clima."

def get_news():
    """
    Obtiene titulares de noticias
    
    Returns:
        str: Titulares de noticias o mensaje de error
    """
    if not NEWSAPI_API_KEY:
        logger.error("NewsAPI API key no configurada")
        return "NewsAPI API key no configurada."
    
    try:
        base_url = "https://newsapi.org/v2/top-headlines?"
        complete_url = f"{base_url}country=us&apiKey={NEWSAPI_API_KEY}"
        response = requests.get(complete_url, timeout=10)
        news_data = response.json()
        
        if news_data.get("status") == "ok":
            articles = news_data.get("articles", [])
            if not articles:
                return "No se encontraron noticias."
                
            headlines = [f"• {article.get('title', 'Sin título')}" for article in articles[:5]]
            news_report = "\n".join(headlines)
            logger.info("Noticias obtenidas exitosamente")
            return news_report
        else:
            error_msg = news_data.get("message", "Error desconocido")
            logger.error(f"Error en API de noticias: {error_msg}")
            return f"Error al obtener noticias: {error_msg}"
            
    except requests.RequestException as e:
        logger.error(f"Error al obtener noticias: {e}")
        return f"Error al obtener noticias: {e}"
    except Exception as e:
        logger.error(f"Error inesperado en get_news: {e}")
        return "Ocurrió un error al procesar las noticias."

def _reminder_task(reminder_text, delay_seconds):
    """
    Tarea para recordatorio en segundo plano
    
    Args:
        reminder_text (str): Texto del recordatorio
        delay_seconds (float): Segundos de espera
    """
    import time
    import pyttsx3
    
    time.sleep(delay_seconds)
    
    engine = pyttsx3.init()
    engine.say(f"¡Recordatorio! {reminder_text}")
    engine.runAndWait()
    
    logger.info(f"Recordatorio activado: {reminder_text}")

def set_reminder(reminder_text, reminder_time):
    """
    Programa un recordatorio
    
    Args:
        reminder_text (str): Texto del recordatorio
        reminder_time (str): Hora del recordatorio en formato "YYYY-MM-DD HH:MM:SS"
        
    Returns:
        str: Mensaje de confirmación o error
    """
    try:
        current_time = datetime.datetime.now()
        reminder_time = datetime.datetime.strptime(reminder_time, "%Y-%m-%d %H:%M:%S")
        time_diff = (reminder_time - current_time).total_seconds()
        
        if time_diff > 0:
            # Iniciar el recordatorio en un hilo separado
            reminder_thread = threading.Thread(
                target=_reminder_task,
                args=(reminder_text, time_diff),
                daemon=True
            )
            reminder_thread.start()
            
            logger.info(f"Recordatorio programado para {reminder_time}")
            return f"Recordatorio programado para {reminder_time}"
        else:
            logger.warning(f"Intento de programar recordatorio en el pasado: {reminder_time}")
            return "La hora del recordatorio ya ha pasado."
            
    except ValueError:
        logger.error(f"Formato de fecha/hora incorrecto: {reminder_time}")
        return "Formato de fecha/hora incorrecto. Usa YYYY-MM-DD HH:MM:SS."
    except Exception as e:
        logger.error(f"Error al programar recordatorio: {e}")
        return f"Error al programar recordatorio: {e}"

def chat_with_jarvis(query):
    """
    Conversa usando modelo de IA
    
    Args:
        query (str): Consulta del usuario
        
    Returns:
        str: Respuesta generada
    """
    if not TRANSFORMERS_AVAILABLE:
        return "Lo siento, la funcionalidad de chat avanzado no está disponible."
    
    try:
        chatbot = pipeline("text-generation", model="microsoft/DialoGPT-medium")
        response = chatbot(query, max_length=50, num_return_sequences=1)
        text = response[0]['generated_text']
        
        # Limpiar la respuesta para que no incluya la consulta original
        if text.startswith(query):
            text = text[len(query):].strip()
        
        logger.info("Respuesta generada con modelo de IA")
        return text if text else "No tengo una respuesta para eso."
        
    except Exception as e:
        logger.error(f"Error en chat_with_jarvis: {e}")
        return "Lo siento, no pude generar una respuesta en este momento."