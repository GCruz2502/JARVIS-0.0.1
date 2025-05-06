# jarvis/plugins/weather.py
import requests
import os
import logging

logger = logging.getLogger(__name__)

class Plugin:
    def __init__(self):
        """Inicializa el plugin de clima"""
        self.api_key = os.getenv("WEATHER_API_KEY")
        if not self.api_key:
            logger.warning("No se ha configurado la API key para el clima")
    
    def can_handle(self, text):
        """Determina si este plugin puede manejar el texto de entrada"""
        keywords = ["tiempo", "clima", "lluvia", "temperatura", "sol"]
        return any(keyword in text for keyword in keywords)
    
    def handle(self, text):
        """
        Procesa la solicitud de clima y retorna la información
        
        Args:
            text (str): Texto del comando (ej: "¿Cuál es el clima en Madrid?")
            
        Returns:
            str: Información sobre el clima o un mensaje de error
        """
        if not self.api_key:
            return "Lo siento, no tengo configurada la API del clima."
        
        # Extraer la ciudad del texto
        import re
        city_match = re.search(r"(clima|tiempo)(\s+en\s+)(\w+)", text)
        
        if city_match:
            city = city_match.group(3)
        else:
            # Si no se detecta ciudad, usar una por defecto o pedir aclaración
            return "¿En qué ciudad quieres saber el clima?"
        
        try:
            # Llamada a la API de OpenWeatherMap
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={self.api_key}&units=metric&lang=es"
            response = requests.get(url)
            
            if response.status_code == 200:
                weather_data = response.json()
                temp = weather_data["main"]["temp"]
                condition = weather_data["weather"][0]["description"]
                
                return f"En {city} hay {condition} con una temperatura de {temp}°C."
            else:
                logger.error(f"Error en API del clima: {response.status_code}")
                return f"Lo siento, no pude obtener el clima para {city}."
                
        except Exception as e:
            logger.error(f"Error al consultar el clima: {str(e)}")
            return "Hubo un problema al obtener la información del clima."