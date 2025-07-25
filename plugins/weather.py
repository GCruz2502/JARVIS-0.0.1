# jarvis/plugins/weather.py
import requests
import logging
from utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)

WEATHER_API_KEY_NAME = "OPENWEATHER_API_KEY"

# --- NEW: Centralized, bilingual text for all responses ---
RESPONSE_TEXTS = {
    "es": {
        "report": "En {city}, la temperatura es de {temp}°C con {desc}. La humedad es del {humidity}%.",
        "ask_city": "¿Para qué ciudad te gustaría saber el clima?",
        "not_found": "No pude encontrar el clima para {city}. Por favor, revisa el nombre.",
        "api_key_error": "Lo siento, la API key para el servicio de clima no está configurada.",
        "timeout_error": "El servicio de clima tardó demasiado en responder.",
        "connection_error": "Error de conexión al obtener el clima para {city}.",
        "unexpected_error": "Lo siento, ocurrió un error inesperado al procesar la información del clima."
    },
    "en": {
        "report": "In {city}, the temperature is {temp}°F with {desc}. The humidity is at {humidity}%.",
        "ask_city": "For which city would you like to know the weather?",
        "not_found": "I couldn't find the weather for {city}. Please check the name.",
        "api_key_error": "Sorry, the API key for the weather service is not configured.",
        "timeout_error": "The weather service took too long to respond.",
        "connection_error": "There was a connection error while getting the weather for {city}.",
        "unexpected_error": "Sorry, an unexpected error occurred while processing the weather information."
    }
}
# --- END NEW ---

class Plugin:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.api_key = self.config_manager.get_env_variable(WEATHER_API_KEY_NAME)
        if not self.api_key:
            logger.warning(f"Weather API key ({WEATHER_API_KEY_NAME}) is not set. Plugin may not work.")
        logger.info("Plugin WeatherPlugin inicializado.")

    def get_description(self) -> str:
        return "Consulta el estado del tiempo actual para una ciudad. / Gets the current weather for a city."

    # --- REMOVED: The can_handle method is no longer used by the new IntentProcessor ---

    def handle(self, text: str, doc=None, context: dict = None, entities: list = None) -> str:
        # --- NEW: Determine language and get localized texts ---
        current_lang = context.get('current_conversation_lang', 'es') if context else 'es'
        responses = RESPONSE_TEXTS[current_lang]

        if not self.api_key:
            return responses["api_key_error"]

        city = None
        # Prioritize entities to find the city
        if entities:
            for ent in entities:
                if ent['label'] in ["GPE", "LOC"]:
                    city = ent['text']
                    logger.info(f"City found in entities: {city}")
                    break
        
        # If still no city, ask the user
        if not city:
            logger.info("Could not extract a city from the input.")
            return responses["ask_city"]

        # --- NEW: Localize API parameters ---
        lang_for_api = "es" if current_lang == "es" else "en"
        units_for_api = "metric" if current_lang == "es" else "imperial" # metric=Celsius, imperial=Fahrenheit
        unit_symbol = "°C" if units_for_api == "metric" else "°F"

        try:
            base_url = "http://api.openweathermap.org/data/2.5/weather?"
            complete_url = f"{base_url}q={city}&appid={self.api_key}&units={units_for_api}&lang={lang_for_api}"
            logger.debug(f"Querying OpenWeatherMap: {complete_url.replace(self.api_key, '***')}")

            response = requests.get(complete_url, timeout=10)
            
            # Check for 404 Not Found specifically
            if response.status_code == 404:
                logger.warning(f"City not found on OpenWeatherMap: {city}")
                return responses["not_found"].format(city=city)

            response.raise_for_status() # Handle other errors (4xx, 5xx)
            weather_data = response.json()

            main = weather_data.get("main", {})
            temperature = main.get("temp", "N/A")
            humidity = main.get("humidity", "N/A")
            weather_list = weather_data.get("weather", [{}])
            weather_description = weather_list[0].get("description", "N/A") if weather_list else "N/A"

            # --- NEW: Build the localized report ---
            weather_report = responses["report"].format(
                city=city.title(),
                temp=round(temperature),
                desc=weather_description,
                humidity=humidity
            )
            logger.info(f"Weather report generated for {city}: {weather_report}")
            return weather_report

        except requests.Timeout:
            logger.error(f"Timeout contacting OpenWeatherMap for {city}.")
            return responses["timeout_error"]
        except requests.RequestException as e:
            logger.error(f"Connection error for {city}: {e}")
            return responses["connection_error"].format(city=city)
        except Exception as e:
            logger.error(f"Unexpected error in weather plugin for {city}: {e}", exc_info=True)
            return responses["unexpected_error"]