# jarvis/plugins/weather.py
import requests
import logging
import re  # Moved from handle method
from spacy.tokens import Doc  # Added for type hinting
from utils.config_manager import ConfigManager  # Import ConfigManager

logger = logging.getLogger(__name__)

WEATHER_API_KEY_NAME = "OPENWEATHER_API_KEY"  # Consistent with report_generator.py


class Plugin:
    def __init__(self):
        """Inicializa el plugin de clima"""
        self.config_manager = ConfigManager()  # Use ConfigManager
        self.api_key = self.config_manager.get_env_variable(WEATHER_API_KEY_NAME)
        if not self.api_key:
            logger.warning(
                f"No se ha configurado la API key para el clima ({WEATHER_API_KEY_NAME}). El plugin podría no funcionar.")
        logger.info("Plugin WeatherPlugin inicializado.")

    def get_description(self) -> str:
        return "Consulta el estado del tiempo actual para una ciudad."

    # Keep can_handle simple for now, or update to check entities if needed
    def can_handle(self, text: str, doc: Doc = None, context: dict = None, entities: list = None) -> bool:  # Added entities
        """Determina si este plugin puede manejar el texto de entrada"""
        text_lower = text.lower()
        # Option 1: Simple keyword check (current approach)
        # Combined Spanish and English keywords
        keywords = [
            "tiempo", "clima", "lluvia", "temperatura", "sol", "pronóstico",  # Spanish
            "weather", "forecast", "temperature", "rain", "sun"  # English
        ]
        if any(keyword in text_lower for keyword in keywords):
            return True

        # Option 2: Check doc lemmas (more robust)
        if doc:
            # Combined Spanish and English lemmas
            lemmas = [
                "tiempo", "clima", "llover", "temperatura", "sol", "pronosticar",  # Spanish
                "weather", "forecast", "temperature", "rain", "sun"  # English
            ]
            if any(token.lemma_ in lemmas for token in doc):
                return True

        # Option 3: Check merged entities (if passed)
        # if entities:
        #    if any(ent['label'] in ["GPE", "LOC"] for ent in entities): # Example: trigger if location found
        #        return True

        return False  # Default if no checks pass

    def handle(self, text: str, doc: Doc = None, context: dict = None, entities: list = None) -> str:  # Added entities
        """
        Procesa la solicitud de clima y retorna la información, prioritizing merged entities.

        Args:
            text (str): Texto del comando (ej: "¿Cuál es el clima en Madrid?")
            doc (spacy.tokens.Doc, optional): Documento spaCy procesado. Defaults to None.
            context (dict, optional): Contexto adicional. Defaults to None.
            entities (list, optional): Merged list of entities from IntentProcessor. Defaults to None.

        Returns:
            str: Información sobre el clima o un mensaje de error
        """
        if not self.api_key:
            return "Lo siento, la API key para el servicio de clima no está configurada."

        city = None

        # 1. Prioritize merged entities if available
        if entities:
            for ent in entities:
                if ent['label'] in ["GPE", "LOC"]:
                    city = ent['text']
                    logger.info(f"Ciudad extraída de merged_entities: {city} (Label: {ent['label']}, Source: {ent.get('source', 'N/A')})")
                    break

        # 2. Fallback to spaCy doc.ents if merged_entities didn't yield a city
        if not city and doc:
            logger.info("No city found in merged_entities, checking doc.ents...")
            for ent in doc.ents:
                if ent.label_ in ["GPE", "LOC"]:
                    city = ent.text
                    logger.info(f"Ciudad extraída de spaCy Doc: {city} (Entidad: {ent.label_})")
                    break

        # 3. Fallback to regex if still no city found
        if not city:
            logger.info("No se encontró ciudad en spaCy Doc, intentando regex sobre el texto.")
            # import re # Moved to top
            city_match = re.search(r"(?:clima|tiempo|temperatura)\s+(?:en|de|para)\s+([A-Za-záéíóúñÁÉÍÓÚÑ\s]+)", text,
                                  re.IGNORECASE)
            if city_match:
                city = city_match.group(1).strip()
                logger.info(f"Ciudad extraída con regex: {city}")

        if not city:
            logger.info("No se pudo extraer la ciudad.")
            # Considerar si el contexto tiene una ciudad previa
            if context and context.get('last_city_queried'):
                city = context.get('last_city_queried')
                logger.info(f"Usando última ciudad consultada del contexto: {city}")
            else:
                return "¿Para qué ciudad te gustaría saber el clima?"

        try:
            base_url = self.config_manager.get_app_setting("openweathermap_base_url",
                                                            "http://api.openweathermap.org/data/2.5/weather?")
            lang_code = self.config_manager.get_app_setting("weather_plugin_lang", "es")
            units = self.config_manager.get_app_setting("weather_plugin_units", "metric")

            complete_url = f"{base_url}q={city}&appid={self.api_key}&units={units}&lang={lang_code}"
            logger.debug(f"Consultando OpenWeatherMap: {complete_url.replace(self.api_key, '***')}")

            response = requests.get(complete_url, timeout=10)
            response.raise_for_status()  # Levanta HTTPError para respuestas 4XX/5XX
            weather_data = response.json()

            if weather_data.get("cod") != 200 and weather_data.get("cod") != "200":  # API a veces retorna "200" como string
                # Manejar errores específicos de la API si es necesario
                error_message = weather_data.get("message", "Ciudad no encontrada o error de API.")
                logger.warning(f"Error de API OpenWeatherMap para {city}: {error_message} (Code: {weather_data.get('cod')})")
                return f"No pude encontrar el clima para {city}. {error_message}"

            main = weather_data.get("main", {})
            temperature = main.get("temp", "N/A")
            humidity = main.get("humidity", "N/A")
            weather_description_list = weather_data.get("weather", [{}])
            weather_description = weather_description_list[0].get("description", "N/A") if weather_description_list else "N/A"

            # Actualizar contexto con la ciudad consultada (ejemplo)
            updated_context = {"last_city_queried": city}

            weather_report = (f"En {city}, la temperatura es de {temperature}°C con {weather_description}. "
                              f"La humedad es del {humidity}%.")
            logger.info(f"Clima obtenido para {city}: {weather_report}")
            return weather_report, updated_context  # Devolver también el contexto actualizado

        except requests.Timeout:
            logger.error(f"Timeout al contactar OpenWeatherMap para {city}.")
            return "El servicio de clima tardó demasiado en responder."
        except requests.RequestException as e:
            logger.error(f"Error de conexión al obtener clima para {city}: {e}")
            return f"Error de conexión al obtener el clima para {city}."
        except Exception as e:
            logger.error(f"Error inesperado en plugin de clima para {city}: {e}", exc_info=True)
            return "Lo siento, ocurrió un error inesperado al procesar la información del clima."
