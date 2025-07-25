"""
Plugin para obtener las últimas noticias.
"""
import requests
import logging
from utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)

NEWS_API_KEY_NAME = "NEWSAPI_API_KEY"

# --- NEW: Centralized, bilingual text for all responses ---
RESPONSE_TEXTS = {
    "es": {
        "headlines_intro": "Aquí tienes los titulares:",
        "no_articles": "No se encontraron noticias en este momento.",
        "api_key_error": "Lo siento, la API key para el servicio de noticias no está configurada.",
        "api_status_error": "Error al obtener noticias: La API reportó un problema.",
        "timeout_error": "El servicio de noticias tardó demasiado en responder.",
        "connection_error": "Error de conexión al obtener noticias.",
        "unexpected_error": "Lo siento, ocurrió un error inesperado al procesar las noticias."
    },
    "en": {
        "headlines_intro": "Here are the top headlines:",
        "no_articles": "No news articles were found at this time.",
        "api_key_error": "Sorry, the API key for the news service is not configured.",
        "api_status_error": "Error getting news: The API reported a problem.",
        "timeout_error": "The news service took too long to respond.",
        "connection_error": "There was a connection error while getting the news.",
        "unexpected_error": "Sorry, an unexpected error occurred while processing the news."
    }
}
# --- END NEW ---

class Plugin:
    def __init__(self):
        self.config_manager = ConfigManager()
        logger.info("Plugin NewsPlugin inicializado.")
        if not self.config_manager.get_env_variable(NEWS_API_KEY_NAME):
            logger.warning(f"Environment variable {NEWS_API_KEY_NAME} not found. News plugin may not work.")

    def get_description(self) -> str:
        # --- NEW: Bilingual description ---
        return "Obtiene los titulares de noticias más recientes. / Gets the latest news headlines."

    # --- REMOVED: The can_handle method is no longer used by the new IntentProcessor ---

    def handle(self, text: str, doc=None, context: dict = None, entities: list = None) -> str:
        # --- NEW: Determine language from context ---
        current_lang = "es" # Default to Spanish
        if context and 'current_conversation_lang' in context:
            current_lang = context['current_conversation_lang']

        # Get the correct set of response texts for the current language
        responses = RESPONSE_TEXTS[current_lang]

        news_api_key = self.config_manager.get_env_variable(NEWS_API_KEY_NAME)
        if not news_api_key:
            logger.error(f"API key for NewsAPI ({NEWS_API_KEY_NAME}) not configured.")
            return responses["api_key_error"]

        try:
            # --- NEW: Localize the country for the API call ---
            country_for_api = "es" if current_lang == "es" else "us"
            # Allow user to override this in config if they want, e.g., 'gb' for British news
            country_for_api = self.config_manager.get_app_setting(f"news_plugin_country_{current_lang}", country_for_api)

            base_url = "https://newsapi.org/v2/top-headlines?"
            complete_url = f"{base_url}country={country_for_api}&apiKey={news_api_key}&pageSize=5"
            
            logger.debug(f"Querying NewsAPI: {complete_url.replace(news_api_key, '***')}")
            response = requests.get(complete_url, timeout=10)
            response.raise_for_status()
            news_data = response.json()
            
            if news_data.get("status") == "ok":
                articles = news_data.get("articles", [])
                if not articles:
                    return responses["no_articles"]
                    
                # Format headlines with their source for better context
                headlines = [f"• {article.get('title', 'No Title')} - {article.get('source', {}).get('name', 'No Source')}" for article in articles]
                news_report = responses["headlines_intro"] + "\n" + "\n".join(headlines)
                logger.info("News headlines retrieved successfully.")
                return news_report
            else:
                error_msg = news_data.get("message", "Unknown API error")
                logger.error(f"News API status error: {error_msg}")
                return responses["api_status_error"]
                
        except requests.Timeout:
            logger.error("Timeout while contacting NewsAPI.")
            return responses["timeout_error"]
        except requests.RequestException as e:
            logger.error(f"Connection error while retrieving news: {e}")
            return responses["connection_error"]
        except Exception as e:
            logger.error(f"Unexpected error in news plugin: {e}", exc_info=True)
            return responses["unexpected_error"]