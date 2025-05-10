"""
Plugin para obtener las últimas noticias.
"""
import requests
import logging
from utils.config_manager import ConfigManager # Import ConfigManager

logger = logging.getLogger(__name__)

NEWS_API_KEY_NAME = "NEWSAPI_API_KEY" # Nombre de la variable de entorno para la API key

class Plugin:
    def __init__(self):
        self.config_manager = ConfigManager() # Asume que ConfigManager es un singleton o es seguro instanciar
        logger.info("Plugin NewsPlugin inicializado.")
        if not self.config_manager.get_env_variable(NEWS_API_KEY_NAME):
            logger.warning(f"Variable de entorno {NEWS_API_KEY_NAME} no encontrada. El plugin de noticias podría no funcionar.")

    def get_description(self) -> str:
        return "Obtiene los titulares de noticias más recientes."

    def can_handle(self, text: str, doc=None, context: dict = None, entities: list = None) -> bool:
        text_lower = text.lower()
        keywords = ["noticias", "news", "titulares", "headlines", "actualidad"]
        return any(keyword in text_lower for keyword in keywords)

    def handle(self, text: str, doc=None, context: dict = None, entities: list = None) -> str:
        news_api_key = self.config_manager.get_env_variable(NEWS_API_KEY_NAME)
        if not news_api_key:
            logger.error(f"API key para NewsAPI ({NEWS_API_KEY_NAME}) no configurada.")
            return f"Lo siento, la API key para el servicio de noticias no está configurada."

        try:
            # Podrías querer hacer la URL base y el país configurables a través de app_config.json
            country = self.config_manager.get_app_setting("news_plugin_country", "us")
            base_url = "https://newsapi.org/v2/top-headlines?"
            complete_url = f"{base_url}country={country}&apiKey={news_api_key}&pageSize=5" # Limitar a 5 noticias
            
            logger.debug(f"Consultando NewsAPI: {complete_url.replace(news_api_key, '***')}") # No loguear la API key
            response = requests.get(complete_url, timeout=10)
            response.raise_for_status() # Levanta HTTPError para respuestas 4XX/5XX
            news_data = response.json()
            
            if news_data.get("status") == "ok":
                articles = news_data.get("articles", [])
                if not articles:
                    return "No se encontraron noticias en este momento."
                    
                headlines = [f"• {article.get('title', 'Sin título')}" for article in articles] # Tomar los 5 que pedimos
                news_report = "Aquí tienes los titulares:\n" + "\n".join(headlines)
                logger.info("Noticias obtenidas exitosamente.")
                return news_report
            else:
                error_msg = news_data.get("message", "Error desconocido de la API de noticias")
                logger.error(f"Error en API de noticias: {error_msg}")
                return f"Error al obtener noticias: {error_msg}"
                
        except requests.Timeout:
            logger.error("Timeout al contactar NewsAPI.")
            return "El servicio de noticias tardó demasiado en responder."
        except requests.RequestException as e:
            logger.error(f"Error de conexión al obtener noticias: {e}")
            return f"Error de conexión al obtener noticias: {e}"
        except Exception as e:
            logger.error(f"Error inesperado en el plugin de noticias: {e}", exc_info=True)
            return "Lo siento, ocurrió un error inesperado al procesar las noticias."
