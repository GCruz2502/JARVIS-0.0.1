"""
Plugin para controlar el navegador web.
"""
import webbrowser
import logging
# Import ConfigManager. Asegúrate que utils está en PYTHONPATH o usa rutas relativas si es necesario.
# from ..utils.config_manager import ConfigManager # Ejemplo de ruta relativa si plugins y utils son hermanos
# Por ahora, asumimos que 'utils' es directamente importable.
from utils.config_manager import ConfigManager


logger = logging.getLogger(__name__)

class Plugin:
    def __init__(self):
        """
        Inicializa el plugin de control del navegador.
        """
        # Es preferible que ConfigManager sea un singleton o se pase a través del contexto
        # para evitar múltiples inicializaciones y asegurar un acceso consistente a la configuración.
        # Aquí se asume que ConfigManager() devuelve la instancia singleton o que es seguro instanciarlo.
        self.config_manager = ConfigManager()
        logger.info("Plugin BrowserControl inicializado.")

    def get_description(self) -> str:
        return "Abre el navegador web predeterminado o una URL específica."

    def can_handle(self, text: str, doc=None, context: dict = None, entities: list = None) -> bool:
        """
        Determina si este plugin puede manejar el comando de voz.
        Busca frases como "abrir navegador" o "open browser".
        """
        text_lower = text.lower()
        keywords = ["abrir navegador", "open browser", "abre el navegador"]
        # Podría ser más sofisticado usando el 'doc' de spaCy para lemas o intenciones.
        return any(keyword in text_lower for keyword in keywords)

    def handle(self, text: str, doc=None, context: dict = None, entities: list = None) -> str:
        """
        Maneja el comando para abrir el navegador.
        Abre la URL configurada en 'app_config.json' bajo la clave 'navegador_predeterminado_url'
        o una URL por defecto si no está configurada.
        """
        # Intenta obtener la URL del navegador desde la configuración.
        # La clave 'navegador_predeterminado_url' es un ejemplo, ajústala según tu app_config.json.
        default_url = "https://google.com"
        url_to_open = self.config_manager.get_app_setting("navegador_predeterminado_url", default_url)
        
        # Aquí se podría añadir lógica para extraer una URL específica del 'text' o 'entities'
        # Por ejemplo: "abrir navegador en example.com"

        try:
            logger.info(f"Abriendo navegador en la URL: {url_to_open}")
            webbrowser.open(url_to_open)
            return f"Navegador abierto en {url_to_open}."
        except Exception as e:
            logger.error(f"No se pudo abrir el navegador: {e}")
            return "Lo siento, ocurrió un error al intentar abrir el navegador."
