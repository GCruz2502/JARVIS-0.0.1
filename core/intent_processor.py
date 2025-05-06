# jarvis/core/intent_processor.py
import logging
import importlib
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

class IntentProcessor:
    def __init__(self):
        """Inicializa el procesador de intenciones y carga los plugins disponibles"""
        self.plugins = {}
        self.load_plugins()
        
    def load_plugins(self):
        """Carga dinámicamente los plugins desde el directorio plugins"""
        try:
            plugins_dir = Path(__file__).parent.parent / "plugins"
            plugin_files = [f for f in os.listdir(plugins_dir) if f.endswith('.py') and f != '__init__.py']
            
            for plugin_file in plugin_files:
                plugin_name = plugin_file[:-3]  # Quitar la extensión .py
                try:
                    module = importlib.import_module(f"jarvis.plugins.{plugin_name}")
                    if hasattr(module, 'Plugin'):
                        self.plugins[plugin_name] = module.Plugin()
                        logger.info(f"Plugin cargado: {plugin_name}")
                except Exception as e:
                    logger.error(f"Error al cargar plugin {plugin_name}: {str(e)}")
        except Exception as e:
            logger.error(f"Error al cargar plugins: {str(e)}")
    
    def process(self, text):
        """
        Procesa el texto para determinar la intención y ejecutar el plugin correspondiente
        
        Args:
            text (str): El texto del comando a procesar
            
        Returns:
            str: La respuesta del plugin o un mensaje de fallback
        """
        text = text.lower()
        
        # Verificar cada plugin para ver si puede manejar esta intención
        for name, plugin in self.plugins.items():
            if hasattr(plugin, 'can_handle') and plugin.can_handle(text):
                logger.info(f"Usando plugin {name} para procesar: '{text}'")
                return plugin.handle(text)
        
        # Si no hay plugin específico, intentemos buscar keywords comunes
        if any(word in text for word in ["tiempo", "clima", "lluvia", "temperatura"]):
            if "weather" in self.plugins:
                return self.plugins["weather"].handle(text)
        
        if any(word in text for word in ["noticia", "noticias", "actualidad"]):
            if "news" in self.plugins:
                return self.plugins["news"].handle(text)
        
        if any(word in text for word in ["recuerda", "recordatorio", "agenda", "evento"]):
            if "reminders" in self.plugins:
                return self.plugins["reminders"].handle(text)
        
        # Respuesta por defecto si no se encuentra un plugin adecuado
        logger.warning(f"No se encontró plugin para manejar: '{text}'")
        return "Lo siento, no sé cómo ayudarte con eso todavía."