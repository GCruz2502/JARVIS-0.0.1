# jarvis/core/intent_processor.py
import logging
import importlib
import os
import re
import speech_recognition as sr
from pathlib import Path
import inspect # Added for signature checking

logger = logging.getLogger(__name__)

class IntentProcessor:
    def __init__(self):
        """Inicializa el procesador de intenciones, carga plugins y contexto."""
        self.plugins = {}
        self.context = {}  # Initialize conversation context
        self.load_plugins()

    def clear_context(self):
        """Clears the conversation context."""
        self.context = {}
        logger.info("Contexto de conversación limpiado.")

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
        Procesa el texto para determinar la intención, actualiza el contexto 
        y ejecuta el plugin correspondiente.

        Args:
            text (str): El texto del comando a procesar.

        Returns:
            str: La respuesta del plugin o un mensaje de fallback. 
                 La gestión del contexto (pasarlo y actualizarlo) se hará
                 en el bucle principal que llama a este método.
        """
        processed_text = text.lower()
        response = None
        plugin_used = None

        # --- Context Management Example (Simple: Clear context on 'clear context' command) ---
        # More sophisticated context logic will be needed.
        if "limpiar contexto" in processed_text or "olvida todo" in processed_text:
             self.clear_context()
             return "Entendido, he limpiado el contexto de nuestra conversación."
        # --- End Context Management Example ---

        # Verificar cada plugin para ver si puede manejar esta intención
        # Pass context to can_handle and handle methods
        for name, plugin in self.plugins.items():
            try:
                # Check if plugin methods accept context using inspect
                can_handle_sig = inspect.signature(plugin.can_handle)
                handle_sig = inspect.signature(plugin.handle)
                
                can_handle_accepts_context = 'context' in can_handle_sig.parameters
                handle_accepts_context = 'context' in handle_sig.parameters

                # Prepare arguments for can_handle based on signature
                can_handle_args = (processed_text, self.context) if can_handle_accepts_context else (processed_text,)
                
                if hasattr(plugin, 'can_handle') and plugin.can_handle(*can_handle_args):
                    logger.info(f"Plugin '{name}' can handle: '{text}'. Context: {self.context}")
                    plugin_used = name
                    
                    # Prepare arguments for handle based on signature
                    handle_args = (processed_text, self.context) if handle_accepts_context else (processed_text,)
                    
                    # Plugin's handle method might return just response, or (response, updated_context)
                    handle_result = plugin.handle(*handle_args) 
                    
                    if isinstance(handle_result, tuple) and len(handle_result) == 2:
                        response, updated_context = handle_result
                        if isinstance(updated_context, dict):
                            self.context.update(updated_context) # Update processor's context
                            logger.info(f"Plugin '{name}' updated context: {updated_context}")
                        else:
                             logger.warning(f"Plugin '{name}' returned non-dict context: {updated_context}")
                    else:
                        response = handle_result # Assume only response was returned

                    break # Stop after finding the first handling plugin
            except Exception as e:
                 logger.error(f"Error checking or handling plugin {name} for text '{text}': {e}", exc_info=True)


        # Fallback keyword matching if no plugin handled explicitly via can_handle
        if response is None:
            logger.debug(f"No plugin handled '{text}' via can_handle. Trying fallback keywords.")
            # Pass context to fallback handlers too
            if any(word in processed_text for word in ["tiempo", "clima", "lluvia", "temperatura"]):
                if "weather" in self.plugins:
                    try:
                        plugin = self.plugins["weather"]
                        handle_sig = inspect.signature(plugin.handle)
                        handle_accepts_context = 'context' in handle_sig.parameters
                        handle_args = (processed_text, self.context) if handle_accepts_context else (processed_text,)
                        handle_result = plugin.handle(*handle_args)
                        if isinstance(handle_result, tuple) and len(handle_result) == 2:
                            response, updated_context = handle_result
                            if isinstance(updated_context, dict):
                                self.context.update(updated_context)
                        else:
                            response = handle_result
                        plugin_used = "weather (fallback)"
                    except Exception as e:
                        logger.error(f"Error in fallback handling for weather plugin: {e}", exc_info=True)

            # Add similar fallback checks for news, reminders etc. if needed, passing context
            # Example for a hypothetical 'news' plugin:
            elif any(word in processed_text for word in ["noticia", "noticias", "actualidad"]):
                 if "news" in self.plugins:
                    try:
                        plugin = self.plugins["news"]
                        handle_sig = inspect.signature(plugin.handle)
                        handle_accepts_context = 'context' in handle_sig.parameters
                        handle_args = (processed_text, self.context) if handle_accepts_context else (processed_text,)
                        handle_result = plugin.handle(*handle_args)
                        if isinstance(handle_result, tuple) and len(handle_result) == 2:
                            response, updated_context = handle_result
                            if isinstance(updated_context, dict):
                                self.context.update(updated_context)
                        else:
                            response = handle_result
                        plugin_used = "news (fallback)"
                    except Exception as e:
                        logger.error(f"Error in fallback handling for news plugin: {e}", exc_info=True)

            # Example for a hypothetical 'reminders' plugin:
            elif any(word in processed_text for word in ["recuerda", "recordatorio", "agenda", "evento"]):
                 if "reminders" in self.plugins:
                    try:
                        plugin = self.plugins["reminders"]
                        handle_sig = inspect.signature(plugin.handle)
                        handle_accepts_context = 'context' in handle_sig.parameters
                        handle_args = (processed_text, self.context) if handle_accepts_context else (processed_text,)
                        handle_result = plugin.handle(*handle_args)
                        if isinstance(handle_result, tuple) and len(handle_result) == 2:
                            response, updated_context = handle_result
                            if isinstance(updated_context, dict):
                                self.context.update(updated_context)
                        else:
                            response = handle_result
                        plugin_used = "reminders (fallback)"
                    except Exception as e:
                        logger.error(f"Error in fallback handling for reminders plugin: {e}", exc_info=True)


        # Final response generation
        if response is not None:
            logger.info(f"Plugin '{plugin_used}' generated response for '{text}'. Final Context: {self.context}")
            return response
        else:
            # Default response if no plugin or fallback worked
            logger.warning(f"No plugin or fallback found for: '{text}'. Context: {self.context}")
            return "Lo siento, no sé cómo ayudarte con eso todavía."
