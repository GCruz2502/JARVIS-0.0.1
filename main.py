"""
JARVIS Virtual Assistant
Punto de entrada principal de la aplicación.
"""
import logging # Para obtener el logger después de la configuración
from pathlib import Path

# Importar los componentes refactorizados
from utils.logger import setup_logging
from utils.config_manager import ConfigManager
from core.context_manager import ContextManager
from core.intent_processor import IntentProcessor
from ui.cli_interface import start_cli
from utils.database_handler import create_connection, create_table, collect_data  # Import database functions
import json
from datetime import datetime, timezone

# Obtener un logger para este módulo (después de que setup_logging haya sido llamado)
# Esto se hace aquí para que el logger esté disponible globalmente en este script si es necesario.
# Sin embargo, la configuración principal del logger se hace en setup_logging.
logger = logging.getLogger(__name__)

def main():
    """
    Función principal para inicializar y ejecutar JARVIS.
    """
    # 0. Initialize Database (Before anything else)
    try:
        conn = create_connection()
        if conn:
            create_table(conn)
            conn.close()
            logger.info("Database initialized successfully.")

            # Add some dummy data for testing
            conn = create_connection()
            if conn:
                try:
                    timestamp = datetime.now(timezone.utc).isoformat()
                    user_input = "Hola JARVIS"
                    intent = "greeting"
                    entities = json.dumps([{"text": "JARVIS", "type": "PERSON"}])
                    sentiment = json.dumps({"label": "NEU", "score": 0.8})
                    plugin_used = "IntentProcessorInternal"
                    response = "Hola! ¿Cómo puedo ayudarte hoy?"
                    success = 1
                    language = "es"

                    if collect_data(timestamp, user_input, intent, entities, sentiment, plugin_used, response, success, language):
                        logger.info("Dummy interaction data saved successfully.")
                    else:
                        logger.warning("Failed to save dummy interaction data.")
                except Exception as e:
                    logger.error(f"Error adding dummy data: {e}")
                finally:
                    conn.close()
    except Exception as e:
        logger.critical(f"Error initializing database: {e}", exc_info=True)
        print(f"Error crítico al iniciar la base de datos: {e}. JARVIS no puede continuar.")
        return

    # 1. Configurar el logging
    # Se puede hacer configurable a través de ConfigManager si se desea.
    # Por ejemplo, leer el nivel de log y la ruta del archivo de log desde app_config.json.
    # Por ahora, usamos valores predeterminados o simples.
    log_file = Path("jarvis_app.log") # Ejemplo de ruta de archivo de log
    setup_logging(log_level_str="INFO", log_to_console=True, log_file_path=log_file)

    logger.info("Iniciando JARVIS...")

    # 2. Inicializar el ConfigManager
    # Asume que .env está en la raíz del proyecto y data/app_config.json, data/user_data.json existen.
    try:
        config_manager = ConfigManager()
        logger.info("ConfigManager inicializado.")
    except Exception as e:
        logger.critical(f"Error crítico al inicializar ConfigManager: {e}", exc_info=True)
        print(f"Error crítico al iniciar la configuración: {e}. JARVIS no puede continuar.")
        return # Salir si la configuración falla

    # 3. Inicializar el ContextManager
    try:
        # La longitud máxima del historial también podría ser configurable
        max_hist_len = config_manager.get_app_setting("conversation_history_length", 10)
        context_manager = ContextManager(max_history_len=max_hist_len)
        logger.info("ContextManager inicializado.")
    except Exception as e:
        logger.critical(f"Error crítico al inicializar ContextManager: {e}", exc_info=True)
        print(f"Error crítico al iniciar el gestor de contexto: {e}. JARVIS no puede continuar.")
        return

    # 4. Inicializar el IntentProcessor
    # IntentProcessor carga modelos NLP y plugins en su __init__
    try:
        intent_processor = IntentProcessor(context_manager=context_manager, config_manager=config_manager)
        logger.info("IntentProcessor inicializado y plugins cargados.")
    except Exception as e:
        logger.critical(f"Error crítico al inicializar IntentProcessor: {e}", exc_info=True)
        print(f"Error crítico al iniciar el procesador de intenciones: {e}. JARVIS no puede continuar.")
        # Podrías querer que JARVIS funcione con funcionalidad limitada si NLP falla,
        # pero por ahora, saldremos.
        return

    # 5. Iniciar la interfaz de línea de comandos (CLI)
    try:
        start_cli(
            intent_processor=intent_processor,
            context_manager=context_manager,
            config_manager=config_manager
        )
    except KeyboardInterrupt:
        logger.info("JARVIS detenido por el usuario (KeyboardInterrupt en main).")
        # La CLI ya debería manejar el mensaje de despedida, pero podemos añadir uno aquí si es necesario.
        # from core.text_to_speech import hablar # Importar solo si es necesario aquí
        # hablar("Cerrando JARVIS.") 
        print("\nJARVIS cerrado.")
    except Exception as e:
        logger.critical(f"Error crítico no manejado en el bucle principal de la CLI: {e}", exc_info=True)
        # from core.text_to_speech import hablar
        # hablar("Se ha producido un error crítico en JARVIS. El sistema se cerrará.")
        print(f"Error crítico en JARVIS: {e}. El sistema se cerrará.")

    logger.info("JARVIS ha finalizado.")

if __name__ == "__main__":
    main()
