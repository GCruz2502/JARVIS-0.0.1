"""
Módulo para la configuración centralizada del logging.
"""
import logging
import sys
from logging.handlers import RotatingFileHandler # Optional: for rotating log files
from pathlib import Path

DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

def setup_logging(
    log_level_str: str = "INFO", 
    log_to_console: bool = True, 
    log_file_path: Path = None,
    log_file_max_bytes: int = 10*1024*1024, # 10 MB
    log_file_backup_count: int = 3
):
    """
    Configura el logging para la aplicación.

    Args:
        log_level_str (str): Nivel de logging (e.g., "DEBUG", "INFO", "WARNING", "ERROR").
        log_to_console (bool): Si es True, los logs se envían a la consola.
        log_file_path (Path, optional): Ruta al archivo de log. Si se proporciona, los logs se guardan en archivo.
        log_file_max_bytes (int): Tamaño máximo del archivo de log antes de rotar.
        log_file_backup_count (int): Número de archivos de log de respaldo a mantener.
    """
    numeric_level = getattr(logging, log_level_str.upper(), None)
    if not isinstance(numeric_level, int):
        logging.warning(f"Nivel de log inválido: {log_level_str}. Usando INFO por defecto.")
        numeric_level = logging.INFO

    # Configurar el logger raíz
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    formatter = logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)

    # Limpiar handlers existentes para evitar duplicación si se llama múltiples veces
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    # Handler para la consola
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Handler para archivo (con rotación)
    if log_file_path:
        try:
            log_file_path.parent.mkdir(parents=True, exist_ok=True) # Asegurar que el directorio exista
            # Usar RotatingFileHandler para evitar que el archivo de log crezca indefinidamente
            file_handler = RotatingFileHandler(
                log_file_path, 
                maxBytes=log_file_max_bytes, 
                backupCount=log_file_backup_count,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            root_logger.info(f"Logging configurado para escribir en archivo: {log_file_path}")
        except Exception as e:
            # Si falla el logging a archivo, al menos el de consola (si está activo) debería funcionar.
            root_logger.error(f"No se pudo configurar el logging a archivo en '{log_file_path}': {e}", exc_info=True)

    root_logger.info(f"Logging configurado. Nivel: {log_level_str.upper()}. Consola: {log_to_console}. Archivo: {log_file_path if log_file_path else 'No'}.")

# Cómo usar en otros módulos:
# import logging
# logger = logging.getLogger(__name__) # Esto obtiene un logger específico para el módulo actual
#
# # En main.py (o al inicio de la app):
# from utils.logger import setup_logging
# from pathlib import Path
# setup_logging(log_level_str="DEBUG", log_file_path=Path("jarvis_app.log"))
#
# logger.debug("Este es un mensaje de debug.")
# logger.info("Este es un mensaje informativo.")
# logger.warning("Esto es una advertencia.")
# logger.error("Esto es un error.")
# logger.critical("Esto es un error crítico.")
