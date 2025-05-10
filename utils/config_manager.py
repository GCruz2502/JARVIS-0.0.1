"""
Módulo para la gestión de configuraciones de la aplicación.
"""
import json
import os
import logging
from dotenv import load_dotenv
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfigManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, project_root_dir: Path = None):
        # Prevenir reinicialización si ya existe la instancia (Singleton)
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        if project_root_dir is None:
            # Asume que este archivo está en My_Project/utils/
            self.project_root = Path(__file__).resolve().parent.parent
        else:
            self.project_root = project_root_dir

        self.env_path = self.project_root / ".env"
        # Estos nombres de archivo son ejemplos; ajústalos si es necesario.
        self.app_config_path = self.project_root / "data" / "app_config.json"
        self.user_data_path = self.project_root / "data" / "user_data.json"

        self._load_env()
        self.app_config = self._load_json_config(self.app_config_path, "Application Configuration")
        self.user_data = self._load_json_config(self.user_data_path, "User Data")
        
        self._initialized = True
        logger.info("ConfigManager inicializado.")

    def _load_env(self):
        """Carga variables de entorno desde el archivo .env en la raíz del proyecto."""
        if self.env_path.exists():
            load_dotenv(dotenv_path=self.env_path, override=True)
            logger.info(f"Variables de entorno cargadas desde: {self.env_path}")
        else:
            logger.warning(f"Archivo .env no encontrado en {self.env_path}. "
                           "Se dependerá de variables de entorno ya existentes en el sistema.")

    def _load_json_config(self, path: Path, config_name: str) -> dict:
        """Carga configuración desde un archivo JSON."""
        try:
            with open(path, "r", encoding='utf-8') as f:
                config = json.load(f)
                logger.info(f"'{config_name}' cargado exitosamente desde '{path}'.")
                return config
        except FileNotFoundError:
            logger.warning(f"Archivo de '{config_name}' no encontrado en '{path}'. Se retorna configuración vacía.")
            return {}
        except json.JSONDecodeError:
            logger.error(f"Error decodificando JSON del archivo '{config_name}' en '{path}'. Se retorna configuración vacía.")
            return {}
        except Exception as e:
            logger.error(f"Error inesperado cargando '{config_name}' desde '{path}': {e}. Se retorna configuración vacía.")
            return {}

    def get_env_variable(self, var_name: str, default=None):
        """Obtiene una variable de entorno."""
        return os.getenv(var_name, default)

    def get_app_setting(self, key: str, default=None):
        """Obtiene un ajuste de la configuración de la aplicación (app_config.json)."""
        return self.app_config.get(key, default)

    def get_user_data(self, key: str, default=None):
        """Obtiene un dato del archivo de datos de usuario (user_data.json)."""
        return self.user_data.get(key, default)

    def save_user_data(self, key: str, value) -> bool:
        """Guarda un par clave-valor en el archivo de datos de usuario (user_data.json)."""
        self.user_data[key] = value
        try:
            # Asegurarse que el directorio data exista
            self.user_data_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.user_data_path, "w", encoding='utf-8') as f:
                json.dump(self.user_data, f, indent=4, ensure_ascii=False)
            logger.info(f"Dato de usuario '{key}' guardado en '{self.user_data_path}'.")
            return True
        except Exception as e:
            logger.error(f"Error guardando dato de usuario en '{self.user_data_path}': {e}")
            return False

# Instancia global para fácil acceso (opcional, pero común para gestores de config)
# config_manager = ConfigManager()

# También se puede mantener una función similar a la original si se necesita cargar JSONs arbitrarios:
def load_json_file(file_path: str, description: str = "JSON file") -> dict:
    """Carga un archivo JSON desde una ruta específica."""
    path_obj = Path(file_path)
    try:
        with open(path_obj, "r", encoding='utf-8') as f:
            data = json.load(f)
            logger.info(f"'{description}' cargado desde '{path_obj}'.")
            return data
    except FileNotFoundError:
        logger.warning(f"Archivo '{description}' no encontrado en '{path_obj}'.")
        return {}
    except json.JSONDecodeError:
        logger.error(f"Error decodificando JSON de '{description}' en '{path_obj}'.")
        return {}
    except Exception as e:
        logger.error(f"Error cargando '{description}' desde '{path_obj}': {e}.")
        return {}
