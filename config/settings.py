import os
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.json")

# Usar correctamente las variables de entorno
# These should be the NAMES of the environment variables, not the keys themselves.
# The actual keys should be stored in a .env file or system environment variables.
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY")

# Configuración por defecto si no hay variables de entorno
if not OPENWEATHER_API_KEY:
    print("Warning: OPENWEATHER_API_KEY not set in enviorment.")         

if not NEWSAPI_API_KEY:
    print("Warning: NEWSAPI_API_KEY not set in enviorment.")
