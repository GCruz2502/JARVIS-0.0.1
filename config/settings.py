import os
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.json")

# Usar correctamente las variables de entorno
OPENWEATHER_API_KEY = os.getenv("0772b676ef44deb64c03aac472b12b48")
NEWSAPI_API_KEY = os.getenv("3a38ee3e0b504ec68460b7ef5a78bc93")

# Configuración por defecto si no hay variables de entorno
if not OPENWEATHER_API_KEY:
    print("Warning: OPENWEATHER_API_KEY not set in enviorment.")         

if not NEWSAPI_API_KEY:
    print("Warning: NEWSAPI_API_KEY not set in enviorment.")
