import webbrowser
import importlib
import json
import os
from dotenv import load_dotenv

load_dotenv(os.path.join("config", ".env"))

# Cargar configuraciones
def cargar_config(ruta):
    try:
        with open(ruta, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Archivo {ruta} no encontrado. Usando configuración por defecto.")
        return {}
    except json.JSONDecodeError:
        print(f"Error en el formato de {ruta}.")
        return {}
    
def cargar_plugins(comandos):
    for archivo in os.listdir("plugins"):
        if archivo.endswith(".py") and archivo != "__init__.py":
            nombre = archivo[:-3]
            modulo = importlib.import_module(f"plugins.{nombre}")
            if hasattr(modulo, "register"):
                modulo.register(comandos)

# Después de definir el diccionario 'comandos'
cargar_plugins('comandos')

config = cargar_config(os.path.join("config", "config.json"))
data = cargar_config(os.path.join("config", "data.json"))

# Comandos básicos
def abrir_navegador():
    webbrowser.open(config.get("navegador", "https://google.com"))

def reproducir_musica():
    webbrowser.open(config.get("spotify", "https://open.spotify.com"))

# Diccionario de comandos (¡expándelo aquí!)
comandos = {
    "abrir navegador": abrir_navegador,
    "reproducir música": reproducir_musica
}

def ejecutar_comando(texto):
    for comando, funcion in comandos.items():
        if comando in texto:
            funcion()
            return True
    return False