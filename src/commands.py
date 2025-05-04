"""
Módulo para manejar comandos del asitente virtual
"""
import webbrowser
import importlib
import json
import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv(os.path.join("config", ".env"))

# Dicionario de comandos
comandos = {}

def cargar_config(ruta):
    """
    Carga configuración desde un archivo JSON
    
    Args:
        ruta (str): Ruta al archivo de configuración 

    Returns:
        dict: Configuración cargada o Diccionario vacío si hay un error
    """
    try:
        with open(ruta, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Archivo {ruta} no encontrado. Usando configuración por defecto.")
        return {}
    except json.JSONDecodeError:
        logger.error(f"Error en el formato de {ruta}.")
        return {}
    
def cargar_plugins():
    """carga plugins dinámicamente desde la carpeta de plugins"""
    if not os.path.exists("plugins"):
        logger.warning("Direcotrio de plugins no encontrado")
        return
    
    for archivo in os.listdir("plugins"):
        if archivo.endswith(".py") and archivo != "__init__.py":
            try:
                nombre = archivo[:-3]
                modulo = importlib.import_module(f"plugins.{nombre}")
                if hasattr(modulo, "register"):
                    modulo.register(comandos)
                    logger.info(f"Plugin cargando: {nombre}")
            except Exception as e:
                logger.error(f"Error al cargar plugin {nombre}: {e}")

# Cargar configuraciones
config = cargar_config(os.path.join("config", "config.json"))
data = cargar_config(os.path.join("config", "data.json"))

# Comandos básicos
def abrir_navegador():
    """Abre el navegador web configurado"""
    url = config.get("navegador", "https://google.com")
    logger.info(f"Abriendo navegador: {url}")
    webbrowser.open(url)
    return "Abriendo navegador"

def reproducir_musica():
    """Abre el servicio de música configurado"""
    url = config.get("spotify", "https://open.spotify.com")
    logger.info(f"Reproduciendo música: {url}")
    webbrowser.open(url)
    return "Reproduciendo música"

def mostrar_ayuda():
    """Muestra la lista de los comandos disponibles"""
    lista_comandos = ", ".join(comandos.keys())
    return f"Comandos disponibles: {lista_comandos}"

# Añadir comandos básicos al diccionario
comandos.update({
    "abrir navegador": abrir_navegador,
    "reproducir música": reproducir_musica,
    "ayuda": mostrar_ayuda
})

# Cargar plugins automanticmaente
cargar_plugins()

def cargar_comandos(texto):
    """
    Ejecuta un comando basado en el texto recibido

    Args:
        texto (str): El texto del comando a ejecutar

    Returns:
        bool: True si se ejecutó el comando, False en caso contrario
    """
    for comando, funcion in comandos.items():
        if comando in texto:
            try:
                resultado = funcion()
                if isinstance(resultado, str):
                    from src.voice import hablar
                    hablar(resultado)
                logger.info(f"Comando ejecutado: {comando}")
                return True
            except Exception as e:
                logger.error(f"Error al ejecutar el comando {comando}: {e}")
                from src.voice import hablar
                hablar("Error al ejecutar {comando}")
                return True

    return False