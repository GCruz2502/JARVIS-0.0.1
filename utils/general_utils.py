"""
Módulo de utilidades generales.
"""
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def saludo():
    """
    Devuelve un saludo apropiado según la hora del día.
    """
    hora = datetime.now().hour
    if 5 <= hora < 12:
        return "Buenos días"
    elif 12 <= hora < 19:
        return "Buenas tardes"
    else:
        return "Buenas noches"

def guardar_dato(ruta_archivo: str, clave: str, valor):
    """
    Guarda un par clave-valor en un archivo JSON.
    Si el archivo existe, actualiza o añade la clave.
    Si no existe, crea el archivo con la clave.

    Args:
        ruta_archivo (str): La ruta al archivo JSON.
        clave (str): La clave a guardar.
        valor: El valor asociado a la clave.
    """
    datos = {}
    try:
        # Intenta leer el archivo existente
        with open(ruta_archivo, "r", encoding='utf-8') as f:
            datos = json.load(f)
    except FileNotFoundError:
        logger.info(f"Archivo no encontrado en '{ruta_archivo}'. Se creará uno nuevo.")
    except json.JSONDecodeError:
        logger.warning(f"Error al decodificar JSON de '{ruta_archivo}'. El archivo será sobrescrito con los nuevos datos.")
        datos = {} # Asegura que 'datos' sea un dict si el archivo está corrupto
    except Exception as e:
        logger.error(f"Error inesperado al leer '{ruta_archivo}': {e}")
        # Dependiendo del comportamiento deseado, podrías querer retornar o levantar una excepción aquí.
        # Por ahora, se intentará continuar y escribir.

    datos[clave] = valor

    try:
        # Escribe los datos (actualizados o nuevos) al archivo
        with open(ruta_archivo, "w", encoding='utf-8') as f:
            json.dump(datos, f, indent=4, ensure_ascii=False)
        logger.info(f"Dato '{clave}' guardado/actualizado en '{ruta_archivo}'.")
    except Exception as e:
        logger.error(f"Error guardando datos en '{ruta_archivo}': {e}")
        # Considerar si se debe levantar una excepción aquí para notificar al llamador del fallo.
