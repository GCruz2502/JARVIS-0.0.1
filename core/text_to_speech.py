"""
Módulo para manejar la salida de voz (Text-to-Speech)
"""
import pyttsx3
import logging

logger = logging.getLogger(__name__)

# Iniciar motor de voz
engine = None
try:
    engine = pyttsx3.init()
    if engine:
        engine.setProperty("rate", 150)  # Velocidad del habla
    else:
        logger.error("Falló la inicialización del motor pyttsx3 (engine is None).")
except Exception as e:
    logger.error(f"Error al inicializar el motor de Text-to-Speech: {e}")
    # engine se mantendrá como None

def hablar(texto):
    """
    Convierte texto a voz
    
    Args:
        texto (str): El texto que se convertirá a voz
    """
    if engine is None:
        logger.error("Motor de Text-to-Speech no inicializado. No se puede hablar.")
        # Considerar notificar al usuario de una manera más visible si es crítico
        # print("Error: El servicio de voz para hablar no está disponible.")
        return

    try:
        logger.info(f"Hablando: {texto}")
        engine.say(texto)
        engine.runAndWait()
    except Exception as e:
        logger.error(f"Error al hablar: {e}")
        # Considerar notificar al usuario
        # print(f"Error al intentar hablar: {e}")
