"""
Módulo para manejar la entrada de voz (Speech Recognition) y fallback de texto.
"""
import speech_recognition as sr
import logging
# import os # No se usa directamente en las funciones migradas
# from dotenv import load_dotenv # No se usa directamente en las funciones migradas

# La importación y uso de Vosk Model y KaldiRecognizer dependerá de si se implementa
# una función de escucha específica para Vosk. El modelo 'modelo-es'
# se inicializaba en el archivo original pero no se usaba en 'escuchar()'.
# from vosk import Model, KaldiRecognizer 

logger = logging.getLogger(__name__)

# Si se decide usar un modelo Vosk globalmente, su inicialización iría aquí.
# try:
#     vosk_model = Model("modelo-es") # Asegúrate que la ruta al modelo es correcta
# except Exception as e:
#     logger.error(f"Error al cargar el modelo Vosk: {e}")
#     vosk_model = None

def escuchar(timeout=5, phrase_time_limit=5):
    """
    Escucha el micrófono y devuelve el texto reconocido usando Google Speech Recognition.
    
    Args:
        timeout (int): Tiempo máximo de espera para comenzar a hablar.
        phrase_time_limit (int): Tiempo máximo para una frase.

    Returns:
        str or None: El texto reconocido o None si hay un error.
    """
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            # print("Escuchando...") # Se puede manejar en la UI/CLI principal
            logger.info("Ajustando para ruido ambiente y escuchando...")
            recognizer.adjust_for_ambient_noise(source) 
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)

        logger.info("Procesando audio...")
        # print("Procesando...") # Se puede manejar en la UI/CLI principal
        texto = recognizer.recognize_google(audio, language="es-ES")
        logger.info(f"Texto reconocido: {texto}")
        # print(f"Has dicho: {texto}") # Se puede manejar en la UI/CLI principal
        return texto.lower()
    
    except sr.WaitTimeoutError:
        logger.warning("Timeout esperando el habla.")
        # print("Tiempo de espera agotado. Por favor, habla después de escuchar 'Escuchando...'")
    except sr.UnknownValueError:
        logger.info("No se pudo entender el audio.")
        # print("No he podido entender lo que has dicho.")
    except sr.RequestError as e:
        logger.error(f"Error con el servicio de reconocimiento de voz (Google): {e}")
        # print(f"Error con el servicio de reconocimiento de voz: {e}")
    except Exception as e:
        logger.error(f"Error inesperado en reconocimiento de voz: {e}")
        # print(f"Error inesperado en reconocimiento de voz: {e}")

    return None

# def escuchar_vosk(timeout=5, phrase_time_limit=5):
# """Función de ejemplo si se quisiera usar Vosk"""
# if vosk_model is None:
# logger.error("Modelo Vosk no cargado.")
# return None
# recognizer = KaldiRecognizer(vosk_model, 16000)
#     # ... Lógica para escuchar con Vosk ...
#     pass

def entrada_texto():
    """
    Solicita entrada de texto como alternativa a la voz.
      
    Returns:
        str: El texto ingresado por el usuario, o una cadena vacía en caso de error.
    """
    try:
        # print("Escribe tu comando: ") # Se puede manejar en la UI/CLI principal
        comando = input() # Prompt se manejará en la UI que llama a esta función
        return comando.lower()
    except EOFError:
        logger.warning("EOFError al leer la entrada de texto (puede ser normal si se interrumpe la entrada).")
        return "" # O None, dependiendo de cómo se quiera manejar
    except Exception as e:
        logger.error(f"Error al obtener entrada de texto: {e}")
        # print(f"Error al obtener entrada de texto: {e}")
        return ""
