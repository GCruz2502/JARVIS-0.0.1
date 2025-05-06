"""
JARVIS Virtual Assistant
Main entry point for the aplication
"""
import sys
import os
import logging
from dotenv import load_dotenv

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="JARVIS.log",
)
logger=logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

# Importar componentes del núcleo
from jarvis.core.speech_recognition import SpeechRecognizer
from jarvis.core.text_to_speech import TextToSpeech
from jarvis.core.intent_processor import IntentProcessor

class Jarvis:
    def __init__(self):
        """Inicializa el asistente JARVIS con sus componentes principales"""
        logger.info("Inicializando JARVIS...")
        try:
            self.speech_recognizer = SpeechRecognizer()
            self.tts = TextToSpeech()
            self.intent_processor = IntentProcessor()
        except Exception as e:
            logger.error(f"Error al inicializar JARVIS: {str(e)}")
            raise
        
    def listen(self):
        """Escucha al usuario y devuelve el texto reconocido"""
        try:
            logger.info("Escuchando...")
            self.tts.speak("Te escucho")
            text = self.speech_recognizer.recognize_speech()
            logger.info(f"Reconocido: {text}")
            return text
        except Exception as e:
            logger.error(f"Error al escuchar: {str(e)}")
            self.tts.speak("Lo siento, no pude entenderte")
            return None
      
    def process_command(self, text):
        """Procesa el comando de texto y ejecuta la acción correspondiente"""
        if not text:
            return
        
        try:
            logger.info(f"Procesando comando: {text}")
            response = self.intent_processor.process(text)
            logger.info(f"Respuesta: {response}")
            self.tts.speak(response)
        except Exception as e:
            logger.error(f"Error al procesar el comando: {str(e)}")
            self.tts.speak("Lo siento, no pude procesar ese comando")

    def run(self):
        """Ejecuta el ciclo principal del asistente"""
        logger.info("Iniciando el ciclo principal de JARVIS")
        self.tts.speak("Hola, soy JARVIS. ¿En qué puedo ayudarte?")

        try:
            while True:
                text = self.listen()
                if text and "adios" in text.lower():
                    self.tts.speak("Hasta luego!")
                    break
                self.process_command(text)
        except KeyboardInterrupt:
            logger.info("Deteniendo JARVIS por interrupción del usuario")
            self.tts.speak("Cerrando sistemas. Hasta pronto")
        except Exception as e:
            logger.critical(f"Error crítico en JARVIS: {str(e)}")
            self.tts.speak("Se ha producido un error crítico. Reiniciando sistemas")



if __name__ == "__main__":
    assistant = Jarvis()
    assistant.run()