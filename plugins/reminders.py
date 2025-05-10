"""
Plugin para gestionar recordatorios.
"""
import logging
import datetime
import threading
import time
import re # For basic parsing
from spacy.tokens import Doc # For type hinting
from core.text_to_speech import hablar # Import a central hablar function
from utils.config_manager import ConfigManager # For consistency, though not heavily used here yet

logger = logging.getLogger(__name__)

class Plugin:
    def __init__(self):
        self.config_manager = ConfigManager() # Standard plugin initialization
        logger.info("Plugin Reminders inicializado.")

    def get_description(self) -> str:
        return "Programa recordatorios. Ejemplo: 'recuérdame llamar a Juan mañana a las 10 am' o 'recordatorio comprar leche en 2 horas'."

    def _parse_datetime_from_entities(self, entities: list, text: str) -> tuple:
        """
        Intenta extraer una fecha y hora futuras de las entidades o texto.
        Retorna (datetime_obj, tarea_restante) o (None, texto_original).
        Esta es una implementación MUY BÁSICA y necesita una librería de parsing robusta.
        """
        # TODO: Implementar un parser de fecha/hora robusto (e.g., using parsedatetime, dateparser, or custom logic with spaCy DATE/TIME entities)
        # Por ahora, busca un formato específico YYYY-MM-DD HH:MM:SS o intenta algo muy simple.
        
        date_str, time_str = None, None
        task_parts = []

        # Intenta encontrar un formato específico primero
        datetime_match = re.search(r'(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})', text)
        if datetime_match:
            try:
                dt_obj = datetime.datetime.strptime(datetime_match.group(1), "%Y-%m-%d %H:%M:%S")
                # Asumir que el resto del texto es la tarea
                remaining_task = text.replace(datetime_match.group(1), "").strip()
                # Limpiar palabras clave comunes de recordatorio
                remaining_task = re.sub(r'^(recuérdame|recordatorio|reminder|remind me)\s*(que|de|sobre)?\s*', '', remaining_task, flags=re.IGNORECASE).strip()
                remaining_task = re.sub(r'\s*(a las|el día|en|para)\s*$', '', remaining_task, flags=re.IGNORECASE).strip()
                if remaining_task:
                    return dt_obj, remaining_task
            except ValueError:
                logger.debug("Se encontró un patrón similar a fecha/hora, pero no es válido.")

        # Si no, intenta usar entidades (esto es muy simplificado)
        # Una lógica real necesitaría convertir "mañana", "5 pm", etc., a datetime.
        # Esta parte es solo un placeholder para la idea.
        if entities:
            # Esta es una lógica de ejemplo y necesitaría ser mucho más robusta
            # Por ejemplo, manejar "mañana a las 5pm", "en 2 horas", "el próximo martes"
            # Aquí solo un ejemplo si se detecta una fecha y hora separadas y se pueden combinar.
            # Esto es altamente propenso a errores y solo ilustrativo.
            pass # Dejar para una implementación más robusta.

        logger.warning("No se pudo parsear una fecha/hora específica de la entrada para el recordatorio.")
        return None, text # Devuelve None si no se pudo parsear, y el texto original como tarea


    def _reminder_task(self, reminder_text: str, delay_seconds: float):
        """
        Tarea que se ejecuta en un hilo para activar el recordatorio.
        Usa la función central 'hablar'.
        """
        try:
            if delay_seconds < 0: # No debería ocurrir si la lógica de handle es correcta
                logger.warning(f"Intento de activar recordatorio con delay negativo: {reminder_text}")
                return

            logger.info(f"Hilo de recordatorio iniciado para '{reminder_text}'. Se activará en {delay_seconds:.2f} segundos.")
            time.sleep(delay_seconds)
            
            speak_text = f"¡Recordatorio! {reminder_text}"
            logger.info(f"Activando recordatorio: {speak_text}")
            hablar(speak_text) # Usar la función 'hablar' importada
        except Exception as e:
            logger.error(f"Error en el hilo del recordatorio para '{reminder_text}': {e}", exc_info=True)

    def can_handle(self, text: str, doc: Doc = None, context: dict = None, entities: list = None) -> bool:
        text_lower = text.lower()
        # Palabras clave expandidas basadas en el antiguo 'report_generator.py' y uso común
        keywords = ["recuérdame", "recordatorio", "reminder", "remind me", "agenda", "evento", "programar"]
        
        if any(keyword in text_lower for keyword in keywords):
            return True
        
        if doc: # Comprobar lemas si el documento spaCy está disponible
            lemmas = ["recordar", "remind", "agendar", "programar"]
            if any(token.lemma_ in lemmas for token in doc):
                return True
        return False

    def handle(self, text: str, doc: Doc = None, context: dict = None, entities: list = None) -> str:
        logger.info(f"Plugin de Recordatorios manejando: '{text}'")
        
        # Intentar extraer la tarea y la hora del recordatorio
        # Esta función _parse_datetime_from_entities necesita ser mucho más robusta.
        target_dt, task_description = self._parse_datetime_from_entities(entities, text)

        if not target_dt:
            # Si no se pudo parsear una fecha/hora específica, pedir más detalles.
            # O intentar una lógica de "recordar X en Y tiempo" (ej. "en 2 horas")
            # Por ahora, se devuelve un mensaje genérico.
            return "No pude determinar la hora para el recordatorio. Por favor, especifica cuándo (ej. 'mañana a las 10am' o 'en 30 minutos' o 'YYYY-MM-DD HH:MM:SS')."

        # Limpiar la descripción de la tarea si es el texto completo
        if task_description == text: # Si _parse_datetime_from_entities no pudo aislar la tarea
            task_description = re.sub(r'^(recuérdame|recordatorio|reminder|remind me)\s*(que|de|sobre)?\s*', '', task_description, flags=re.IGNORECASE).strip()
            # Quitar frases comunes de tiempo si están al final y no fueron parseadas
            task_description = re.sub(r'\s*(a las|el día|en|para)\s*[\w\s\d:-]+$', '', task_description, flags=re.IGNORECASE).strip()
            if not task_description: task_description = "recordatorio" # Fallback

        try:
            current_time = datetime.datetime.now()
            time_diff_seconds = (target_dt - current_time).total_seconds()
            
            if time_diff_seconds > 0:
                reminder_thread = threading.Thread(
                    target=self._reminder_task,
                    args=(task_description, time_diff_seconds),
                    daemon=True 
                )
                reminder_thread.start()
                
                # Formatear target_dt para el mensaje de confirmación
                time_str_format = "%Y-%m-%d a las %H:%M:%S"
                user_friendly_time = target_dt.strftime(time_str_format)
                
                logger.info(f"Recordatorio programado para '{task_description}' el {user_friendly_time}.")
                return f"Entendido. Te recordaré sobre '{task_description}' el {user_friendly_time}."
            else:
                logger.warning(f"Intento de programar recordatorio en el pasado: {target_dt.strftime('%Y-%m-%d %H:%M:%S')} para '{task_description}'.")
                return "La hora especificada para el recordatorio ya ha pasado. Por favor, elige una hora futura."
                
        except ValueError: # Esto podría ocurrir si strptime falla, aunque _parse_datetime_from_entities debería manejarlo
            logger.error(f"Error de formato de fecha/hora al procesar el recordatorio.")
            return "Hubo un error con el formato de la fecha/hora para el recordatorio."
        except Exception as e:
            logger.error(f"Error al programar recordatorio para '{task_description}': {e}", exc_info=True)
            return f"Lo siento, ocurrió un error al intentar programar el recordatorio."
