# plugins/date_plugin.py
import logging
import datetime

logger = logging.getLogger(__name__)

class Plugin:
    def __init__(self):
        """Initializes the date plugin."""
        logger.info("Date plugin initialized.")

    def get_description(self) -> str:
        """Returns a brief description of the plugin's capabilities."""
        return "Dice la fecha actual."

    def can_handle(self, text: str, doc=None, context: dict = None, entities: list = None) -> bool:
        # We will rely on INTENT_GET_DATE from the intent map
        return False

    def handle(self, text: str, doc=None, context: dict = None, entities: list = None) -> str:
        """
        Handles the request to get the current date.
        """
        try:
            today = datetime.date.today()
            current_lang = "es" # Default
            if context and context.get("current_conversation_lang") == "en": # Check context from CLI
                current_lang = "en"

            if current_lang == "es":
                day_names_es = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
                month_names_es = ["enero", "febrero", "marzo", "abril", "mayo", "junio", 
                                  "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
                
                day_name = day_names_es[today.weekday()]
                month_name = month_names_es[today.month - 1]
                date_str = f"{day_name}, {today.day} de {month_name} de {today.year}"
                response = f"Hoy es {date_str}."
            else: # English
                date_str = today.strftime("%A, %B %d, %Y") 
                response = f"Today is {date_str}."
            
            logger.info(f"Respondiendo con la fecha actual: {response}")
            return response
        except Exception as e:
            logger.error(f"Error al obtener la fecha: {e}", exc_info=True)
            return "Lo siento, no pude obtener la fecha en este momento."