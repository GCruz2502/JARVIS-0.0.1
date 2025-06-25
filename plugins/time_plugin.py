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
            now = datetime.date.now()
            current_lang = "es" # Default
            if context and context.get("current_conversation_lang") == "en": # Check context from CLI
                current_lang = "en"

            # +++ CORRECTED CODE FOR TIME +++
            time_str = now.strftime("%I:%M %p") # e.g., 04:45 PM
            if current_lang == "es":
                response = f"Son las {time_str}."
            else: # English
                response = f"The current time is {time_str}."
            # +++++++++++++++++++++++++++++++

            logger.info(f"Respondiendo con la hora actual: {response}")
            return response
        except Exception as e:
            logger.error(f"Error al obtener la hora: {e}", exc_info=True)
            return "Lo siento, no pude obtener la hora en este momento."