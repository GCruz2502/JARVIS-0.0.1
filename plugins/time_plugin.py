# plugins/time_plugin.py
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# --- NEW: Centralized, bilingual text for all responses ---
RESPONSE_TEXTS = {
    "es": {
        "description": "Dice la hora actual.",
        "report_time": "Son las {time}.",
        "error": "Lo siento, no pude obtener la hora en este momento."
    },
    "en": {
        "description": "Tells the current time.",
        "report_time": "The current time is {time}.",
        "error": "Sorry, I was unable to get the time right now."
    }
}
# --- END NEW ---

class Plugin:
    def __init__(self):
        # The filename is time_plugin.py, so the log message should match.
        logger.info("Plugin TimePlugin initialized.")

    def get_description(self) -> str:
        return f"{RESPONSE_TEXTS['es']['description']} / {RESPONSE_TEXTS['en']['description']}"

    def handle(self, text: str, doc=None, context: dict = None, entities: list = None) -> str:
        current_lang = context.get('current_conversation_lang', 'es') if context else 'es'
        responses = RESPONSE_TEXTS[current_lang]

        try:
            now = datetime.now()
            # Use the same logic you had: 12-hour for English, 24-hour for Spanish
            time_format = "%I:%M %p" if current_lang == 'en' else "%H:%M"
            current_time_str = now.strftime(time_format)
            
            logger.info(f"Responding with the current time: {current_time_str}")
            return responses["report_time"].format(time=current_time_str)
            
        except Exception as e:
            logger.error(f"Error getting the current time: {e}", exc_info=True)
            return responses["error"]