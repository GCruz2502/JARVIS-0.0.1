# plugins/date_plugin.py
import logging
from datetime import date

logger = logging.getLogger(__name__)

# --- NEW: Centralized, bilingual text for all responses ---
RESPONSE_TEXTS = {
    "es": {
        "description": "Dice la fecha actual.",
        "report_date": "Hoy es {date_str}.",
        "error": "Lo siento, no pude obtener la fecha en este momento."
    },
    "en": {
        "description": "Tells the current date.",
        "report_date": "Today is {date_str}.",
        "error": "Sorry, I was unable to get the date right now."
    }
}
# --- END NEW ---

class Plugin:
    def __init__(self):
        logger.info("Plugin DatePlugin initialized.")

    def get_description(self) -> str:
        return f"{RESPONSE_TEXTS['es']['description']} / {RESPONSE_TEXTS['en']['description']}"

    def handle(self, text: str, doc=None, context: dict = None, entities: list = None) -> str:
        current_lang = context.get('current_conversation_lang', 'es') if context else 'es'
        responses = RESPONSE_TEXTS[current_lang]

        try:
            today = date.today()
            date_str = ""

            # --- Your excellent localization logic, now inside the new structure ---
            if current_lang == "es":
                day_names_es = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
                month_names_es = ["enero", "febrero", "marzo", "abril", "mayo", "junio", 
                                  "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
                
                day_name = day_names_es[today.weekday()]
                month_name = month_names_es[today.month - 1]
                date_str = f"{day_name}, {today.day} de {month_name} de {today.year}"
            else: # English
                date_str = today.strftime("%A, %B %d, %Y")
            
            logger.info(f"Responding with the current date: {date_str}")
            return responses["report_date"].format(date_str=date_str)
            
        except Exception as e:
            logger.error(f"Error getting the current date: {e}", exc_info=True)
            return responses["error"]