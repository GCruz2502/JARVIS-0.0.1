# plugins/reminders.py
import logging
from spacy.tokens import Doc

logger = logging.getLogger(__name__)

class Plugin:
    def __init__(self):
        logger.info("Reminders plugin initialized.")

    def can_handle(self, text: str, doc: Doc = None, context: dict = None, entities: list = None) -> bool:
        text_lower = text.lower()
        keywords = ["recuérdame", "recordatorio", "reminder", "remind me"]
        
        if any(keyword in text_lower for keyword in keywords):
            return True
        
        if doc:
            lemmas = ["recordar", "remind"] # Simple lemmas for now
            if any(token.lemma_ in lemmas for token in doc):
                return True
        return False

    def handle(self, text: str, doc: Doc = None, context: dict = None, entities: list = None) -> str:
        # Placeholder handler
        # In a real scenario, this would parse date/time/task from entities or text
        # and schedule a reminder.
        
        # For now, just acknowledge the reminder intent
        task_description = "tarea no especificada"
        if entities:
            # Attempt to find a task description, e.g., from a WORK_OF_ART or other entities
            # This is a very basic example
            non_time_date_phone_entities = [e['text'] for e in entities if e['label'] not in ["TIME", "DATE", "PHONE", "PER"]]
            if non_time_date_phone_entities:
                task_description = " ".join(non_time_date_phone_entities)
        
        logger.info(f"Handling reminder for: {task_description}")
        return f"Entendido, te recordaré sobre '{task_description}'."
