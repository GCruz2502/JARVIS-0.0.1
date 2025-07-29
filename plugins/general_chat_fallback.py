import logging
import random

logger = logging.getLogger(__name__)

class Plugin:
    def __init__(self):
        self.name = "GeneralChatFallback"
        self.description = "Provides general conversational responses when no specific intent is matched."

    def get_description(self):
        return self.description

    def handle(self, text, entities=None, sentiment=None, context=None, doc=None):
        logger.info(f"Handling general chat fallback for text: {text}")
        responses = [
            "Interesante. Cuéntame más.",
            "Entiendo. ¿Hay algo más en lo que pueda ayudarte?",
            "Gracias por compartir eso conmigo.",
            "Ya veo. Sigo aprendiendo y mejorando.",
            "Eso es algo en lo que estoy trabajando para entender mejor.",
            "A veces, las conversaciones generales son las más interesantes.",
            "Mi propósito es asistirte. ¿Hay alguna tarea específica que tengas en mente?",
            "Estoy aquí para escucharte."
        ]
        return {"response": random.choice(responses), "plugin_used": self.name}