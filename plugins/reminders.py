# plugins/reminders.py
import logging
import re
import json
import os
from datetime import datetime
import dateparser

logger = logging.getLogger(__name__)

RESPONSE_TEXTS = {
    "es": {
        "description": "Gestiona recordatorios y alarmas.",
        "reminder_set": "Entendido. Te recordaré: '{task}' el {date} a las {time}.",
        "alarm_set": "Ok, alarma configurada para las {time}.",
        "cancel_success": "Ok, he cancelado tus recordatorios/alarmas pendientes.",
        "parse_error": "No pude entender la hora o la fecha para el recordatorio.",
        "past_time_error": "No puedo programar un recordatorio en el pasado.",
        "task_not_found": "No pude entender qué recordarte."
    },
    "en": {
        "description": "Manages reminders and alarms.",
        "reminder_set": "Okay. I'll remind you to: '{task}' on {date} at {time}.",
        "alarm_set": "Okay, alarm set for {time}.",
        "cancel_success": "Okay, I've cancelled your pending reminders/alarms.",
        "parse_error": "I couldn't understand the time or date for the reminder.",
        "past_time_error": "I can't schedule a reminder in the past.",
        "task_not_found": "I couldn't understand what to remind you about."
    }
}

class Plugin:
    """
    To integrate the reminder checking functionality, you should call the `check_reminders` method
    of the `Plugin` instance regularly from your main application loop.

    For example:

    ```python
    # In your main application file
    from plugins.reminders import Plugin as RemindersPlugin
    import time

    reminders_plugin = RemindersPlugin()

    while True:
        # Your main application logic here
        reminders_plugin.check_reminders()
        time.sleep(1) # Check every second
    ```
    """
    def __init__(self):
        self.active_reminders = []
        self.load_reminders()
        logger.info("Plugin Reminders inicializado.")

    def get_description(self) -> str:
        return f"{RESPONSE_TEXTS['es']['description']} / {RESPONSE_TEXTS['en']['description']}"

    def handle(self, text: str, doc=None, context: dict = None, entities: list = None) -> str:
        current_lang = context.get('current_conversation_lang', 'es') if context else 'es'
        specific_intent = context.get('recognized_intent_for_plugin', '')

        if specific_intent in ["INTENT_SET_REMINDER", "INTENT_SET_ALARM"]:
            return self._handle_set_reminder(text, doc, current_lang, specific_intent)
        elif specific_intent == "INTENT_CANCEL":
            return self._handle_cancel_reminders(current_lang)

        return "Internal error in reminder plugin."

    def _handle_set_reminder(self, text: str, doc, current_lang: str, specific_intent: str) -> str:
        responses = RESPONSE_TEXTS[current_lang]
        reminder_time, time_phrase = self._parse_time(doc, current_lang)

        if not reminder_time:
            logger.warning(f"All parsing strategies failed for: '{text}'")
            return responses["parse_error"]

        if reminder_time < datetime.now():
            return responses["past_time_error"]

        task = self._extract_task(text, time_phrase, current_lang)

        reminder = {"task": task, "time": reminder_time.isoformat(), "lang": current_lang}
        self.active_reminders.append(reminder)
        self.save_reminders()
        
        time_str = reminder_time.strftime("%I:%M %p" if current_lang == 'en' else "%H:%M")
        if specific_intent == "INTENT_SET_ALARM":
            return responses["alarm_set"].format(time=time_str)
        else:
            date_str = reminder_time.strftime("%x")
            return responses["reminder_set"].format(task=task, date=date_str, time=time_str)

    def _handle_cancel_reminders(self, current_lang: str) -> str:
        self.active_reminders.clear()
        self.save_reminders()
        logger.info("All pending reminders have been cancelled.")
        return RESPONSE_TEXTS[current_lang]["cancel_success"]

    def _parse_time(self, doc, current_lang: str) -> (datetime, str):
        time_ents = [ent for ent in doc.ents if ent.label_ in ['TIME', 'DATE']]
        if not time_ents:
            return None, ""

        start_char = min(ent.start_char for ent in time_ents)
        end_char = max(ent.end_char for ent in time_ents)
        time_phrase = doc.text[start_char:end_char]
        
        logger.info(f"Parsing time phrase: '{time_phrase}'")
        settings = {'PREFER_DATES_FROM': 'future'}
        reminder_time = dateparser.parse(time_phrase, languages=[current_lang], settings=settings)
        
        return reminder_time, time_phrase

    def _extract_task(self, text: str, time_phrase: str, current_lang: str) -> str:
        # The core idea is to remove the time phrase and the trigger phrase.
        # We need to be careful about the order of operations.
        
        # 1. Remove the time phrase first.
        task = text.replace(time_phrase, "").strip()

        # 2. Define trigger phrases for both languages.
        trigger_phrases_es = ["recuérdame que", "recuérdame", "avísame que", "avísame", "pon una alarma"]
        trigger_phrases_en = ["remind me to", "remind me", "set an alarm for"]
        
        # 3. Create a regex pattern to match any of the trigger phrases.
        if current_lang == 'es':
            pattern = r"^(" + "|".join(trigger_phrases_es) + r")\s+"
        else:
            pattern = r"^(" + "|".join(trigger_phrases_en) + r")\s+"
            
        # 4. Remove the trigger phrase from the beginning of the task.
        task = re.sub(pattern, "", task, flags=re.IGNORECASE).strip()
        
        # 5. Clean up any leading/trailing colons.
        task = task.strip(":")
        
        # 6. If the task is empty, provide a default.
        if not task:
            task = "tarea sin especificar" if current_lang == 'es' else "unspecified task"
            
        return task

    def load_reminders(self):
        if os.path.exists('reminders.json'):
            with open('reminders.json', 'r') as f:
                try:
                    self.active_reminders = json.load(f)
                except json.JSONDecodeError:
                    self.active_reminders = []
                    logger.error("Could not decode reminders from reminders.json")

    def save_reminders(self):
        with open('reminders.json', 'w') as f:
            json.dump(self.active_reminders, f)

    def check_reminders(self):
        now = datetime.now()
        due_reminders = [r for r in self.active_reminders if datetime.fromisoformat(r['time']) < now]
        for reminder in due_reminders:
            # This is where you would trigger the reminder.
            # For now, we'll just print to the console.
            print(f"REMINDER: {reminder['task']}")
            self.active_reminders.remove(reminder)
        self.save_reminders()