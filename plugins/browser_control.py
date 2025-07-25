"""
Plugin para controlar el navegador web.
"""
import webbrowser
import logging
import re
from urllib.parse import quote_plus
from utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)

RESPONSE_TEXTS = {
    "es": {
        "description": "Abre una URL o busca en la web.",
        "opening_url": "Abriendo la página {url}.",
        "searching_for": "Buscando '{query}' en la web.",
        "url_not_found": "No pude encontrar una URL para abrir. Abriendo tu página de inicio.",
        "query_not_found": "No entendí qué buscar. Abriendo el buscador.",
        "open_error": "Lo siento, ocurrió un error al intentar abrir el navegador."
    },
    "en": {
        "description": "Opens a URL or searches the web.",
        "opening_url": "Opening the page {url}.",
        "searching_for": "Searching the web for '{query}'.",
        "url_not_found": "I couldn't find a URL to open. Opening your home page.",
        "query_not_found": "I didn't understand what to search for. Opening the search engine.",
        "open_error": "Sorry, an error occurred while trying to open the browser."
    }
}

# --- IMPROVED: More robust helper function ---
def _extract_search_query(text: str) -> str:
    """Removes common trigger words and phrases to get a clean search query."""
    text = text.lower() # Work with lowercase
    triggers = [
        "busca información sobre", "busca en la web", "busca en google", "busca",
        "googlea acerca de", "googlea",
        "search the web for", "search for", "search",
        "google about", "google"
    ]
    # Sort triggers by length, longest first, to avoid partial matches
    triggers.sort(key=len, reverse=True)

    for trigger in triggers:
        # Check if the text starts with the trigger phrase
        if text.startswith(trigger + " "):
            # Return the part of the string that comes after the trigger
            return text[len(trigger):].strip()
            
    # If no trigger phrase was found at the start, return the original text
    return text.strip()

class Plugin:
    def __init__(self):
        self.config_manager = ConfigManager()
        logger.info("Plugin BrowserControl inicializado.")

    def get_description(self) -> str:
        return f"{RESPONSE_TEXTS['es']['description']} / {RESPONSE_TEXTS['en']['description']}"

    def handle(self, text: str, doc=None, context: dict = None, entities: list = None) -> str:
        current_lang = context.get('current_conversation_lang', 'es') if context else 'es'
        specific_intent = context.get('recognized_intent_for_plugin', '')
        responses = RESPONSE_TEXTS[current_lang]
        
        action_message = ""
        url_to_open = ""

        if specific_intent == "INTENT_OPEN_URL":
            url_match = re.search(r"([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", text)
            if url_match:
                url_to_open = "http://" + url_match.group(1).lower()
                action_message = responses["opening_url"].format(url=url_match.group(1).lower())
            else:
                url_to_open = self.config_manager.get_app_setting("browser_default_url", "https://google.com")
                action_message = responses["url_not_found"]

        elif specific_intent == "INTENT_SEARCH_WEB":
            query = _extract_search_query(text)
            if query:
                encoded_query = quote_plus(query)
                search_url_template = self.config_manager.get_app_setting("browser_search_url_template", "https://www.google.com/search?q={query}")
                url_to_open = search_url_template.format(query=encoded_query)
                action_message = responses["searching_for"].format(query=query)
            else:
                url_to_open = self.config_manager.get_app_setting("browser_default_url", "https://google.com")
                action_message = responses["query_not_found"]
        
        else:
            logger.warning(f"BrowserControl plugin handled an unexpected intent: '{specific_intent}'")
            url_to_open = self.config_manager.get_app_setting("browser_default_url", "https://google.com")
            action_message = responses["url_not_found"]

        try:
            logger.info(f"{action_message} URL: {url_to_open}")
            webbrowser.open(url_to_open)
            return action_message
        except Exception as e:
            logger.error(f"Could not open browser: {e}")
            return responses["open_error"]