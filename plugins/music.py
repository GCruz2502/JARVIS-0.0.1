# plugins/music.py
import logging
import re
import webbrowser
from utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)

RESPONSE_TEXTS = {
    "es": {
        "description": "Reproduce música o busca canciones/artistas en Spotify.",
        "searching_for": "Buscando '{query}' en Spotify.",
        "opening_spotify": "Abriendo Spotify.",
        "open_error": "Lo siento, no pude abrir Spotify en este momento."
    },
    "en": {
        "description": "Plays music or searches for songs/artists on Spotify.",
        "searching_for": "Searching for '{query}' on Spotify.",
        "opening_spotify": "Opening Spotify.",
        "open_error": "Sorry, I couldn't open Spotify right now."
    }
}

# --- NEW: Helper function to clean the query ---
def _extract_clean_query(text: str) -> str:
    """
    Removes common trigger words and phrases from the beginning of the text
    to get a cleaner search query.
    """
    # List of trigger phrases to remove (in both languages)
    triggers = [
        "la canción", "el artista", "la playlist", "la música de", "música de",
        "canción", "artista", "playlist",
        "the song", "the artist", "the playlist", "music by",
        "song", "artist",
        "play", "reproducir", "pon", "escuchar", "listen to"
    ]
    
    text_lower = text.lower()
    # Sort triggers by length, longest first, to avoid partial matches (e.g., "la canción" before "canción")
    triggers.sort(key=len, reverse=True)

    for trigger in triggers:
        if text_lower.startswith(trigger + " "):
            # Return the original text, but with the trigger part sliced off
            return text[len(trigger):].strip()
            
    # If no trigger phrase is found at the start, return the original text
    return text.strip()
# --- END NEW ---


class Plugin:
    def __init__(self):
        self.config_manager = ConfigManager()
        logger.info("Music plugin initialized.")

    def get_description(self) -> str:
        return f"{RESPONSE_TEXTS['es']['description']} / {RESPONSE_TEXTS['en']['description']}"

    def handle(self, text: str, doc=None, context: dict = None, entities: list = None) -> str:
        current_lang = context.get('current_conversation_lang', 'es') if context else 'es'
        responses = RESPONSE_TEXTS[current_lang]

        search_query = None
        
        # 1. Prioritize entities from IntentProcessor
        if entities:
            # Join all music-related entities into a single string
            full_entity_text = " ".join([ent['text'] for ent in entities if ent['label'] in ["WORK_OF_ART", "PERSON", "ORG"]])
            if full_entity_text:
                # --- MODIFIED: Use the new cleaning function ---
                search_query = _extract_clean_query(full_entity_text)
                logger.info(f"Extracted and cleaned query from entities: '{search_query}'")

        # 2. Fallback to regex on the original text if no entities were useful
        if not search_query:
            match = re.search(r"(?:play|reproducir|pon|escuchar)\s+(.+)", text, re.IGNORECASE)
            if match:
                raw_query = match.group(1).strip()
                # --- MODIFIED: Also clean the regex result ---
                search_query = _extract_clean_query(raw_query)
                logger.info(f"Extracted and cleaned query using regex fallback: '{search_query}'")

        if search_query:
            spotify_search_url_template = self.config_manager.get_app_setting("spotify_search_url_template", "https://open.spotify.com/search/{query}")
            url_to_open = spotify_search_url_template.format(query=search_query)
            action_message = responses["searching_for"].format(query=search_query)
        else:
            # Generic playback if no query could be found
            url_to_open = self.config_manager.get_app_setting("spotify_default_url", "https://open.spotify.com")
            action_message = responses["opening_spotify"]

        try:
            logger.info(f"{action_message} URL: {url_to_open}")
            webbrowser.open(url_to_open)
            return action_message
        except Exception as e:
            logger.error(f"Error attempting to open Spotify: {e}")
            return responses["open_error"]