# plugins/music.py
import logging
import re
import webbrowser # For opening URLs
from spacy.tokens import Doc
from utils.config_manager import ConfigManager # Import ConfigManager

logger = logging.getLogger(__name__)

class Plugin:
    def __init__(self):
        """Initializes the music plugin."""
        self.config_manager = ConfigManager() # Use ConfigManager
        logger.info("Music plugin initialized.")
        # Example: self.spotify_client_id = self.config_manager.get_env_variable("SPOTIFY_CLIENT_ID")

    def get_description(self) -> str:
        return "Reproduce música o busca canciones/artistas en Spotify."

    def can_handle(self, text: str, doc: Doc = None, context: dict = None, entities: list = None) -> bool: # Added entities
        """Determines if this plugin can handle the text based on keywords or lemmas."""
        # Using keywords/lemmas is likely sufficient for triggering the music plugin
        keywords_es = ["reproducir", "pon", "escuchar", "música", "canción", "artista", "playlist", "sonar"]
        keywords_en = ["play", "put on", "listen", "music", "song", "artist", "playlist", "sound", "hear"] # Added "hear"
        
        if doc:
            # Check lemmas for robustness
            lemmas_es = ["reproducir", "poner", "escuchar", "música", "canción", "artista", "playlist", "sonar"]
            lemmas_en = ["play", "put", "listen", "music", "song", "artist", "playlist", "sound", "hear"] # Added "hear"
            
            # Determine language based on doc object's vocab lang if possible, else use context or default
            lang = context.get('current_lang', 'es') # Assuming context might hold lang
            if hasattr(doc.vocab, 'lang'):
                 lang = doc.vocab.lang

            target_lemmas = lemmas_es if lang == 'es' else lemmas_en
            if any(token.lemma_ in target_lemmas for token in doc):
                return True

        # Fallback to simple keyword check if doc/context is not available or lemma check fails
        target_keywords = keywords_es + keywords_en
        return any(keyword in text.lower() for keyword in target_keywords)

    def handle(self, text: str, doc: Doc = None, context: dict = None, entities: list = None) -> str: # Added entities
        """Handles music playback requests, prioritizing merged entities."""
        
        song_or_artist = None
        
        # 1. Prioritize merged entities if available
        if entities:
            # Look for WORK_OF_ART, PERSON, ORG labels from any source
            music_entities = [ent['text'] for ent in entities if ent['label'] in ["WORK_OF_ART", "PERSON", "ORG"]]
            if music_entities:
                song_or_artist = " ".join(music_entities)
                logger.info(f"Extracted music entity from merged_entities: '{song_or_artist}' (Sources: {[ent['source'] for ent in entities if ent['label'] in ['WORK_OF_ART', 'PERSON', 'ORG']]})")

        # 2. Fallback to spaCy doc.ents if merged_entities didn't yield result
        if not song_or_artist and doc:
            logger.info("No music entity found in merged_entities, checking doc.ents...")
            # Look for WORK_OF_ART (songs, albums), PERSON (artists), ORG (bands?)
            doc_entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ["WORK_OF_ART", "PERSON", "ORG"]]
            if doc_entities:
                song_or_artist = " ".join([ent[0] for ent in doc_entities])
                logger.info(f"Extracted music entity from spaCy doc: '{song_or_artist}' (Entities: {doc_entities})")
        
        # 3. Fallback to regex if still no entities found
        if not song_or_artist:
            # Simple regex to find text after "play" or "reproducir" etc.
            match = re.search(r"(?:play|reproducir|pon|escuchar)\s+(.+)", text, re.IGNORECASE)
            if match:
                song_or_artist = match.group(1).strip()
                # Remove potential trailing prepositional phrases like "on spotify"
                song_or_artist = re.sub(r"\s+(on|en)\s+\w+$", "", song_or_artist, flags=re.IGNORECASE).strip()
                logger.info(f"Extracted music query using regex: '{song_or_artist}'")

        if song_or_artist:
            # Construct a Spotify search URL
            # This could be made more robust, e.g. URL encoding the query
            spotify_search_url_template = self.config_manager.get_app_setting(
                "spotify_search_url_template", 
                "https://open.spotify.com/search/{query}"
            )
            url_to_open = spotify_search_url_template.format(query=song_or_artist)
            action_message = f"Buscando '{song_or_artist}' en Spotify."
        else:
            # Generic playback: open Spotify main page or a default playlist
            url_to_open = self.config_manager.get_app_setting(
                "spotify_default_url", 
                "https://open.spotify.com"
            )
            action_message = "Abriendo Spotify."

        try:
            logger.info(f"{action_message} URL: {url_to_open}")
            webbrowser.open(url_to_open)
            return f"{action_message}"
        except Exception as e:
            logger.error(f"Error al intentar abrir Spotify: {e}")
            return "Lo siento, no pude abrir Spotify en este momento."

    # Example helper method (replace with actual implementation for direct API control)
    # def _initialize_spotify_client(self):
    #     # Logic to initialize a Spotify API client
    #     # client_id = self.config_manager.get_env_variable("SPOTIFY_CLIENT_ID")
    #     # client_secret = self.config_manager.get_env_variable("SPOTIFY_CLIENT_SECRET")
    #     # ... setup client ...
    #     return None
