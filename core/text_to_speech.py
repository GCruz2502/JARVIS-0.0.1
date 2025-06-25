"""
Módulo para manejar la salida de voz (Text-to-Speech)
"""
import pyttsx3
import logging

logger = logging.getLogger(__name__)  # Get logger at the module level

# --- Configuration for Voice IDs (YOU WILL UPDATE THESE) ---
VOICE_ID_ES = "com.apple.voice.compact.es-MX.Paulina"
VOICE_ID_EN = "com.apple.voice.compact.en-US.Samantha"
# --- End Configuration ---

# Iniciar motor de voz
engine = None
try:
    # Configure logger temporarily to see DEBUG messages on console for this run
    #logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True) # << TEMP for this test  <--- UNCOMMENTED
    # For regular use, your main.py setup_logging is better.
    
    logger.info("Initializing TTS engine...")
    engine = pyttsx3.init()
    if engine:
        engine.setProperty("rate", 150)  # Set default speaking rate (words per minute)
        
        # --- THIS IS THE IMPORTANT PART FOR GETTING VOICE IDs ---
        voices = engine.getProperty('voices') 
        logger.info("--- Available TTS Voices (Raw Details) ---") #changed to INFO
        if voices:
            for i, voice in enumerate(voices):
                try:
                    # On macOS, voice.languages might be a list of strings directly, or needs careful handling
                    lang_info = voice.languages if voice.languages else "N/A"
                    logger.info(f"  voice {i}:") 
                    logger.info(f"    ID: {voice.id}") 
                    logger.info(f"    Name: {voice.name}")
                    logger.info(f"    Languages Raw: {lang_info}") # Log the raw attribute
                    logger.info(f"    Gender: {voice.gender}") 
                    logger.info(f"    Age: {voice.age}") 
                    logger.info(f"    ---") 
                except Exception as e:
                    logger.error(f"Error inspecting voice {getattr(voice, 'id', 'UNKNOWN_ID')}: {e}")
        else:
            logger.warning("No voices found by engine.getProperty('voices')")
        # --- END OF IMPORTANT PART ---

        # The automatic selection loop is unlikely to work well with NSTaggedPointerString
        # It's better to rely on manually setting VOICE_ID_ES and VOICE_ID_EN above.
        # We can comment out or remove the automatic selection part for now.

        # logger.info("Attempting automatic voice selection (may not be effective on macOS for language list)...")
        # for voice in voices: # This loop can be commented out if IDs are hardcoded
            #if VOICE_ID_ES is None and voice.languages:
                #try:
                    # This part is problematic on macOS due to NSTaggedPointerString
                    # lang_codes = [str(lang_item).lower() for lang_item in voice.languages] # Try direct str conversion
                    #if any(lc.startswith('es') for lc in lang_codes):
                        #VOICE_ID_ES = voice.id
                        #logger.info(f"Automatically selected Spanish voice: {voice.name} (ID: {VOICE_ID_ES})")
                    #pass # Skip auto-selection based on voice.languages for now
                #except Exception as e:
                    #logger.debug(f"Could not parse languages for voice {voice.id}: {voice.languages} - {e}")

        # logger.info("Attempting automatic voice selection (may not be effective on macOS for language list)...")
        # for voice in voices: # This loop can be commented out if IDs are hardcoded
            #if VOICE_ID_EN is None and voice.languages:
                #try:
                    # This part is problematic on macOS due to NSTaggedPointerString
                    # lang_codes = [str(lang_item).lower() for lang_item in voice.languages] # Try direct str conversion
                    #if any(lc.startswith('es') for lc in lang_codes):
                        #VOICE_ID_EN = voice.id
                        #logger.info(f"Automatically selected Spanish voice: {voice.name} (ID: {VOICE_ID_EN})")
                    #pass # Skip auto-selection based on voice.languages for now
                #except Exception as e:
                    #logger.debug(f"Could not parse languages for voice {voice.id}: {voice.languages} - {e}")

        if VOICE_ID_ES is None:
            logger.warning("VOICE_ID_ES is not set. Default voice will be used for Spanish.")
        else:
            logger.info(f"Spanish voice ID set to: {VOICE_ID_ES}")

        if VOICE_ID_EN is None:
            logger.warning("VOICE_ID_EN is not set. Default voice will be used for English.")
        else:
            logger.info(f"English voice ID set to: {VOICE_ID_EN}")

    else:
        logger.error("Falló la inicialización del motor pyttsx3 (engine is None).")
except Exception as e:
    logger.error(f"Error al inicializar el motor de Text-to-Speech: {e}", exc_info=True)


# +++ MODIFIED hablar function +++
def hablar(texto: str, lang: str = "es"): # Added lang argument with default "es"
    """
    Convierte texto a voz, attempting to use a language-specific voice.
    
    Args:
        texto (str): El texto que se convertirá a voz
        lang (str): The language code ('es' or 'en') for voice selection.
    """
    if engine is None:
        logger.error("Motor de Text-to-Speech no inicializado. No se puede hablar.")
        return

    try:
        # --- Attempt to set voice based on language ---
        current_engine_voice_id = engine.getProperty('voice') # Get current voice to revert if needed
        logger.debug(f"Current engine voice before potential switch: {current_engine_voice_id}")

        target_voice_id = None
        if lang == "es":
            target_voice_id = VOICE_ID_ES
        elif lang == "en": # Fallback to default if no specific Spanish voice is set
                target_voice_id = VOICE_ID_EN

        # If no specific voice is found for the target language, use the current engine default.
        # This avoids unnecessarily trying to set a None voice_id.
        if not target_voice_id:
          logger.debug(f"No specific voice ID for lang '{lang}'. Using current engine voice: {current_engine_voice_id}") # <<< TYPO: debbug -> debug
          # No need to setProperty if we're using the one that's already there or default for the lang
        elif target_voice_id != current_engine_voice_id:
            try:
                logger.debug(f"Attempting to switch TTS voice to ID: {current_engine_voice_id} to {target_voice_id} for language '{lang}'")
                engine.setProperty('voice', target_voice_id)

                # +++ ADDED DUMMY SAY AND RUNANDWAIT +++
                # Only do this if we actually attempted a change.
                # No need to do it if target_voice_id was already the current_engine_voice_id from the start.
                logger.debug("Speaking empty string to attempt to latch voice change...")
                engine.say(" ") # Speak a tiny (or even empty) string
                engine.runAndWait()
                # +++++++++++++++++++++++++++++++++++++++

                # Verify the voice AFTER attempting to set and "latch" it
                new_current_voice_id_after_set = engine.getProperty('voice')
                if new_current_voice_id_after_set == target_voice_id:
                    logger.info(f"Successfully set and latched voice property to ID: {target_voice_id} for language '{lang}'")
                else:
                    logger.warning(f"Attempted to set voice to {target_voice_id}, but engine reports current voice is {new_current_voice_id_after_set}. Reverting to original.")
                    # If it didn't stick, or stuck to something else, revert to original for this call
                    if current_engine_voice_id: # Ensure we have an original ID to revert to
                        engine.setProperty('voice', current_engine_voice_id)
                        # It might be good to "latch" the revert too
                        engine.say(" ") 
                        engine.runAndWait()
                    else: # Should not happen if engine is initialized
                        logger.error("Original engine voice ID was None, cannot revert.")

            except Exception as e:
                logger.error(f"Error during voice setting for ID '{target_voice_id}' for lang '{lang}': {e}. Current voice: {current_engine_voice_id}")
                # If setProperty itself failed, the voice likely remains current_engine_voice_id
                # If it changed but then an error occurred, it might be in an intermediate state.
                # Reverting might be good here too if current_engine_voice_id is known.
                if current_engine_voice_id:
                    engine.setProperty('voice', current_engine_voice_id)

        else: # This means target_voice_id == current_engine_voice_id
            logger.debug(f"Target voice ID {target_voice_id} is already current. No change needed.")

        final_voice_used = engine.getProperty('voice')
        logger.info(f"Hablando (lang={lang}, voice_id_to_use={final_voice_used}): {texto}")
        engine.say(texto)
        engine.runAndWait()


    except Exception as e:
        logger.error(f"Error al hablar: {e}", exc_info=True)

if __name__ == '__main__':
    # This block is just for testing this file directly
    # Make sure to configure VOICE_ID_ES and VOICE_ID_EN above with values from your system
    # after running this script once to see the voice list.

    # Ensure basicConfig is called if running standalone for logger to work
    if not logging.getLogger().hasHandlers(): # Check if handlers are already set
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


    logger.info("Running text_to_speech.py directly for testing...")
    if engine:
        # --- MANUALLY SET YOUR VOICE IDs HERE FOR TESTING THIS SCRIPT ---
        # VOICE_ID_ES = "com.apple.voice.compact.es-MX.Paulina" # Example from your logs
        # VOICE_ID_EN = "com.apple.voice.compact.en-US.Samantha"  # Example from your logs (if available)
        # logger.info(f"TESTING WITH ES VOICE: {VOICE_ID_ES}")
        # logger.info(f"TESTING WITH EN VOICE: {VOICE_ID_EN}")
        # ---------------------------------------------------------------

        hablar("Hola, esto es una prueba en español.", lang="es")
        hablar("Hello, this is a test in English.", lang="en")
        hablar("Prueba con el lenguage por defecto") # Should use Spanish default voice
    else:
          logger.error("Engine not initialized, cannot run direct test.")
    logger.info("text_to_speech.py direct test finished.")