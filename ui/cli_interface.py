"""
Módulo para la interfaz de línea de comandos (CLI) de JARVIS.
"""
import logging
from datetime import datetime, timezone
from core.text_to_speech import hablar
from utils.database_handler import collect_data # Para guardar interacciones
import json # needed for json.dumps for database_handler
from langdetect import detect, LangDetectException

logger = logging.getLogger(__name__)

def start_cli(intent_processor, context_manager, config_manager):
    """
    Inicia la interfaz de línea de comandos para interactuar con JARVIS.

    Args:
        intent_processor: Instancia de IntentProcessor.
        context_manager: Instancia de ContextManager.
        config_manager: Instancia de ConfigManager.
    """
    logger.info("Iniciando CLI de JARVIS...")
    
    initial_greeting = config_manager.get_app_setting("initial_greeting", "Hola, soy JARVIS. ¿Cómo puedo ayudarte hoy?")
    print(f"JARVIS: {initial_greeting}")
    hablar(initial_greeting)

    current_session_lang = config_manager.get_app_setting("default_language_hint", "es")
    logger.info(f"Default session language set to: {current_session_lang}")

    supported_langs = ["es", "en"] # +++ ADD THIS: Define supported languages +++
    default_lang_for_notifications = "es" # +++ ADD THIS: For unsupported lang messages +++

    while True:
        cli_prompt_template = config_manager.get_app_setting("cli_prompt_template", "Tú ({lang}): ") # Use a template
        actual_prompt = cli_prompt_template.format(lang=current_session_lang.upper())
        
        user_input_raw = input(actual_prompt)
        if user_input_raw is None: 
            logger.info("Entrada de usuario es None (EOF), saliendo.")
            break
        user_input_text = user_input_raw.strip() # Use a different variable for the text itself

        if not user_input_text:
            logger.info("Entrada de usuario vacía, continuando.")
            continue

        # Exit commands check (using the raw input text to catch mixed case)
        if user_input_text.lower() in ["exit", "quit", "bye", "salir", "adiós", "chao"]:
            farewell_message_parts = {
                "es": config_manager.get_app_setting("farewell_message_es", "¡Adiós!"),
                "en": config_manager.get_app_setting("farewell_message_en", "Goodbye!")
            }
            farewell_message = farewell_message_parts.get(current_session_lang, farewell_message_parts["es"])
            print(f"JARVIS: {farewell_message}")
            hablar(farewell_message, lang=current_session_lang)
            logger.info("Saliendo de la CLI de JARVIS.")
            if context_manager:
                context_manager.add_utterance('user', user_input_text, intent="exit_command", language=current_session_lang)
            break # Exit the while loop

        # --- Process /lang command first (Manual Override) ---
        if user_input_text.lower().startswith("/lang "):
            try:
                new_lang_manual = user_input_text.split(" ")[1].lower()
                if new_lang_manual in supported_langs:
                    current_session_lang = new_lang_manual
                    # Update context if your plugins use it for language-specific responses
                    if context_manager:
                        context_manager.set_current_turn_data("current_conversation_lang", current_session_lang)
                    
                    lang_changed_msg_parts = {
                        "es": f"Idioma de la sesión cambiado a: ESPAÑOL",
                        "en": f"Session language changed to: ENGLISH"
                    }
                    lang_changed_msg = lang_changed_msg_parts.get(current_session_lang)

                    print(f"JARVIS: {lang_changed_msg}")
                    hablar(lang_changed_msg, lang=current_session_lang) # Speak in the new language
                    logger.info(f"Session language manually switched to: {current_session_lang}")
                else:
                    lang_invalid_msg = "Idioma no válido. Usa '/lang es' o '/lang en'."
                    print(f"JARVIS: {lang_invalid_msg}")
                    hablar(lang_invalid_msg, lang=current_session_lang) # Speak in current session lang
            except IndexError:
                lang_incomplete_msg = "Comando de idioma incompleto. Usa '/lang es' o '/lang en'."
                print(f"JARVIS: {lang_incomplete_msg}")
                hablar(lang_incomplete_msg, lang=current_session_lang)
            continue 
        # --- End /lang command processing ---  

        # +++ Language Detection for general input +++
        detected_lang_code = None
        try:
            if user_input_text: # Ensure text is not empty before detecting
                detected_lang_code = detect(user_input_text)
                logger.info(f"Detected language of input '{user_input_text}': {detected_lang_code}")
            else: # Should have been caught by "if not user_input_text" earlier
                logger.warning("Empty input reached language detection somehow.")
                continue
        except LangDetectException:
            detected_lang_code = "unknown" # Could not detect
            logger.warning(f"Language detection failed for input: '{user_input_text}'. Marked as 'unknown'.")

        if detected_lang_code in supported_langs:
            # If detected language is supported, update the session language
            if current_session_lang != detected_lang_code:
                logger.info(f"Input language '{detected_lang_code}' differs from session language '{current_session_lang}'. Updating session language.")
                current_session_lang = detected_lang_code
                if context_manager: # Update context for plugins
                    context_manager.set_current_turn_data("current_conversation_lang", current_session_lang)
            # Proceed with processing in the (potentially updated) current_session_lang
            lang_hint_for_processor = current_session_lang
        elif detected_lang_code == "unknown":
            # If detection fails, use the current session language as a fallback hint
            logger.warning(f"Language was unknown, using current session language '{current_session_lang}' as hint for processor.")
            lang_hint_for_processor = current_session_lang
        else: # Language is detected but not supported
            logger.warning(f"Detected language '{detected_lang_code}' is not supported. Notifying user in '{default_lang_for_notifications}'.")
            unsupported_lang_msg_parts = {
                "es": f"Lo siento, no entiendo el idioma '{detected_lang_code}'. Actualmente solo puedo procesar español e inglés.",
                "en": f"Sorry, I don't understand the language '{detected_lang_code}'. I can currently only process Spanish and English."
            }
            # Notify in the default notification language (Spanish)
            notification_msg = unsupported_lang_msg_parts.get(default_lang_for_notifications, unsupported_lang_msg_parts["es"])
            
            print(f"JARVIS: {notification_msg}")
            hablar(notification_msg, lang=default_lang_for_notifications)
            
            # Log this interaction attempt (optional, but good for analytics)
            timestamp_utc = datetime.now(timezone.utc).isoformat()
            if config_manager.get_app_setting("collect_interaction_data", False):
                collect_data(
                    timestamp=timestamp_utc, user_input=user_input_text, intent="UNSUPPORTED_LANGUAGE_DETECTED",
                    entities=None, sentiment=None, plugin_used="CLI_LanguageHandler",
                    response=notification_msg, success=0, language=detected_lang_code # Log detected language
                )
            continue # Skip NLU processing for unsupported language
        # +++ End Language Detection +++

        # Process user input (already stripped, case will be handled by tokenizer)
        logger.info(f"Input de usuario para NLU ({lang_hint_for_processor}): '{user_input_text}'")

        
        # Call IntentProcessor with the determined language hint
        result_dict = intent_processor.process(user_input_text, lang_hint=lang_hint_for_processor)
        
        response_text = result_dict.get("final_response", "No pude procesar eso.")
        timestamp_utc = datetime.now(timezone.utc).isoformat() 

        # Add user utterance with details from processing
        if context_manager:
            user_turn_details = {
                "timestamp": timestamp_utc, "intent": result_dict.get("intent_label"),
                "entities": result_dict.get("merged_entities"), "sentiment": result_dict.get("sentiment"),
                "language": result_dict.get("current_lang"), # Lang used by intent_processor
            }
            user_turn_details = {k: v for k, v in user_turn_details.items() if v is not None}
            context_manager.add_utterance('user', user_input_text, **user_turn_details)

            # Add assistant utterance with details from processing
            assistant_turn_details = {
                "timestamp": timestamp_utc, "plugin_triggered": result_dict.get("plugin_used"),
                "triggered_by_intent": result_dict.get("intent_label"),
                "empathetic_response_used": result_dict.get("empathetic_triggered"),
            }
            assistant_turn_details = {k: v for k, v in assistant_turn_details.items() if v is not None}
            context_manager.add_utterance('assistant', response_text, **assistant_turn_details)

        print(f"JARVIS: {response_text}")
        # Speak response in the current_session_lang (which should match result_dict.get("current_lang"))
        hablar(response_text, lang=current_session_lang)

        # Database logging
        if config_manager.get_app_setting("collect_interaction_data", False):
            try:
                # ... (your existing database logging, ensure language_for_db uses current_session_lang or result_dict.get("current_lang"))
                plugin_used_for_db = result_dict.get("plugin_used")
                intent_label_for_db = result_dict.get("intent_label")
                success_for_db = 1 if plugin_used_for_db and not plugin_used_for_db.endswith(("_Error", "_NoResponse", "Fallback")) and not str(intent_label_for_db).startswith("UNKNOWN_") else 0
                entities_for_db_json = json.dumps(result_dict.get("merged_entities")) if result_dict.get("merged_entities") is not None else None
                sentiment_for_db_json = json.dumps(result_dict.get("sentiment")) if result_dict.get("sentiment") is not None else None
                language_for_db = result_dict.get("current_lang", current_session_lang)

                if collect_data(
                    timestamp=timestamp_utc, user_input=user_input_text, # Log original case user_input_text
                    intent=intent_label_for_db, entities=entities_for_db_json,
                    sentiment=sentiment_for_db_json, plugin_used=plugin_used_for_db,
                    response=response_text, success=success_for_db, language=language_for_db
                ):
                    logger.info("Interaction data saved to database successfully.")
                else:
                    logger.error("Failed to save interaction data to database.")
            except Exception as e:
                logger.error(f"Error saving data to database: {e}", exc_info=True)

if __name__ == '__main__':
    # Esto es solo para pruebas directas de este módulo.
    # En la aplicación real, main.py configurará e iniciará todo.
    print("Modo de prueba para cli_interface.py")
    
    # Crear instancias dummy/mock para probar
    # Esto requeriría importar las clases y posiblemente configurar mucho.
    # Por simplicidad, este __main__ no ejecutará start_cli sin un setup adecuado.
    
    # class DummyIntentProcessor:
    #     def process(self, text, lang_hint=None):
    #         return {"final_response": f"Procesado (dummy): {text}"}
    # class DummyContextManager:
    #     def add_utterance(self, speaker, text): pass
    # class DummyConfigManager:
    #     def get_app_setting(self, key, default=None): return default

    # intent_proc = DummyIntentProcessor()
    # ctx_mngr = DummyContextManager()
    # cfg_mngr = DummyConfigManager()
    
    # print("Para probar, necesitarías un entorno JARVIS completamente configurado.")
    # print("Ejecuta main.py para la funcionalidad completa.")
    # # start_cli(intent_proc, ctx_mngr, cfg_mngr) # No se puede ejecutar sin más setup
    pass
