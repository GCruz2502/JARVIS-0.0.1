# jarvis/core/intent_processor.py
import logging
import importlib
import os
import re
import random
import speech_recognition as sr
from pathlib import Path
import inspect # Added for signature checking
import spacy # Added for NLP
from .context_manager import ContextManager # Import ContextManager
from pathlib import Path
from .my_custom_nlu import tokenize, NaiveBayesClassifier

logger = logging.getLogger(__name__)

class IntentProcessor:
    def __init__(self, context_manager: ContextManager, config_manager): # Added config_manager for potential plugin init needs
        """
        Inicializa el procesador de intenciones.
        Args:
            context_manager (ContextManager): Instancia del gestor de contexto.
            config_manager (ConfigManager): Instancia del gestor de configuración.
        """
        self.plugins = {}
        self.context_manager = context_manager # Use the passed ContextManager
        self.config_manager = config_manager # Store for use if plugins need config at init or direct access

        # For custom Naive Bayes intent classifiers
        self.intent_classifier_es: NaiveBayesClassifier | None = None
        self.intent_classifier_en: NaiveBayesClassifier | None = None
        self.custom_tokenizer = tokenize # Store your tokenizer function

        # For spaCy's linguistic features (tokenizer, POS, dep, NER from base model) & EntityRuler
        self.spacy_nlp_es = None # Will hold a spaCy model (e.g., es_core_news_sm)
        self.spacy_nlp_en = None # Will hold a spaCy model (e.g., en_core_web_sm)

        # Path for custom NLU models (Naive Bayes JSON files)
        # Assuming this file is in core/, models are in project_root/custom_jarvis_models/
        project_root = Path(__file__).resolve().parent.parent
        self.MODEL_NB_ES_PATH = str(project_root / "custom_jarvis_models" / "naive_bayes_es.json")
        self.MODEL_NB_EN_PATH = str(project_root / "custom_jarvis_models" / "naive_bayes_en.json")

        # --- Define the mapping from recognized intent labels to plugin keys ---
        # The keys here (e.g., "INTENT_GET_WEATHER") MUST EXACTLY MATCH the labels 
        # you used in your training data (ALL_INTENT_LABELS_ES/EN).
        # The values (e.g., "weather") MUST EXACTLY MATCH the keys used in 
        # self.plugins when plugins are loaded (typically the plugin's filename without .py).
        self.intent_to_plugin_map = {
            "INTENT_GET_WEATHER": "weather",
            "INTENT_PLAY_SONG": "music",
            "INTENT_PLAY_ARTIST": "music",
            "INTENT_PLAY_PLAYLIST": "music",
            "INTENT_PLAY_MUSIC": "music", # General music also maps to music plugin
            "INTENT_STOP": "music",
            "INTENT_SET_REMINDER": "reminders",
            "INTENT_GET_NEWS": "news_plugin", # Assuming your news plugin file is news_plugin.py
            "INTENT_OPEN_URL": "browser_control", # Assuming browser_control.py
            "INTENT_SEARCH_WEB": "browser_control", # browser_control plugin will need to differentiate these if needed
            "INTENT_SET_ALARM": "reminders", # Or a dedicated alarm plugin if you create one
                                            # For now, let's assume reminders plugin can handle alarms
            "INTENT_GET_TIME": "time_plugin", # <<<< YOU WILL NEED TO CREATE A time_plugin.py
            "INTENT_GET_DATE": "date_plugin", # <<<< YOU WILL NEED TO CREATE A date_plugin.py
            "INTENT_STOP": "music",           # Example: music plugin might handle 'stop'
                                            # Or you might need a general 'media_control_plugin'
            "INTENT_CANCEL": "reminders",     # Example: reminders plugin might handle cancellation
            # "INTENT_TELL_JOKE": "joke_plugin", # If you create one
            # "INTENT_ANSWER_QUESTION": "knowledge_plugin", # If you create one for specific Q&A
            # INTENT_HELP and INTENT_CLEAR_CONTEXT are handled internally before plugin routing in process()
        }
        logger.info(f"Intent-to-Plugin map initialized: {self.intent_to_plugin_map}")

        # Load base spaCy models (for linguistic features & EntityRuler)
        # This method will now load base spaCy models, not your custom trained pipelines
        self._load_base_spacy_models_and_ruler()
        
        # Load custom Naive Bayes intent classifiers
        self._load_custom_intent_classifiers()

        # AdvancedNLPProcessor (for sentiment, etc. - optional, phase out if desired)
        #self.advanced_nlp_processor = None 
        # self._initialize_advanced_nlp()
        
        self.load_plugins()
        logger.info("IntentProcessor initialized with Custom NLU.")

    def _add_custom_entity_ruler(self, nlp):
        """Adds custom rules for various entities to a spaCy pipeline."""
        logger.info("Adding custom EntityRuler for custom patterns...")
        ruler_name = "custom_entity_ruler"
        if ruler_name not in nlp.pipe_names:
            ruler = nlp.add_pipe("entity_ruler", name=ruler_name, after="ner", config={"overwrite_ents": True})
        else:
            ruler = nlp.get_pipe(ruler_name)

        # --- NEW, MORE POWERFUL PATTERNS ---
        patterns = [
            # TIME patterns (specific)
            {"label": "TIME", "pattern": [{"IS_DIGIT": True}, {"LOWER": {"IN": ["pm", "am"]}}]},
            {"label": "TIME", "pattern": [{"IS_DIGIT": True}, {"ORTH": ":"}, {"IS_DIGIT": True}]},
            {"label": "TIME", "pattern": [{"LOWER": "a"}, {"LOWER": "las"}, {"IS_DIGIT": True}]}, # a las 5
            {"label": "TIME", "pattern": [{"LOWER": "a"}, {"LOWER": "la"}, {"IS_DIGIT": True}]}, # a la 1

            # DATE patterns (specific words)
            {"label": "DATE", "pattern": [{"LOWER": "mañana"}]},
            {"label": "DATE", "pattern": [{"LOWER": "hoy"}]},
            {"label": "DATE", "pattern": [{"LOWER": "ayer"}]},
            
            # --- NEW: Relative Time Patterns ---
            {"label": "TIME", "pattern": [{"LOWER": "en"}, {"IS_DIGIT": True}, {"LOWER": {"IN": ["minuto", "minutos"]}}]}, # en 10 minutos
            {"label": "TIME", "pattern": [{"LOWER": "en"}, {"IS_DIGIT": True}, {"LOWER": {"IN": ["hora", "horas"]}}]},       # en 2 horas
            {"label": "DATE", "pattern": [{"LOWER": "próximo"}, {"LOWER": {"IN": ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]}}]}, # próximo martes
            {"label": "DATE", "pattern": [{"LOWER": "next"}, {"LOWER": {"IN": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]}}]}, # next tuesday

            # WORK_OF_ART patterns
            {"label": "WORK_OF_ART", "pattern": [{"LOWER": {"IN": ["canción", "song"]}}, {"IS_TITLE": True, "OP": "+"}]},
            {"label": "WORK_OF_ART", "pattern": [{"ORTH": '"'}, {"IS_ASCII": True, "OP": "+"}, {"ORTH": '"'}]},

            # GPE patterns
            {"label": "GPE", "pattern": [{"LOWER": "panamá"}]},
            {"label": "GPE", "pattern": [{"LOWER": "londres"}]},
            {"label": "GPE", "pattern": [{"LOWER": "madrid"}]},
            # ... add other locations as needed
        ]
        
        ruler.initialize(lambda: [], nlp=nlp, patterns=patterns)
        logger.info(f"EntityRuler '{ruler_name}' initialized/updated with {len(patterns)} patterns.")
        return nlp

    def _load_base_spacy_models_and_ruler(self):
        """
        Loads BASE spaCy models (for tokenizer, POS, dep, base NER) 
        and configures the EntityRuler on them.
        These spaCy models are NOT used for intent classification here.
        """
        # Load Spanish Model
        try:
            logger.info(f"Loading BASE Spanish spaCy model (es_core_news_lg)...")
            self.spacy_nlp_es = spacy.load("es_core_news_lg")
            logger.info("BASE Spanish spaCy model loaded successfully.")
            
            if self.spacy_nlp_es:
                self.spacy_nlp_es = self._add_custom_entity_ruler(self.spacy_nlp_es)
                logger.info("EntityRuler configured for BASE Spanish spaCy model.")
                logger.debug(f"Spanish BASE model pipe names: {self.spacy_nlp_es.pipe_names}")

        except Exception as e:
            logger.error(f"Error loading BASE Spanish spaCy model: {e}", exc_info=True)
            logger.warning("Spanish spaCy linguistic features (and EntityRuler) will be unavailable.")
            self.spacy_nlp_es = None

        # Load English Model
        try:
            logger.info("Loading BASE English spaCy model (en_core_web_lg)...")
            self.spacy_nlp_en = spacy.load("en_core_web_lg") # Or "en_core_web_lg"
            logger.info("BASE English spaCy model loaded successfully.")

            if self.spacy_nlp_en:
                self.spacy_nlp_en = self._add_custom_entity_ruler(self.spacy_nlp_en)
                logger.info("EntityRuler configured for BASE English spaCy model.")
                logger.debug(f"English BASE model pipe names: {self.spacy_nlp_en.pipe_names}")

        except Exception as e:
            logger.error(f"Error loading BASE English spaCy model: {e}", exc_info=True)
            logger.warning("English spaCy linguistic features (and EntityRuler) will be unavailable.")
            self.spacy_nlp_en = None

        # Optional: Log pipe names to verify
        if self.spacy_nlp_es:
            logger.debug(f"Spanish model pipe names: {self.spacy_nlp_es.pipe_names}")
        if self.spacy_nlp_en:
            logger.debug(f"English model pipe names: {self.spacy_nlp_en.pipe_names}")

    def _load_custom_intent_classifiers(self):
        """Loads the custom Naive Bayes intent classifiers from file."""
        logger.info("Loading custom Naive Bayes intent classifiers...")
        self.intent_classifier_es = NaiveBayesClassifier.load_model(self.MODEL_NB_ES_PATH)
        if self.intent_classifier_es:
            logger.info("Custom Spanish Naive Bayes intent classifier loaded.")
        else:
            logger.error(f"Failed to load Spanish Naive Bayes classifier from {self.MODEL_NB_ES_PATH}.")

        self.intent_classifier_en = NaiveBayesClassifier.load_model(self.MODEL_NB_EN_PATH)
        if self.intent_classifier_en:
            logger.info("Custom English Naive Bayes intent classifier loaded.")
        else:
            logger.error(f"Failed to load English Naive Bayes classifier from {self.MODEL_NB_EN_PATH}.")


    def clear_context(self):
        """Clears the conversation context using the ContextManager."""
        if self.context_manager:
            self.context_manager.clear_all_context()
            # logger.info("Contexto de conversación limpiado via ContextManager.") # Logged by ContextManager
        else:
            logger.warning("ContextManager no disponible en IntentProcessor para limpiar contexto.")

    def _get_help_string(self) -> str:
        """Generates a help string listing available plugin capabilities."""
        if not self.plugins:
            return "No hay plugins cargados actualmente."
        
        help_lines = ["Puedo ayudarte con lo siguiente:"]
        for name, plugin_instance in self.plugins.items():
            # Try to get a description from the plugin, otherwise just use the name
            if hasattr(plugin_instance, 'get_description') and callable(plugin_instance.get_description):
                try:
                    description = plugin_instance.get_description()
                    if description:
                        help_lines.append(f"- {name.replace('_', ' ').capitalize()}: {description}")
                    else: # Fallback if get_description returns empty
                        help_lines.append(f"- {name.replace('_', ' ').capitalize()}")
                except Exception as e:
                    logger.error(f"Error getting description from plugin {name}: {e}")
                    help_lines.append(f"- {name.replace('_', ' ').capitalize()} (Error al obtener descripción)")
            else:
                help_lines.append(f"- {name.replace('_', ' ').capitalize()}") # Basic help: plugin name
        
        # Add help for internal commands if any
        help_lines.append("- Limpiar contexto (di 'limpiar contexto' u 'olvida todo')") # Corrected 'dice' to 'di'
        help_lines.append("- Ayuda (di 'ayuda' o 'help')") # Corrected 'dice' to 'di'

        return "\n".join(help_lines)

    def load_plugins(self):
        """Carga dinámicamente los plugins desde el directorio plugins"""
        try:
            plugins_dir = Path(__file__).parent.parent / "plugins"
            plugin_files = [
                f for f in os.listdir(plugins_dir) 
                if f.endswith('.py') and f != '__init__.py' and not f.startswith('.')
            ]

            for plugin_file in plugin_files:
                plugin_name = plugin_file[:-3]  # Quitar la extensión .py
                try:
                    # Corrected import path for plugins
                    module = importlib.import_module(f"plugins.{plugin_name}")
                    if hasattr(module, 'Plugin'):
                        self.plugins[plugin_name] = module.Plugin()
                        logger.info(f"Plugin cargado: {plugin_name}")
                except Exception as e:
                    logger.error(f"Error al cargar plugin {plugin_name}: {str(e)}")
        except Exception as e:
            logger.error(f"Error al cargar plugins: {str(e)}")


    def process(self, text, lang_hint=None):
        # --- Initialize default return values ---
        final_response_str = "Lo siento, no estoy seguro de cómo ayudarte con eso todavía."
        recognized_intent_label = "UNKNOWN_INTENT"
        plugin_used_for_output = "IntentProcessorInternal_Fallback"
        merged_entities_for_output = []
        current_lang_for_output = "es"
        spacy_doc_entities_for_output = []

        # --- 1. Language and NLU Component Selection ---
        current_lang = "es"
        active_intent_classifier = self.intent_classifier_es
        active_spacy_nlp = self.spacy_nlp_es

        if lang_hint == "en" and self.intent_classifier_en and self.spacy_nlp_en:
            current_lang = "en"
            active_intent_classifier = self.intent_classifier_en
            active_spacy_nlp = self.spacy_nlp_en
        
        if not active_intent_classifier or not active_spacy_nlp:
            logger.error("NLU components for the determined language are not available.")
            return {"final_response": "Language components are not available.", "intent_label": "NLU_ERROR"}

        current_lang_for_output = current_lang
        logger.info(f"Language for NLU processing determined as: {current_lang}")

        # --- 2. Custom NLU Pipeline ---
        tokens = self.custom_tokenizer(text, current_lang)
        if tokens and active_intent_classifier:
            recognized_intent_label = active_intent_classifier.predict(tokens)
            logger.info(f"==> Custom Naive Bayes Recognized Intent: {recognized_intent_label}")
        else:
            logger.warning(f"Could not recognize intent for tokens: {tokens}")
            recognized_intent_label = "UNKNOWN_INTENT"

        # --- 3. SpaCy Linguistic Processing (NO Coreference for now) ---
        doc = active_spacy_nlp(text)
        # The coreference call is the source of an error, let's disable it for now.
        # if self.context_manager:
        #     doc = self._resolve_coreference(doc, self.context_manager.get_context_for_processing())
        
        ruler_entity_labels = {"TIME", "DATE", "PHONE", "WORK_OF_ART", "GPE"}
        if doc.ents:
            for ent in doc.ents:
                source = 'ruler' if ent.label_ in ruler_entity_labels else 'spacy_base_ner'
                spacy_doc_entities_for_output.append({'text': ent.text, 'label': ent.label_, 'source': source})
        merged_entities_for_output = spacy_doc_entities_for_output
        logger.debug(f"Entities from spaCy: {merged_entities_for_output}")

        # --- 4. Internal Command Handling ---
        internal_intent_handled = False
        if recognized_intent_label == "INTENT_HELP":
            final_response_str = self._get_help_string()
            plugin_used_for_output = "IntentProcessorInternal_Help"
            internal_intent_handled = True
        elif recognized_intent_label == "INTENT_CLEAR_CONTEXT":
            if self.context_manager: self.context_manager.clear_all_context()
            final_response_str = "Entendido, he limpiado el contexto."
            plugin_used_for_output = "IntentProcessorInternal_ClearContext"
            internal_intent_handled = True
        elif recognized_intent_label == "INTENT_GREET":
            responses = {"es": "¡Hola! ¿Cómo puedo ayudarte?", "en": "Hello! How can I help you?"}
            final_response_str = responses.get(current_lang, responses["es"])
            plugin_used_for_output = "IntentProcessorInternal_Greet"
            internal_intent_handled = True
        elif recognized_intent_label == "INTENT_FAREWELL":
            responses = {"es": "¡Hasta luego!", "en": "Goodbye!"}
            final_response_str = responses.get(current_lang, responses["es"])
            plugin_used_for_output = "IntentProcessorInternal_Farewell"
            internal_intent_handled = True

        # --- 5. Plugin Routing ---
        # THIS IS THE KEY FIX. We explicitly check the conditions for routing.
        
        # +++ START OF THE FIX +++
        logger.info(f"DEBUG: Checking routing conditions...")
        logger.info(f"DEBUG: Was internal intent handled? {'Yes' if internal_intent_handled else 'No'}")
        logger.info(f"DEBUG: Recognized Intent: '{recognized_intent_label}'")
        logger.info(f"DEBUG: Is Intent in Plugin Map? {'Yes' if recognized_intent_label in self.intent_to_plugin_map else 'No'}")

        if not internal_intent_handled and recognized_intent_label in self.intent_to_plugin_map:
            chosen_plugin_name = self.intent_to_plugin_map[recognized_intent_label]
            logger.info(f"Intent '{recognized_intent_label}' maps to plugin '{chosen_plugin_name}'.")

            if chosen_plugin_name in self.plugins:
                plugin_instance = self.plugins[chosen_plugin_name]
                logger.info(f"Plugin '{chosen_plugin_name}' instance found. Executing handle method.")
                try:
                    # Prepare context for the plugin
                    if self.context_manager:
                        self.context_manager.set_current_turn_data("recognized_intent_for_plugin", recognized_intent_label)
                        self.context_manager.set_current_turn_data("current_conversation_lang", current_lang)
                    
                    context_for_plugin = self.context_manager.get_context_for_processing() if self.context_manager else {}
                    
                    handle_result = plugin_instance.handle(
                        text=doc.text,
                        doc=doc,
                        entities=merged_entities_for_output,
                        context=context_for_plugin
                    )

                    if handle_result:
                        final_response_str = handle_result
                        plugin_used_for_output = chosen_plugin_name
                        logger.info(f"Plugin '{chosen_plugin_name}' provided response.")
                    else:
                        logger.warning(f"Plugin '{chosen_plugin_name}' returned None or empty.")
                        final_response_str = "El plugin se ejecutó pero no generó una respuesta."
                        plugin_used_for_output = f"{chosen_plugin_name}_NoResponse"

                except Exception as e:
                    logger.error(f"Error during handling by plugin '{chosen_plugin_name}': {e}", exc_info=True)
                    final_response_str = f"Lo siento, el plugin {chosen_plugin_name} encontró un error."
                    plugin_used_for_output = f"{chosen_plugin_name}_Error"
            else:
                logger.error(f"Plugin '{chosen_plugin_name}' is defined in map but not loaded in self.plugins.")
        # +++ END OF THE FIX +++

        # --- 6. Final Output Construction ---
        logger.info(f"Final processing: Intent='{recognized_intent_label}', Plugin='{plugin_used_for_output}'")
        output_data = {
            "final_response": final_response_str,
            "intent_label": recognized_intent_label,
            "plugin_used": plugin_used_for_output,
            "merged_entities": merged_entities_for_output,
            "current_lang": current_lang_for_output,
            # Add other keys back if needed, with default values
            "sentiment": None,
            "qa_result": None,
            "zero_shot_result": None,
            "empathetic_triggered": False,
            "raw_spacy_doc_entities": spacy_doc_entities_for_output,
            "raw_hf_ner_entities": []
        }
        return output_data

    def _resolve_coreference(self, doc, context):
        # ... (Your existing _resolve_coreference method - ensure it's robust and returns a spaCy Doc object)
        # For now, let's ensure it has the safety checks we discussed.
        if not doc: return doc # Should not happen if called after doc creation
        logger.debug("Attempting coreference resolution...")
        
        history = context.get("history", [])
        if not history:
            logger.debug("No history for coreference resolution.")
            return doc

        last_user_utterance_entities_dicts = []
        for turn in reversed(history):
            if turn.get("speaker") == "user" and isinstance(turn.get("entities"), list):
                last_user_utterance_entities_dicts = turn.get("entities")
                logger.debug(f"Coreference: Found last user utterance with entities: {turn['text']} -> {last_user_utterance_entities_dicts}")
                break
        
        if not last_user_utterance_entities_dicts:
            logger.debug("No previous user utterance with entities found in history for coreference resolution.")
            return doc

        pronouns = [token for token in doc if token.pos_ == "PRON"]
        if not pronouns:
            logger.debug("No pronouns in current utterance for coreference resolution.")
            return doc
            
        logger.debug(f"Identified pronouns in current utterance: {[p.text for p in pronouns]}")

        resolved_replacements = []
        modified_text_intermediate = doc.text # Work on a copy

        for pron_token in pronouns:
            best_match_text = None
            for entity_dict in last_user_utterance_entities_dicts:
                if isinstance(entity_dict, dict) and 'text' in entity_dict:
                    # Simplistic: use the text of the first entity from the last user turn
                    # This is a placeholder for more sophisticated matching (gender, number, type, proximity)
                    best_match_text = entity_dict['text']
                    logger.debug(f"Coreference: Potential match for '{pron_token.text}' is '{best_match_text}'")
                    break # Take first found entity from previous turn for simplicity

            if best_match_text:
                # Simple replacement in the current working text.
                # This is still basic and can have issues with overlapping pronouns or complex sentences.
                # We are only resolving the first pronoun found for this basic version.
                pron_start_char = pron_token.idx
                pron_end_char = pron_token.idx + len(pron_token.text)
                
                # Apply to modified_text_intermediate for this iteration
                # This simple replacement might not handle all cases perfectly if multiple pronouns exist
                # or if the replacement changes string length significantly affecting subsequent indices.
                # For a single pronoun, it's more straightforward.
                modified_text_intermediate = modified_text_intermediate[:pron_start_char] + best_match_text + modified_text_intermediate[pron_end_char:]
                logger.info(f"Coreference: Replaced '{pron_token.text}' with '{best_match_text}'. New intermediate text: '{modified_text_intermediate}'")
                # For this basic version, only resolve the first pronoun we find a match for.
                break 
        
        if modified_text_intermediate != doc.text:
            logger.info(f"Text modified by coreference. Original: '{doc.text}'. Resolved: '{modified_text_intermediate}'. Re-processing with spaCy.")
            active_spacy_nlp_for_resolve = None
            if doc.lang_ == 'es' and self.spacy_nlp_es:
                active_spacy_nlp_for_resolve = self.spacy_nlp_es
            elif doc.lang_ == 'en' and self.spacy_nlp_en:
                active_spacy_nlp_for_resolve = self.spacy_nlp_en
            
            if active_spacy_nlp_for_resolve:
                return active_spacy_nlp_for_resolve(modified_text_intermediate)
            else:
                logger.warning(f"No spaCy model for lang {doc.lang_} to re-process resolved text. Returning original doc.")
                return doc 
        else:
            logger.debug("No changes made by coreference resolution.")
            return doc
