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
from .nlp_engine import AdvancedNLPProcessor # Corrected: AdvancedNLPProcessor is now in nlp_engine
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
        self.advanced_nlp_processor = None 
        self._initialize_advanced_nlp()
        
        self.load_plugins()
        logger.info("IntentProcessor initialized with Custom NLU.")

    def _add_custom_entity_ruler(self, nlp):
        """Adds custom rules for various entities (TIME, DATE, PHONE, etc.) to a spaCy pipeline."""
        logger.info("Adding custom EntityRuler for custom patterns...")
        ruler_name = "custom_entity_ruler" # More general name
        # Check if ruler already exists
        if ruler_name in nlp.pipe_names:
            ruler = nlp.get_pipe(ruler_name)
            logger.info(f"EntityRuler '{ruler_name}' already exists, will be updated with new patterns.")
        else:
            # Add ruler AFTER NER and configure to OVERWRITE statistical entities
            # OPTIONAL MODIFICATION FOR ROBUSTNESS:
            if "ner" in nlp.pipe_names:
                ruler = nlp.add_pipe("entity_ruler", name=ruler_name, after="ner", config={"overwrite_ents": True})
                logger.info(f"EntityRuler '{ruler_name}' added after 'ner' with overwrite_ents=True.")
            elif "textcat" in nlp.pipe_names: # Fallback: add before textcat if ner isn't there but textcat is
                ruler = nlp.add_pipe("entity_ruler", name=ruler_name, before="textcat", config={"overwrite_ents": True})
                logger.info(f"EntityRuler '{ruler_name}' added before 'textcat' (ner not found) with overwrite_ents=True.")
            else: # Further fallback: add it last if neither ner nor textcat are found (unlikely for our case)
                ruler = nlp.add_pipe("entity_ruler", name=ruler_name, config={"overwrite_ents": True})
                logger.info(f"EntityRuler '{ruler_name}' added last (ner and textcat not found) with overwrite_ents=True.")

        # Define patterns
        patterns = [
            # --- TIME Patterns ---
            {"label": "TIME", "pattern": [{"IS_DIGIT": True}, {"LOWER": {"IN": ["pm", "am"]}}]}, # 5 pm
            {"label": "TIME", "pattern": [{"IS_DIGIT": True}, {"ORTH": ":"}, {"IS_DIGIT": True}]}, # 14:30
            {"label": "TIME", "pattern": [{"IS_DIGIT": True}, {"LOWER": "de"}, {"LOWER": "la"}, {"LOWER": {"IN": ["mañana", "tarde", "noche"]}}]}, # 7 de la mañana
            {"label": "TIME", "pattern": [{"IS_DIGIT": True}, {"LOWER": "en"}, {"LOWER": "punto"}]}, # 5 en punto

            # --- DATE Patterns ---
            {"label": "DATE", "pattern": [{"LOWER": "mañana"}, {"LOWER": "por"}, {"LOWER": "la"}, {"LOWER": "tarde"}]}, # mañana por la tarde
            {"label": "DATE", "pattern": [{"LOWER": "mañana"}]}, # mañana
            {"label": "DATE", "pattern": [{"LOWER": "hoy"}]}, # hoy
            {"label": "DATE", "pattern": [{"LOWER": "ayer"}]}, # ayer
            {"label": "DATE", "pattern": [{"LOWER": "pasado"}, {"LOWER": "mañana"}]}, # pasado mañana

            # --- PHONE Number Patterns (Using Regex) ---
            # Matches ####-#### or ######## (Panama style)
            # Regex needs to match entire token(s)
            # Option 1: Match single token if number is not split
            {"label": "PHONE", "pattern": [{"TEXT": {"REGEX": r"^\d{4}-?\d{4}$"}}]},
            # Option 2: Match sequence if tokenized with hyphen (less likely but possible)
            # {"label": "PHONE", "pattern": [{"SHAPE": "dddd"}, {"ORTH": "-"}, {"SHAPE": "dddd"}]}, # Keep original shape as fallback?

            # --- WORK_OF_ART Patterns (Tentative/Experimental) ---
            # Text after "canción" or "álbum" - might be too broad
            {"label": "WORK_OF_ART", "pattern": [{"LOWER": {"IN": ["canción", "song", "album", "álbum"]}}, {"IS_TITLE": True, "OP": "+"}]}, # Added "song"
            # Text in quotes (Corrected to capture content within quotes)
            {"label": "WORK_OF_ART", "pattern": [{"ORTH": '"'}, {"IS_ASCII": True, "OP": "+"}, {"ORTH": '"'}]},
            {"label": "WORK_OF_ART", "pattern": [{"ORTH": "'"}, {"IS_ASCII": True, "OP": "+"}, {"ORTH": "'"}]},

            # --- LOCATION Patterns (GPE - Geopolitical Entity) ---
            {"label": "GPE", "pattern": [{"LOWER": "panamá"}]},
            {"label": "GPE", "pattern": [{"LOWER": "panama"}]}, # English spelling
            {"label": "GPE", "pattern": [{"LOWER": "londres"}]},
            {"label": "GPE", "pattern": [{"LOWER": "london"}]}, # English spelling
            {"label": "GPE", "pattern": [{"LOWER": "nueva"}, {"LOWER": "york"}]},
            {"label": "GPE", "pattern": [{"LOWER": "new"}, {"LOWER": "york"}]}, # English spelling
            # Add more specific city/country patterns here as needed
            # For example:
            # {"label": "GPE", "pattern": [{"LOWER": "parís"}]},
            # {"label": "GPE", "pattern": [{"LOWER": "paris"}]},
        ]

        # Initialize the ruler with the new patterns. This will overwrite existing patterns if the ruler already exists.
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

    def _initialize_advanced_nlp(self):
        """Initializes the Advanced NLP Processor."""
        try:
            logger.info("Attempting to initialize AdvancedNLPProcessor...")
            self.advanced_nlp_processor = AdvancedNLPProcessor()

            # Check readiness of individual language pipelines
            en_sentiment_ready = self.advanced_nlp_processor and self.advanced_nlp_processor.sentiment_analyzer_en
            en_qa_ready = self.advanced_nlp_processor and self.advanced_nlp_processor.qa_pipeline_en
            es_sentiment_ready = self.advanced_nlp_processor and self.advanced_nlp_processor.sentiment_analyzer_es
            es_qa_ready = self.advanced_nlp_processor and self.advanced_nlp_processor.qa_pipeline_es
            zero_shot_ready = self.advanced_nlp_processor and self.advanced_nlp_processor.zero_shot_classifier
            ner_ready = self.advanced_nlp_processor and self.advanced_nlp_processor.ner_pipeline # Check NER

            if not (en_sentiment_ready or en_qa_ready or es_sentiment_ready or es_qa_ready or zero_shot_ready or ner_ready):
                logger.warning("AdvancedNLPProcessor initialized, but no language pipelines (Sentiment/QA/ZeroShot/NER) seem to be ready.")
                self.advanced_nlp_processor = None # Set to None if no pipelines are useful
            else:
                logger.info("AdvancedNLPProcessor initialized. Readiness:")
                logger.info(f"  EN Sentiment: {'Ready' if en_sentiment_ready else 'Not Ready'}")
                logger.info(f"  EN QA: {'Ready' if en_qa_ready else 'Not Ready'}")
                logger.info(f"  ES Sentiment: {'Ready' if es_sentiment_ready else 'Not Ready'}")
                logger.info(f"  ES QA: {'Ready' if es_qa_ready else 'Not Ready'}")
                logger.info(f"  Zero-Shot Classification: {'Ready' if zero_shot_ready else 'Not Ready'}")
                logger.info(f"  HF NER: {'Ready' if ner_ready else 'Not Ready'}") # Log NER readiness

        except Exception as e:
            logger.error(f"Failed to initialize AdvancedNLPProcessor: {e}", exc_info=True)
            self.advanced_nlp_processor = None


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
        sentiment_result_for_output = None
        current_lang_for_output = "es"
        empathetic_triggered_for_output = False
        spacy_doc_entities_for_output = []

        # --- 1. Language Determination ---
        current_lang = "es"
        active_nlp = None # For custom Naive Bayes
        active_spacy_nlp = None # For base spaCy features

        if lang_hint == "en":
            if self.intent_classifier_en and self.spacy_nlp_en:
                current_lang = "en"
                active_intent_classifier = self.intent_classifier_en
                active_spacy_nlp = self.spacy_nlp_en
            else:
                logger.warning("English NLU components not fully loaded, defaulting to Spanish if available.")
        elif lang_hint == "es": # Default or explicit Spanish
            if self.intent_classifier_es and self.spacy_nlp_es:
                current_lang = "es"
                active_intent_classifier = self.intent_classifier_es
                active_spacy_nlp = self.spacy_nlp_es
            else:
                logger.warning("Spanish NLU components not fully loaded, trying English if available.")
        
        # Fallback if primary choice (based on hint or default 'es') failed
        if not active_intent_classifier or not active_spacy_nlp:
            if current_lang == "en" and self.intent_classifier_es and self.spacy_nlp_es: # Tried EN, failed, fallback to ES
                current_lang = "es"
                active_intent_classifier = self.intent_classifier_es
                active_spacy_nlp = self.spacy_nlp_es
                logger.info("Fell back to Spanish NLU components.")
            elif current_lang == "es" and self.intent_classifier_en and self.spacy_nlp_en: # Tried ES, failed, fallback to EN
                current_lang = "en"
                active_intent_classifier = self.intent_classifier_en
                active_spacy_nlp = self.spacy_nlp_en
                logger.info("Fell back to English NLU components.")
            else: # All attempts failed
                logger.error("Neither Spanish nor English NLU components are available.")
                return {
                    "final_response": "Lo siento, el procesamiento de lenguaje no está disponible.",
                    "intent_label": "NLU_UNAVAILABLE_ERROR", "plugin_used": plugin_used_for_output,
                    "merged_entities": [], "sentiment": None, "qa_result": None, "zero_shot_result": None,
                    "current_lang": current_lang, "empathetic_triggered": False,
                    "raw_spacy_doc_entities": [], "raw_hf_ner_entities": []}
        
        current_lang_for_output = current_lang
        logger.info(f"Language for NLU processing determined as: {current_lang} (Hint was: {lang_hint})")

        # --- 2. Tokenize Text using Custom Tokenizer ---
        tokens = self.custom_tokenizer(text, current_lang)
        if not tokens and text.strip(): # If text is not just whitespace but tokenizer returned empty
            logger.warning(f"Custom tokenizer returned no tokens for non-empty input: '{text}'")
            # If tokenizer fails, we can't proceed with Naive Bayes.
            # spaCy processing for entities might still work if it has a more robust tokenizer.
            recognized_intent_label = "NLU_TOKENIZER_FAILED_ERROR"
        
        # --- 3. Predict Intent using Custom Naive Bayes Classifier ---
        if tokens and active_intent_classifier: # Only predict if we have tokens and a classifier
            predicted_nb_intent = active_intent_classifier.predict(tokens)
            if predicted_nb_intent:
                recognized_intent_label = predicted_nb_intent
                logger.info(f"==> Custom Naive Bayes Recognized Intent: {recognized_intent_label}")
            else:
                recognized_intent_label = "UNKNOWN_INTENT_NB_FAILED"
                logger.warning(f"Custom Naive Bayes classifier failed to predict for tokens: {tokens}")
        elif not active_intent_classifier:
            logger.error(f"Custom intent classifier for language '{current_lang}' not loaded.")
            recognized_intent_label = "NLU_CLASSIFIER_MISSING_ERROR"
        # If tokens are empty, recognized_intent_label remains as initialized or NLU_TOKENIZER_FAILED_ERROR

        # --- 4. Process with Base spaCy for Linguistic Features & EntityRuler/BaseNER ---
        doc = None
        if active_spacy_nlp:
            try:
                doc = active_spacy_nlp(text) 
                logger.debug(f"Text processed with BASE spaCy model ({current_lang}) for entities/features.")
                # Optional Coreference (call if doc exists)
                # if self.context_manager:
                #    # ... (coref logic, ensure it handles doc being None if needed)
                ruler_entity_labels = {"TIME", "DATE", "PHONE", "WORK_OF_ART", "GPE"}
                if doc.ents: # Check if doc has ents
                    for ent in doc.ents:
                        source = 'ruler' if ent.label_ in ruler_entity_labels else 'spacy_base_ner'
                        spacy_doc_entities_for_output.append({
                            'text': ent.text, 'label': ent.label_,
                            'start_char': ent.start_char, 'end_char': ent.end_char, 'source': source})
                merged_entities_for_output = spacy_doc_entities_for_output
                logger.debug(f"Entities from spaCy: {merged_entities_for_output}")
            except Exception as e:
                logger.error(f"Error processing text with BASE spaCy model ({current_lang}): {e}", exc_info=True)
        else:
            logger.warning(f"BASE spaCy model for '{current_lang}' not loaded. SpaCy entities unavailable.")

        # --- 5. Handle Specific Internal Commands (These should return early) ---
        if recognized_intent_label == "INTENT_HELP": 
            final_response_str = self._get_help_string()
            plugin_used_for_output = "IntentProcessorInternal_Help"
            logger.info("Help intent recognized, providing help string.")
            return {
                "final_response": final_response_str, "intent_label": recognized_intent_label,
                "plugin_used": plugin_used_for_output, "merged_entities": merged_entities_for_output, 
                "sentiment": None, "qa_result": None, "zero_shot_result": None, 
                "current_lang": current_lang_for_output, "empathetic_triggered": False,
                "raw_spacy_doc_entities": spacy_doc_entities_for_output, "raw_hf_ner_entities": []}

        if recognized_intent_label == "INTENT_CLEAR_CONTEXT": 
            if self.context_manager: self.context_manager.clear_all_context()
            final_response_str = "Entendido, he limpiado el contexto de nuestra conversación."
            plugin_used_for_output = "IntentProcessorInternal_ClearContext"
            logger.info("Clear context intent recognized.")
            return {
                "final_response": final_response_str, "intent_label": recognized_intent_label,
                "plugin_used": plugin_used_for_output, "merged_entities": merged_entities_for_output, 
                "sentiment": None, "qa_result": None, "zero_shot_result": None, 
                "current_lang": current_lang_for_output, "empathetic_triggered": False,
                "raw_spacy_doc_entities": spacy_doc_entities_for_output, "raw_hf_ner_entities": []}
        
        # --- Handle other internal intents that DON'T return early ---
        # These set the response but allow flow to continue for sentiment, etc.
        internal_intent_handled = False
        if recognized_intent_label == "INTENT_GREET":
            greeting_responses_es = ["¡Hola! ¿Cómo puedo ayudarte?", "¡Qué tal! Dime qué necesitas.", "¡Hola! A tu servicio."]
            greeting_responses_en = ["Hello! How can I help you?", "Hi there! What can I do for you?", "Hello! At your service."]
            final_response_str = random.choice(greeting_responses_es if current_lang_for_output == "es" else greeting_responses_en)
            plugin_used_for_output = "IntentProcessorInternal_Greet"
            logger.info("Greet intent recognized, providing greeting.")
            internal_intent_handled = True
    
        elif recognized_intent_label == "INTENT_FAREWELL":
            farewell_responses_es = ["¡Hasta luego!", "Adiós, ¡que tengas un buen día!", "Nos vemos."]
            farewell_responses_en = ["Goodbye!", "Farewell, have a great day!", "See you!"]
            final_response_str = random.choice(farewell_responses_es if current_lang_for_output == "es" else farewell_responses_en)
            plugin_used_for_output = "IntentProcessorInternal_Farewell"
            logger.info("Farewell intent recognized, providing farewell.")
            internal_intent_handled = True
        
        # --- 6. Plugin Routing (Only if not handled by an internal intent above) ---
        if not internal_intent_handled and recognized_intent_label and recognized_intent_label in self.intent_to_plugin_map:
            chosen_plugin_name = self.intent_to_plugin_map[recognized_intent_label]
            if chosen_plugin_name and chosen_plugin_name in self.plugins:
                plugin_instance = self.plugins[chosen_plugin_name]
                logger.info(f"Intent '{recognized_intent_label}' routing to plugin '{chosen_plugin_name}'.")
                try:
                    handle_sig = inspect.signature(plugin_instance.handle)
                    handle_args = {}
                    current_processing_context = self.context_manager.get_context_for_processing() if self.context_manager else {}

                    if 'text' in handle_sig.parameters: handle_args['text'] = doc.text if doc else text 
                    if 'doc' in handle_sig.parameters: handle_args['doc'] = doc 
                    if 'context' in handle_sig.parameters: handle_args['context'] = current_processing_context
                    if 'entities' in handle_sig.parameters: handle_args['entities'] = merged_entities_for_output
                    
                    if self.context_manager:
                        self.context_manager.set_current_turn_data("recognized_intent_for_plugin", recognized_intent_label)
                        handle_args['context'] = self.context_manager.get_context_for_processing()

                    handle_result = plugin_instance.handle(**handle_args)
                    plugin_used_for_output = chosen_plugin_name 

                    if isinstance(handle_result, tuple) and len(handle_result) == 2:
                        plugin_response_text, updated_context_dict = handle_result
                        if isinstance(updated_context_dict, dict) and self.context_manager:
                            for key, value in updated_context_dict.items(): self.context_manager.set_current_turn_data(key, value)
                        final_response_str = plugin_response_text if plugin_response_text is not None else final_response_str # Use plugin response if not None
                    elif handle_result is not None:
                        final_response_str = handle_result # Use plugin response
                    else: 
                        logger.warning(f"Plugin '{chosen_plugin_name}' returned None for intent '{recognized_intent_label}'.")
                        final_response_str = f"Entendido ({recognized_intent_label.replace('INTENT_', '')}), pero no tengo una acción específica."
                        plugin_used_for_output = f"{chosen_plugin_name}_NoResponse"
                except Exception as e:
                    logger.error(f"Error during handling by plugin '{chosen_plugin_name}': {e}", exc_info=True)
                    final_response_str = f"Lo siento, el plugin {chosen_plugin_name} encontró un error."
                    plugin_used_for_output = f"{chosen_plugin_name}_Error"
            else: 
                logger.warning(f"Intent '{recognized_intent_label}' mapped to plugin '{chosen_plugin_name}', but plugin not loaded.")
        elif not internal_intent_handled and recognized_intent_label and \
            not recognized_intent_label.startswith("UNKNOWN_") and \
            not recognized_intent_label.startswith("NLU_"):
            logger.warning(f"Intent '{recognized_intent_label}' recognized but not in intent_to_plugin_map and not internal.")
            # final_response_str is already default "I don't know"

        # --- (OPTIONAL) Sentiment Analysis for Empathetic Response ---
        if self.advanced_nlp_processor and doc: 
            try:
                sentiment_result_for_output = self.advanced_nlp_processor.analyze_sentiment(doc.text, lang=current_lang)
                if sentiment_result_for_output and not sentiment_result_for_output.get("error"):
                    sentiment_label = sentiment_result_for_output.get('label')
                    sentiment_score = sentiment_result_for_output.get('score', 0)
                    negative_threshold = 0.7 
                    is_negative = (sentiment_label == 'NEGATIVE' or sentiment_label == 'NEG') and sentiment_score > negative_threshold
                    if is_negative and final_response_str: 
                        empathetic_phrase = "Entiendo que eso pueda ser frustrante. " if current_lang == "es" else "I understand that might be frustrating. "
                        if not final_response_str.lower().startswith(("lo siento", "i'm sorry", "entiendo que")):
                            final_response_str = empathetic_phrase + final_response_str
                            empathetic_triggered_for_output = True
            except Exception as e:
                logger.error(f"Error during Sentiment processing for empathetic response: {e}", exc_info=True)
                sentiment_result_for_output = {"error": "Sentiment analysis failed"}
        
        # --- 8. Final Output Construction ---
        logger.info(f"Final processing: Intent='{recognized_intent_label}', Plugin='{plugin_used_for_output}'")
        output_data = {
            "final_response": final_response_str,
            "intent_label": recognized_intent_label,
            "plugin_used": plugin_used_for_output,
            "merged_entities": merged_entities_for_output,
            "sentiment": sentiment_result_for_output,
            "qa_result": None, "zero_shot_result": None,
            "current_lang": current_lang_for_output,
            "empathetic_triggered": empathetic_triggered_for_output,
            "raw_spacy_doc_entities": spacy_doc_entities_for_output, 
            "raw_hf_ner_entities": []
        }
        return output_data

    def _resolve_coreference(self, doc, context):
        """
        Performs coreference resolution using spaCy or a rule-based approach.

        Args:
            doc (spacy.tokens.Doc): The spaCy Doc object for the current utterance
            context (dict): The current conversation context from ContextManager.

        Returns:
            spacy.tokens.Doc: The Doc object with resolved coreferences, or the original doc if resolution fails.
        """
        logger.debug("Attempting coreference resolution...")
        resolved_text = doc.text # Start with the original text
        current_entities = [ent for ent in doc.ents] # Get entities from the current doc

        # Get conversation history from ContextManager
        history = context.get("history", [])
        
        # Find the most recent user utterance with extracted entities
        last_user_utterance_entities = []
        for turn in reversed(history):
            if turn.get("speaker") == "user" and turn.get("entities"):
                last_user_utterance_entities = turn["entities"]
                logger.debug(f"Found last user utterance with entities: {turn['text']} -> {last_user_utterance_entities}")
                break

        if not last_user_utterance_entities:
            logger.debug("No previous user utterance with entities found for coreference resolution.")
            return doc # No history with entities to resolve against

        # Identify pronouns in the current utterance
        pronouns = [token for token in doc if token.pos_ == "PRON"]
        logger.debug(f"Identified pronouns in current utterance: {[p.text for p in pronouns]}")

        # Simple rule-based resolution
        # Iterate through pronouns and try to link them to entities in the last user utterance
        resolved_replacements = [] # Store (start_char, end_char, replacement_text)

        for pron in pronouns:
            best_match = None
            # Simple matching: look for an entity in the last user utterance
            # This is a very basic rule. More advanced rules would consider gender, number, type compatibility.
            for entity in last_user_utterance_entities:
                # Basic compatibility check (can be expanded)
                # For simplicity, let's assume 'it' can refer to non-person/non-location entities
                # and 'he'/'she' to persons, 'they' to plural entities.
                # This requires more sophisticated entity typing than spaCy's default or our ruler provides.
                # For now, let's just try to match any pronoun to the most recent entity.
                # A more robust approach needs entity type information in the stored context.

                # For this basic implementation, we'll just link the first pronoun to the most recent entity
                # This is highly simplistic and will fail often, but serves as a placeholder.
                best_match = entity
                break # Link the first pronoun to the first entity found in the last utterance (simplistic)

            if best_match:
                logger.debug(f"Attempting to resolve pronoun '{pron.text}' to entity '{best_match['text']}'")
                # Replace the pronoun in the resolved_text string
                # Need to be careful with character indices due to string manipulation
                # A better approach might be to build a new list of tokens or use spaCy's Token.set_text
                # For now, simple string replacement (might break indices of subsequent entities)
                # This is a known limitation of simple string replacement for coreference.

                # Simple string replacement (might affect subsequent entity indices)
                # Find the exact span in the original text
                start_char = pron.idx
                end_char = pron.idx + len(pron.text)
                
                # Perform replacement on the current resolved_text
                # This requires careful index tracking if multiple replacements happen.
                # For this basic version, let's just do one replacement for the first pronoun found.
                
                # A more robust way would be to collect all replacements and apply them from end to start
                resolved_replacements.append((start_char, end_char, best_match['text']))
                logger.debug(f"Scheduled replacement: '{pron.text}' at [{start_char}:{end_char}] with '{best_match['text']}'")
                break # Only resolve the first pronoun for this basic version

        # Apply replacements from end to start to avoid index issues
        resolved_replacements.sort(key=lambda x: x[0], reverse=True)
        
        modified_text = doc.text
        for start, end, replacement in resolved_replacements:
            modified_text = modified_text[:start] + replacement + modified_text[end:]
            logger.debug(f"Applied replacement: Replaced text from {start}:{end} with '{replacement}'. New text: '{modified_text}'")

        # Re-process the modified text with spaCy to get a new Doc object with updated entities/structure
        # This is necessary because string replacement invalidates the original Doc object's indices and structure.
        if modified_text != doc.text:
            logger.info("Text modified due to coreference resolution. Re-processing with spaCy.")
            # Use the appropriate language model
            if doc.lang_ == 'es' and self.nlp_es:
                resolved_doc = self.nlp_es(modified_text)
            elif doc.lang_ == 'en' and self.nlp_en:
                resolved_doc = self.nlp_en(modified_text)
            else:
                logger.warning(f"Could not re-process resolved text in language {doc.lang_}. Returning original doc.")
                resolved_doc = doc # Fallback to original doc if re-processing fails
        else:
            resolved_doc = doc # No changes were made, return original doc


        logger.debug("Coreference resolution completed.")
        return resolved_doc # Return the (potentially) resolved doc
