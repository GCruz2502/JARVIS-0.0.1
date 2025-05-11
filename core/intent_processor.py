# jarvis/core/intent_processor.py
import logging
import importlib
import os
import re
import speech_recognition as sr
from pathlib import Path
import inspect # Added for signature checking
import spacy # Added for NLP
from .nlp_engine import AdvancedNLPProcessor # Corrected: AdvancedNLPProcessor is now in nlp_engine
from .context_manager import ContextManager # Import ContextManager

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
        self.nlp_es = None
        self.nlp_en = None
        self.advanced_nlp_processor = None # Added for Hugging Face
        # Load models and add custom components like EntityRuler
        self._load_and_configure_spacy_models()
        self._initialize_advanced_nlp() # Added for Hugging Face
        self.load_plugins()

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
            ruler = nlp.add_pipe("entity_ruler", name=ruler_name, after="ner", config={"overwrite_ents": True})
            logger.info(f"EntityRuler '{ruler_name}' added after NER with overwrite_ents=True.")

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
        ]

        # Initialize the ruler with the new patterns. This will overwrite existing patterns if the ruler already exists.
        ruler.initialize(lambda: [], nlp=nlp, patterns=patterns)

        logger.info(f"EntityRuler '{ruler_name}' initialized/updated with {len(patterns)} patterns.")
        return nlp

    def _load_and_configure_spacy_models(self):
        """Loads spaCy models and adds custom components like EntityRuler."""
        # Load Spanish Model and add ruler
        try:
            # Try loading the large model first, fallback to small
            try:
                 self.nlp_es = spacy.load("es_core_news_lg")
                 logger.info("Modelo spaCy 'es_core_news_lg' cargado.")
            except OSError:
                 logger.warning("Modelo 'es_core_news_lg' no encontrado, intentando 'es_core_news_sm'.")
                 self.nlp_es = spacy.load("es_core_news_sm")
                 logger.info("Modelo spaCy 'es_core_news_sm' cargado.")

            # Add custom rules
            self.nlp_es = self._add_custom_entity_ruler(self.nlp_es) # Use renamed method

        except OSError:
            logger.error("Error al cargar 'es_core_news_sm'. Ejecuta: python -m spacy download es_core_news_sm")
        except Exception as e:
            logger.error(f"Error inesperado al cargar 'es_core_news_sm': {e}")

        try:
            self.nlp_en = spacy.load("en_core_web_sm")
            logger.info("Modelo spaCy 'en_core_web_sm' cargado.")
            # Add custom rules to the English model as well
            self.nlp_en = self._add_custom_entity_ruler(self.nlp_en)
        except OSError:
            logger.error("Error al cargar 'en_core_web_sm'. Ejecuta: python -m spacy download en_core_web_sm")
        except Exception as e:
            logger.error(f"Error inesperado al cargar 'en_core_web_sm': {e}")

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

    def process(self, text, lang_hint=None): # Added lang_hint
        """
        Procesa el texto para determinar la intención, actualiza el contexto
        y ejecuta el plugin correspondiente.

        Args:
            text (str): El texto del comando a procesar.
            lang_hint (str, optional): Una pista sobre el idioma ('en' o 'es').
                                       Si se proporciona, se prioriza este idioma.

        Returns:
            dict: Un diccionario con los resultados del procesamiento, incluyendo:
                  'final_response', 'intent_label', 'plugin_used', 'merged_entities',
                  'sentiment', 'qa_result', 'zero_shot_result', 'current_lang',
                  'empathetic_triggered', 'raw_spacy_doc_entities', 'raw_hf_ner_entities'.
        """
        response = None
        plugin_used = None
        intent_label_for_output = None # For the final identified intent/plugin
        qa_result_for_output = None
        zero_shot_result_for_output = None
        empathetic_triggered = False

        # --- Procesamiento NLP con spaCy ---
        doc = None
        current_lang = "es"  # Default
        
        if lang_hint == "en" and self.nlp_en:
            current_lang = "en"
            try:
                doc = self.nlp_en(text)
                logger.debug(f"Texto procesado con spaCy (en) basado en lang_hint: {[token.text for token in doc]}")
                if doc.ents: logger.debug(f"Entidades detectadas (en): {[(ent.text, ent.label_) for ent in doc.ents]}")
            except Exception as e:
                logger.error(f"Error procesando texto con spaCy (en) con lang_hint: {e}")
                doc = None
        elif lang_hint == "es" and self.nlp_es:
            current_lang = "es"
            try:
                doc = self.nlp_es(text)
                logger.debug(f"Texto procesado con spaCy (es) basado en lang_hint: {[token.text for token in doc]}")
                if doc.ents: logger.debug(f"Entidades detectadas (es): {[(ent.text, ent.label_) for ent in doc.ents]}")
            except Exception as e:
                logger.error(f"Error procesando texto con spaCy (es) con lang_hint: {e}")
                doc = None
        else: # Fallback to default logic if no valid hint or corresponding model
            processed_with_es = False
            if self.nlp_es:
                try:
                    doc = self.nlp_es(text)
                    current_lang = "es"
                    processed_with_es = True
                    logger.debug(f"Texto procesado con spaCy (es) - fallback: {[token.text for token in doc]}")
                    if doc.ents: logger.debug(f"Entidades detectadas (es) - fallback: {[(ent.text, ent.label_) for ent in doc.ents]}")
                except Exception as e:
                    logger.error(f"Error procesando texto con spaCy (es) - fallback: {e}. Intentando inglés si está disponible.")
                    doc = None

            if not processed_with_es and self.nlp_en:
                try:
                    doc = self.nlp_en(text)
                    current_lang = "en"
                    logger.debug(f"Texto procesado con spaCy (en) - fallback: {[token.text for token in doc]}")
                    if doc.ents: logger.debug(f"Entidades detectadas (en) - fallback: {[(ent.text, ent.label_) for ent in doc.ents]}")
                except Exception as e_en:
                     logger.error(f"Error procesando texto con spaCy (en) también - fallback: {e_en}")
                     doc = None
                     current_lang = "es" # Revert to default if EN also fails
        
        logger.info(f"Language for NLP processing: {current_lang} (Hint was: {lang_hint})")

        # --- Help Command Handling (Early Exit) ---
        if "ayuda" in text.lower() or "help" in text.lower():
            help_response = self._get_help_string()
            # Ensure 'doc' related entities are initialized if other parts of the return expect them
            spacy_entities_for_output = []
            if doc and doc.ents: # Check if doc and doc.ents exist
                ruler_entity_labels_help = {"TIME", "DATE", "PHONE", "WORK_OF_ART"} 
                for ent in doc.ents:
                    source = 'ruler' if ent.label_ in ruler_entity_labels_help else 'spacy_model'
                    spacy_entities_for_output.append({
                        'text': ent.text, 'label': ent.label_,
                        'start_char': ent.start_char, 'end_char': ent.end_char,
                        'source': source
                    })
            
            return {
                "final_response": help_response,
                "intent_label": "show_help",
                "plugin_used": "IntentProcessorInternal",
                "merged_entities": [], # No specific entities for help command itself
                "sentiment": None, 
                "qa_result": None,
                "zero_shot_result": None, 
                "current_lang": current_lang,
                "empathetic_triggered": False, 
                "raw_spacy_doc_entities": spacy_entities_for_output, # Provide empty or actual if doc was processed
                "raw_hf_ner_entities": [] # No HF NER for help command
            }
        # --- End Help Command Handling ---

        # --- Advanced NLP (Hugging Face) Integration ---
        sentiment_result = None # Store sentiment for later use
        hf_entities = [] # Store HF NER results
        if self.advanced_nlp_processor:
            # Sentiment Analysis
            try:
                sentiment_result = self.advanced_nlp_processor.analyze_sentiment(text, lang=current_lang)
                logger.info(f"Hugging Face Sentiment (lang={current_lang}) for '{text}': {sentiment_result}")
            except Exception as e:
                logger.error(f"Error during advanced NLP processing (Sentiment): {e}", exc_info=True)

            # HF NER Extraction
            try:
                hf_entities = self.advanced_nlp_processor.extract_entities_hf(text)
                if hf_entities: # Log only if entities were found
                     logger.info(f"Hugging Face NER Entities for '{text}': {hf_entities}")
            except Exception as e:
                 logger.error(f"Error during advanced NLP processing (HF NER): {e}", exc_info=True)

        # --- Consolidate Entities (Subtask 4) ---
        spacy_entities = []
        ruler_entity_labels = {"TIME", "DATE", "PHONE", "WORK_OF_ART"} # Labels added by our ruler

        if doc:
            for ent in doc.ents:
                source = 'ruler' if ent.label_ in ruler_entity_labels else 'spacy_model'
                spacy_entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start_char': ent.start_char,
                    'end_char': ent.end_char,
                    'source': source
                })

        hf_entities_formatted = []
        for ent in hf_entities:
             # HF pipeline format: {'entity_group': 'PER', 'score': ..., 'word': ..., 'start': ..., 'end': ...}
             hf_entities_formatted.append({
                 'text': ent['word'],
                 'label': ent['entity_group'], # Use entity_group for simplified label
                 'start_char': ent['start'],
                 'end_char': ent['end'],
                 'source': 'hf_ner',
                 'score': ent['score'] # Keep score for potential filtering later
             })

        logger.debug(f"spaCy+Ruler Entities: {spacy_entities}")
        logger.debug(f"HF NER Entities: {hf_entities_formatted}")

        # --- Merge Entities based on Priority (Ruler > HF NER > spaCy Model) ---
        merged_entities = []
        covered_spans = [] # List of (start, end) tuples

        def overlaps(start1, end1, start2, end2):
            """Checks if two spans overlap."""
            return max(start1, start2) < min(end1, end2)

        # 1. Add Ruler entities (highest priority)
        for ent in spacy_entities:
            if ent['source'] == 'ruler':
                merged_entities.append(ent)
                covered_spans.append((ent['start_char'], ent['end_char']))

        # 2. Add HF NER entities (if they don't overlap with ruler entities)
        for hf_ent in hf_entities_formatted:
            is_overlapped = False
            for r_start, r_end in covered_spans: # Check against already added ruler spans
                if overlaps(hf_ent['start_char'], hf_ent['end_char'], r_start, r_end):
                    is_overlapped = True
                    break
            if not is_overlapped:
                merged_entities.append(hf_ent)
                covered_spans.append((hf_ent['start_char'], hf_ent['end_char']))

        # 3. Add spaCy model entities (if they don't overlap with ruler or accepted HF entities)
        for ent in spacy_entities:
            if ent['source'] == 'spacy_model':
                is_overlapped = False
                for m_start, m_end in covered_spans: # Check against all previously added spans
                     if overlaps(ent['start_char'], ent['end_char'], m_start, m_end):
                        is_overlapped = True
                        break
                if not is_overlapped:
                    merged_entities.append(ent)
                    covered_spans.append((ent['start_char'], ent['end_char']))

        # Sort final list by start position
        merged_entities.sort(key=lambda x: x['start_char'])

        logger.debug(f"Final Merged Entities: {merged_entities}")
        # --- End Entity Consolidation ---

        # Note: QA and Zero-Shot logic are in the fallback section below.
        # --- End Advanced NLP Integration ---

        # Usar texto en minúsculas para compatibilidad con plugins existentes por ahora
        processed_text = text.lower()

        # --- Context Management Example (Simple: Clear context on 'clear context' command) ---
        # This now calls the modified clear_context which uses ContextManager
        # Check raw text for clear context command before processing further
        if "limpiar contexto" in text.lower() or "olvida todo" in text.lower():
             self.clear_context() 
             # Return the full dictionary structure even for this command
             return {
                 "final_response": "Entendido, he limpiado el contexto de nuestra conversación.",
                 "intent_label": "clear_context",
                 "plugin_used": "IntentProcessorInternal",
                 "merged_entities": [],
                 "sentiment": None,
                 "qa_result": None,
                 "zero_shot_result": None,
                 "current_lang": current_lang, # Use determined lang
                 "empathetic_triggered": False,
                 "raw_spacy_doc_entities": [],
                 "raw_hf_ner_entities": []
             }
        # --- End Context Management Example ---

        # --- Plugin Handling Logic ---
        candidate_plugins = []
        for name, plugin_instance in self.plugins.items():
            try:
                can_handle_sig = inspect.signature(plugin_instance.can_handle)
                can_handle_args = {}
                current_processing_context = self.context_manager.get_context_for_processing() if self.context_manager else {}
                
                if 'entities' in can_handle_sig.parameters: can_handle_args['entities'] = merged_entities
                if 'doc' in can_handle_sig.parameters and doc: can_handle_args['doc'] = doc
                can_handle_args['text'] = processed_text # Always pass text
                if 'context' in can_handle_sig.parameters: can_handle_args['context'] = current_processing_context

                if hasattr(plugin_instance, 'can_handle') and plugin_instance.can_handle(**can_handle_args):
                    candidate_plugins.append(name)
            except Exception as e:
                logger.error(f"Error checking can_handle for plugin {name}: {e}", exc_info=True)

        chosen_plugin_name = None
        if len(candidate_plugins) == 1:
            chosen_plugin_name = candidate_plugins[0]
            logger.info(f"Single candidate plugin: {chosen_plugin_name}")
        elif len(candidate_plugins) > 1:
            logger.info(f"Multiple candidate plugins: {candidate_plugins}. Using zero-shot for disambiguation.")
            if self.advanced_nlp_processor and self.advanced_nlp_processor.zero_shot_classifier:
                # Use plugin names (or more descriptive labels mapped to them) as candidates
                # For now, using plugin names directly. This might need refinement.
                # The intent_labels_map could be enhanced to map plugin names to descriptive phrases.
                # For TC008, candidate_plugins would be ['music', 'reminders']
                # We need to map these back to descriptive phrases for zero-shot if possible,
                # or ensure zero-shot can work with plugin names as labels.
                # Let's assume for now we have a way to get descriptive labels for these plugins.
                # This part needs a robust mapping from plugin name to a descriptive label for zero-shot.
                # For simplicity in this step, I'll use a placeholder mapping.
                # A more robust solution would be for plugins to declare their "zero-shot phrases".
                
                plugin_to_phrase_map = {
                    "weather": "check weather",
                    "music": "play music",
                    "reminders": "set reminder" 
                    # Add other plugins here
                }
                zeroshot_candidate_labels = [plugin_to_phrase_map.get(p_name, p_name) for p_name in candidate_plugins if plugin_to_phrase_map.get(p_name)]
                
                if zeroshot_candidate_labels:
                    classification_result = self.advanced_nlp_processor.classify_intent(text, zeroshot_candidate_labels)
                    zero_shot_result_for_output = classification_result # Capture for output
                    logger.info(f"Zero-shot for disambiguation among {candidate_plugins} (using labels {zeroshot_candidate_labels}): {classification_result}")

                    if classification_result and not classification_result.get("error") and classification_result['scores'][0] > 0.5: # Stricter threshold for disambiguation
                        top_phrase = classification_result['labels'][0]
                        # Find which plugin name corresponds to this top_phrase
                        for p_name, phrase in plugin_to_phrase_map.items():
                            if phrase == top_phrase and p_name in candidate_plugins:
                                chosen_plugin_name = p_name
                                logger.info(f"Zero-shot chose '{chosen_plugin_name}' from candidates with score {classification_result['scores'][0]}.")
                                break
                    else:
                        logger.warning(f"Zero-shot disambiguation failed or score too low. Candidates: {candidate_plugins}")
                else:
                    logger.warning(f"Could not map candidate plugins {candidate_plugins} to descriptive phrases for zero-shot.")

            if not chosen_plugin_name and candidate_plugins: # Fallback if zero-shot fails to pick one
                 logger.warning(f"Zero-shot failed to disambiguate. Picking first candidate: {candidate_plugins[0]}")
                 # chosen_plugin_name = candidate_plugins[0] # Or handle as ambiguous / error
                 # For now, let's not pick one if zero-shot fails, to let it go to general fallback.
                 pass


        if chosen_plugin_name:
            plugin = self.plugins[chosen_plugin_name]
            name = chosen_plugin_name # for logging and output
            try:
                handle_sig = inspect.signature(plugin.handle)
                handle_args = {}
                current_processing_context = self.context_manager.get_context_for_processing() if self.context_manager else {}

                if 'entities' in handle_sig.parameters: handle_args['entities'] = merged_entities
                if 'doc' in handle_sig.parameters and doc: handle_args['doc'] = doc
                handle_args['text'] = processed_text # Always pass text
                if 'context' in handle_sig.parameters: handle_args['context'] = current_processing_context
                
                plugin_used = name
                handle_result = plugin.handle(**handle_args)

                if isinstance(handle_result, tuple) and len(handle_result) == 2:
                    response, updated_context_dict = handle_result # Renamed for clarity
                    if isinstance(updated_context_dict, dict) and self.context_manager:
                        for key, value in updated_context_dict.items():
                            self.context_manager.set_current_turn_data(key, value)
                        logger.info(f"Plugin '{name}' updated context via ContextManager: {updated_context_dict}")
                    elif not self.context_manager:
                        logger.warning(f"Plugin '{name}' returned context but no ContextManager available in IntentProcessor.")
                    else: # Not a dict
                        logger.warning(f"Plugin '{name}' returned non-dict context: {updated_context_dict}")
                else:
                    response = handle_result
                intent_label_for_output = name
            except Exception as e:
                logger.error(f"Error handling plugin {name} for text '{text}': {e}", exc_info=True)
        
        # Fallback handling if no plugin chosen or chosen plugin failed
        if response is None:
            logger.debug(f"No plugin handled '{text}' via can_handle. Trying fallbacks.")

            # --- Advanced QA Fallback ---
            # Trigger QA if it looks like a question and no plugin handled it.
            is_question = text.strip().endswith("?") or \
                          text.lower().startswith(("what", "who", "where", "when", "why", "how", "which", \
                                                   "qué", "quién", "dónde", "cuándo", "cómo", "cuál", "explica"))

            if is_question and self.advanced_nlp_processor and self.advanced_nlp_processor.qa_pipeline_es and current_lang == "es" or \
               is_question and self.advanced_nlp_processor and self.advanced_nlp_processor.qa_pipeline_en and current_lang == "en":
                try:
                    # Improved dynamic context for QA
                    current_processing_context = self.context_manager.get_context_for_processing() if self.context_manager else {}
                    qa_override_context = current_processing_context.get('qa_context_override') # Use get_context_for_processing result

                    if qa_override_context:
                        context_for_qa = qa_override_context
                        logger.info(f"Using qa_context_override for QA: {context_for_qa}")
                    else:
                        # These now come from ContextManager's structured context
                        last_user = current_processing_context.get('last_user_utterance', '')
                        last_agent = current_processing_context.get('last_assistant_response', '')
                        # previous_user = current_processing_context.get('previous_user_utterance', '') # Example if needed
                        # previous_agent = current_processing_context.get('previous_assistant_response', '') # Example if needed

                        base_context = "Contexto General: JARVIS es un asistente virtual que usa Python, spaCy y Transformers. JARVIS is a voice assistant using Python, spaCy, and Transformers."
                        # Combine context, prioritizing recent interaction
                        context_parts = [part for part in [last_user, last_agent, text, base_context] if part] # Filter empty strings
                        context_for_qa = "\n".join(context_parts)
                        logger.info(f"Using constructed context for QA: {context_for_qa}")
                    
                    question_part = text # Use the full text as the question for now

                    qa_result_attempt = self.advanced_nlp_processor.answer_question(question_part, context_for_qa, lang=current_lang)
                    logger.info(f"Hugging Face QA (lang={current_lang}) attempt for question '{question_part}': {qa_result_attempt}")

                    # Use the answer if confidence is high enough
                    confidence_threshold = 0.3 # Adjust as needed
                    if qa_result_attempt and not qa_result_attempt.get("error") and qa_result_attempt.get('score', 0) > confidence_threshold:
                       response = qa_result_attempt['answer']
                       plugin_used = f"AdvancedNLP_QA (lang={current_lang})"
                       intent_label_for_output = "qa_fallback" # Set intent label
                       qa_result_for_output = qa_result_attempt # Capture for output ONLY if used
                       logger.info(f"Using QA answer with score {qa_result_attempt.get('score')}")
                    else:
                       logger.info(f"QA score below threshold or error occurred. Score: {qa_result_attempt.get('score', 0)}. QA result will not be used for final response or output.")
                       # qa_result_for_output remains None or its previous value if not used

                except Exception as e:
                    logger.error(f"Error during advanced NLP QA fallback: {e}", exc_info=True)

            # --- Zero-Shot Classification Fallback (Intent Recognition Model - Subtask 3) ---
            if response is None and self.advanced_nlp_processor and self.advanced_nlp_processor.zero_shot_classifier:
                try:
                    # Define more nuanced candidate intent labels
                    # Map these labels back to the plugin keys responsible for handling them
                    intent_labels_map = {
                        # Weather Intents
                        "check current weather": "weather",
                        "ask for weather forecast": "weather",
                        "check temperature": "weather",
                        "ask about rain": "weather",
                        # Music Intents
                        "play music": "music",
                        "play specific song": "music",
                        "play artist": "music",
                        "pause music": "music", # Assuming music plugin handles pause
                        "stop music": "music",  # Assuming music plugin handles stop
                        # Add other potential general intents or plugin-specific intents
                        "tell joke": "jokes", # Example for a hypothetical 'jokes' plugin
                        "set reminder": "reminders", # Example
                        "ask general question": None, # Could be handled by QA or specific plugin
                        "greeting": None, # Could be handled by a specific plugin or default response
                        "farewell": None,
                        "unknown": None,
                    }
                    candidate_labels = list(intent_labels_map.keys())

                    if candidate_labels:
                        classification_result = self.advanced_nlp_processor.classify_intent(text, candidate_labels)
                        zero_shot_result_for_output = classification_result # Capture for output
                        logger.info(f"Hugging Face Zero-Shot Classification for '{text}': {classification_result}")

                        if classification_result and not classification_result.get("error"):
                            top_label = classification_result['labels'][0]
                            top_score = classification_result['scores'][0]
                            confidence_threshold = 0.6 # Slightly lower threshold maybe, as labels are more specific? Adjust as needed.

                            # Get the target plugin key from the classified intent label
                            plugin_key = intent_labels_map.get(top_label)

                            if plugin_key and top_score > confidence_threshold and plugin_key in self.plugins:
                                logger.info(f"Zero-shot classification identified intent '{top_label}' -> plugin '{plugin_key}' with score {top_score}. Attempting to handle.")
                                plugin = self.plugins[plugin_key]

                                # Prepare arguments for handle (similar to primary loop)
                                current_processing_context = self.context_manager.get_context_for_processing() if self.context_manager else {}
                                handle_sig = inspect.signature(plugin.handle)
                                handle_args_zs = {} # Use dict for keyword args
                                if 'entities' in handle_sig.parameters: handle_args_zs['entities'] = merged_entities
                                if 'doc' in handle_sig.parameters and doc: handle_args_zs['doc'] = doc
                                handle_args_zs['text'] = processed_text
                                if 'context' in handle_sig.parameters: handle_args_zs['context'] = current_processing_context
                                
                                # Execute handle method
                                handle_result = plugin.handle(**handle_args_zs)

                                # Process result
                                if isinstance(handle_result, tuple) and len(handle_result) == 2:
                                    response, updated_context_dict = handle_result # Renamed
                                    if isinstance(updated_context_dict, dict) and self.context_manager:
                                        for key, value in updated_context_dict.items():
                                            self.context_manager.set_current_turn_data(key, value)
                                        logger.info(f"Plugin '{plugin_key}' (ZeroShot) updated context via ContextManager: {updated_context_dict}")
                                    elif not self.context_manager:
                                         logger.warning(f"Plugin '{plugin_key}' (ZeroShot) returned context but no ContextManager.")
                                    else: # Not a dict
                                         logger.warning(f"Plugin '{plugin_key}' (ZeroShot) returned non-dict context: {updated_context_dict}")
                                else:
                                    response = handle_result
                                plugin_used = f"{plugin_key} (ZeroShot fallback)"
                                intent_label_for_output = plugin_key # Set intent label from zero-shot
                            else:
                                logger.info(f"Zero-shot classification score ({top_score}) below threshold or label '{top_label}' not mapped to known plugin.")
                    else:
                         logger.warning("Zero-shot classification failed or returned error.")

                except Exception as e:
                    logger.error(f"Error during advanced NLP Zero-Shot fallback: {e}", exc_info=True)


            # --- spaCy Keyword/Entity Fallback Logic (if QA and ZeroShot didn't provide an answer) ---
            if response is None and doc: # Use spaCy doc if available and other fallbacks failed
                # Weather fallback using entities
                weather_keywords = {"tiempo", "clima", "lluvia", "temperatura", "pronóstico"}
                # Check for weather keywords (using lemmas for robustness) AND location entities
                has_weather_keyword = any(token.lemma_ in weather_keywords for token in doc)
                location_entities = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]]

                if has_weather_keyword and location_entities:
                     if "weather" in self.plugins:
                        try:
                            plugin = self.plugins["weather"]
                            # Pass the original text and context for now, plugin needs update later
                            current_processing_context = self.context_manager.get_context_for_processing() if self.context_manager else {}
                            handle_sig = inspect.signature(plugin.handle)
                            handle_args_spacy_fb = {}
                            if 'entities' in handle_sig.parameters: handle_args_spacy_fb['entities'] = merged_entities # Or just doc.ents for this specific fallback
                            if 'doc' in handle_sig.parameters and doc: handle_args_spacy_fb['doc'] = doc
                            handle_args_spacy_fb['text'] = processed_text
                            if 'context' in handle_sig.parameters: handle_args_spacy_fb['context'] = current_processing_context

                            handle_result = plugin.handle(**handle_args_spacy_fb)
                            if isinstance(handle_result, tuple) and len(handle_result) == 2:
                                response, updated_context_dict = handle_result
                                if isinstance(updated_context_dict, dict) and self.context_manager:
                                     for key, value in updated_context_dict.items():
                                        self.context_manager.set_current_turn_data(key, value)
                                else: # No context manager or not a dict
                                     pass # Logged by previous checks if needed
                            else:
                                response = handle_result
                            plugin_used = "weather (spaCy fallback)"
                        except Exception as e:
                            logger.error(f"Error in spaCy fallback handling for weather plugin: {e}", exc_info=True)

                # Add other spaCy-based fallbacks here (e.g., for news, reminders)
                # elif ... :
                #    pass

            # --- Original Keyword Fallback (if spaCy doc not available or didn't match) ---
            if response is None: # Check again if spaCy fallback didn't set a response
                logger.debug(f"spaCy fallback did not handle '{text}'. Trying original keyword fallbacks.")
                # Pass context to fallback handlers too
                if any(word in processed_text for word in ["tiempo", "clima", "lluvia", "temperatura"]):
                    if "weather" in self.plugins:
                        try:
                            plugin = self.plugins["weather"]
                            current_processing_context = self.context_manager.get_context_for_processing() if self.context_manager else {}
                            handle_sig = inspect.signature(plugin.handle)
                            handle_args_kw_fb = {'text': processed_text}
                            if 'context' in handle_sig.parameters: handle_args_kw_fb['context'] = current_processing_context
                            # Keyword fallbacks usually don't get doc/entities unless they are more advanced

                            handle_result = plugin.handle(**handle_args_kw_fb)
                            if isinstance(handle_result, tuple) and len(handle_result) == 2:
                                response, updated_context_dict = handle_result
                                if isinstance(updated_context_dict, dict) and self.context_manager:
                                    for key, value in updated_context_dict.items():
                                        self.context_manager.set_current_turn_data(key, value)
                            else:
                                response = handle_result
                            plugin_used = "weather (keyword fallback)"
                        except Exception as e:
                            logger.error(f"Error in keyword fallback handling for weather plugin: {e}", exc_info=True)

                # Add similar fallback checks for news, reminders etc. if needed, passing context
                # Example for a hypothetical 'news' plugin:
                elif any(word in processed_text for word in ["noticia", "noticias", "actualidad"]):
                     if "news" in self.plugins:
                        try:
                            plugin = self.plugins["news"]
                            current_processing_context = self.context_manager.get_context_for_processing() if self.context_manager else {}
                            handle_sig = inspect.signature(plugin.handle)
                            handle_args_kw_fb_news = {'text': processed_text}
                            if 'context' in handle_sig.parameters: handle_args_kw_fb_news['context'] = current_processing_context
                            
                            handle_result = plugin.handle(**handle_args_kw_fb_news)
                            if isinstance(handle_result, tuple) and len(handle_result) == 2:
                                response, updated_context_dict = handle_result
                                if isinstance(updated_context_dict, dict) and self.context_manager:
                                    for key, value in updated_context_dict.items():
                                        self.context_manager.set_current_turn_data(key, value)
                            else:
                                response = handle_result
                            plugin_used = "news (keyword fallback)"
                        except Exception as e:
                            logger.error(f"Error in keyword fallback handling for news plugin: {e}", exc_info=True)

                # Example for a hypothetical 'reminders' plugin:
                elif any(word in processed_text for word in ["recuerda", "recordatorio", "agenda", "evento"]):
                     if "reminders" in self.plugins:
                        try:
                            plugin = self.plugins["reminders"]
                            current_processing_context = self.context_manager.get_context_for_processing() if self.context_manager else {}
                            handle_sig = inspect.signature(plugin.handle)
                            handle_args_kw_fb_rem = {'text': processed_text}
                            if 'context' in handle_sig.parameters: handle_args_kw_fb_rem['context'] = current_processing_context

                            handle_result = plugin.handle(**handle_args_kw_fb_rem)
                            if isinstance(handle_result, tuple) and len(handle_result) == 2:
                                response, updated_context_dict = handle_result
                                if isinstance(updated_context_dict, dict) and self.context_manager:
                                    for key, value in updated_context_dict.items():
                                        self.context_manager.set_current_turn_data(key, value)
                            else:
                                response = handle_result
                            plugin_used = "reminders (keyword fallback)"
                        except Exception as e:
                            logger.error(f"Error in keyword fallback handling for reminders plugin: {e}", exc_info=True)


        # Final response generation / modification based on sentiment
        final_response = None
        if response is not None:
            final_response = response
            logger.info(f"Plugin '{plugin_used}' generated response for '{text}'.")
        else:
            # --- Final Fallback: General Chat (Now via Ollama) ---
            if self.advanced_nlp_processor: # Check if advanced_nlp_processor itself is initialized
                logger.info(f"No plugin handled '{text}'. Attempting general chat fallback via Ollama.")
                # The model_tag used by Ollama can be passed if needed, or use default from generate_chat_response
                # Example: chat_response = self.advanced_nlp_processor.generate_chat_response(text, model_tag="llama3:8b-instruct")
                chat_response = self.advanced_nlp_processor.generate_chat_response(text) # Uses default model_tag from method
                
                # Standardized fallback messages from Ollama method if it fails or gives empty response
                standard_fallback_messages = {
                    "Lo siento, mi capacidad de chat general no está disponible en este momento.",
                    "No estoy seguro de cómo responder a eso.",
                    "No se me ocurre qué decir.",
                    "Lo siento, el servicio de chat tardó demasiado en responder.",
                    "Lo siento, no pude conectarme al servicio de chat. Asegúrate de que Ollama esté en ejecución.",
                    "Lo siento, tuve un problema al comunicarme con el servicio de chat.",
                    "Lo siento, tuve un problema inesperado al generar una respuesta de chat."
                }

                if chat_response and chat_response not in standard_fallback_messages:
                    final_response = chat_response
                    plugin_used = "GeneralChatFallback_Ollama" 
                    intent_label_for_output = "general_chat"
                    logger.info(f"General chat fallback (Ollama) provided response: '{final_response}'")
                else:
                    logger.warning(f"General chat fallback (Ollama) did not provide a suitable response for '{text}'. Response: '{chat_response}'")
            
            if final_response is None:
                # Default response if no plugin or any fallback (including chat) worked
                final_response = "Lo siento, no sé cómo ayudarte con eso todavía."
                logger.warning(f"No plugin or any fallback found for: '{text}'. Using default response.")

        # Modify response based on sentiment (Example)
        if sentiment_result and not sentiment_result.get("error"):
             sentiment_label = sentiment_result.get('label')
             sentiment_score = sentiment_result.get('score', 0)
             # Adjust threshold and labels as needed (Spanish model uses NEG, NEU, POS)
             negative_threshold = 0.7
             is_negative = (sentiment_label == 'NEGATIVE' or sentiment_label == 'NEG') and sentiment_score > negative_threshold

             if is_negative:
                 logger.info(f"Detected negative sentiment (Label: {sentiment_label}, Score: {sentiment_score}). Prepending empathetic phrase.")
                 empathetic_phrase = "Entiendo que eso pueda ser frustrante. " if current_lang == "es" else "I understand that might be frustrating. "
                 # Avoid prepending if the response already seems to handle negativity
                 if not final_response.lower().startswith(("lo siento", "i'm sorry")):
                      final_response = empathetic_phrase + final_response
                      empathetic_triggered = True # Ensure this flag is set

        final_context_for_log = self.context_manager.get_context_for_processing() if self.context_manager else "ContextManager not available"
        logger.info(f"Final response for '{text}'. Final Context (from ContextManager): {final_context_for_log}")
        # Note: add_utterance for user and assistant response is now handled by the caller (e.g., cli_interface)
        # using the context_manager instance.

        # Construct the detailed output dictionary
        output_data = {
            "final_response": final_response,
            "intent_label": intent_label_for_output,
            "plugin_used": plugin_used,
            "merged_entities": merged_entities, # Already a list of dicts
            "sentiment": sentiment_result, # Already a dict or None
            "qa_result": qa_result_for_output, # Already a dict or None
            "zero_shot_result": zero_shot_result_for_output, # Already a dict or None
            "current_lang": current_lang,
            "empathetic_triggered": empathetic_triggered,
            "raw_spacy_doc_entities": spacy_entities, # List of dicts
            "raw_hf_ner_entities": hf_entities_formatted # List of dicts
        }
        return output_data
