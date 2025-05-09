# jarvis/core/intent_processor.py
import logging
import importlib
import os
import re
import speech_recognition as sr
from pathlib import Path
import inspect # Added for signature checking
import spacy # Added for NLP
from .advanced_nlp import AdvancedNLPProcessor # Added for Hugging Face integration

logger = logging.getLogger(__name__)

class IntentProcessor:
    def __init__(self):
        """Inicializa el procesador de intenciones, carga plugins, modelos NLP y contexto."""
        self.plugins = {}
        self.context = {}  # Initialize conversation context
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
            logger.info(f"EntityRuler '{ruler_name}' already exists, updating patterns.")
        else:
            ruler = nlp.add_pipe("entity_ruler", name=ruler_name, before="ner")
            logger.info(f"EntityRuler '{ruler_name}' added before NER.")

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
            {"label": "WORK_OF_ART", "pattern": [{"LOWER": {"IN": ["canción", "album", "álbum"]}}, {"IS_TITLE": True, "OP": "+"}]},
            # Text in quotes (might capture other things too)
            {"label": "WORK_OF_ART", "pattern": [{"ORTH": '"'}, {"OP": "+"}, {"ORTH": '"'}]},
            {"label": "WORK_OF_ART", "pattern": [{"ORTH": "'"}, {"OP": "+"}, {"ORTH": "'"}]},
        ]

        # Overwrite patterns to ensure clean state for testing/reloading
        # ruler.add_patterns(patterns) # This might duplicate if run multiple times
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
        """Clears the conversation context."""
        self.context = {}
        logger.info("Contexto de conversación limpiado.")

    def load_plugins(self):
        """Carga dinámicamente los plugins desde el directorio plugins"""
        try:
            plugins_dir = Path(__file__).parent.parent / "plugins"
            plugin_files = [f for f in os.listdir(plugins_dir) if f.endswith('.py') and f != '__init__.py']

            for plugin_file in plugin_files:
                plugin_name = plugin_file[:-3]  # Quitar la extensión .py
                try:
                    module = importlib.import_module(f"jarvis.plugins.{plugin_name}")
                    if hasattr(module, 'Plugin'):
                        self.plugins[plugin_name] = module.Plugin()
                        logger.info(f"Plugin cargado: {plugin_name}")
                except Exception as e:
                    logger.error(f"Error al cargar plugin {plugin_name}: {str(e)}")
        except Exception as e:
            logger.error(f"Error al cargar plugins: {str(e)}")

    def process(self, text):
        """
        Procesa el texto para determinar la intención, actualiza el contexto
        y ejecuta el plugin correspondiente.

        Args:
            text (str): El texto del comando a procesar.

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
        current_lang = "es" # Default to Spanish, as it's the primary language

        processed_with_es = False
        if self.nlp_es:
            try:
                doc = self.nlp_es(text)
                current_lang = "es"
                processed_with_es = True
                logger.debug(f"Texto procesado con spaCy (es): {[token.text for token in doc]}")
                if doc.ents: logger.debug(f"Entidades detectadas (es): {[(ent.text, ent.label_) for ent in doc.ents]}")
            except Exception as e:
                logger.error(f"Error procesando texto con spaCy (es): {e}. Intentando inglés si está disponible.")
                doc = None # Reset doc if Spanish processing failed

        # Try English only if Spanish model is not available OR Spanish processing failed
        if not processed_with_es and self.nlp_en:
            try:
                doc = self.nlp_en(text)
                current_lang = "en" # Switch lang only if English processing succeeds
                logger.debug(f"Texto procesado con spaCy (en): {[token.text for token in doc]}")
                if doc.ents: logger.debug(f"Entidades detectadas (en): {[(ent.text, ent.label_) for ent in doc.ents]}")
            except Exception as e_en:
                 logger.error(f"Error procesando texto con spaCy (en) también: {e_en}")
                 doc = None # Reset doc if English also failed
                 # Keep current_lang as "es" (default) if both fail? Or set to None? Let's keep "es".

        logger.info(f"Determined language for advanced NLP (defaulting to 'es'): {current_lang}")

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
        # More sophisticated context logic will be needed.
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

        # Verificar cada plugin para ver si puede manejar esta intención
        # Pass context to can_handle and handle methods
        for name, plugin in self.plugins.items():
            try:
                # Check if plugin methods accept context using inspect
                can_handle_sig = inspect.signature(plugin.can_handle)
                handle_sig = inspect.signature(plugin.handle)

                # Check if plugin methods accept context, doc, and/or merged_entities
                can_handle_accepts_context = 'context' in can_handle_sig.parameters
                can_handle_accepts_doc = 'doc' in can_handle_sig.parameters
                can_handle_accepts_entities = 'entities' in can_handle_sig.parameters # New check
                handle_accepts_context = 'context' in handle_sig.parameters
                handle_accepts_doc = 'doc' in handle_sig.parameters
                handle_accepts_entities = 'entities' in handle_sig.parameters # New check

                # --- Prepare arguments for can_handle based on signature ---
                # Prioritize passing entities if accepted, then doc, then text
                can_handle_args = {}
                if can_handle_accepts_entities:
                    can_handle_args['entities'] = merged_entities
                if can_handle_accepts_doc and doc:
                     can_handle_args['doc'] = doc
                # Always pass text for now, plugins might still expect it
                can_handle_args['text'] = processed_text
                if can_handle_accepts_context:
                    can_handle_args['context'] = self.context

                # --- Check if plugin can handle ---
                # Use keyword arguments for clarity and flexibility
                if hasattr(plugin, 'can_handle') and plugin.can_handle(**can_handle_args):
                    logger.info(f"Plugin '{name}' can handle: '{text}'. Context: {self.context}")
                    plugin_used = name

                    # --- Prepare arguments for handle based on signature ---
                    handle_args = {}
                    if handle_accepts_entities:
                        handle_args['entities'] = merged_entities
                    if handle_accepts_doc and doc:
                        handle_args['doc'] = doc
                    # Always pass text
                    handle_args['text'] = processed_text
                    if handle_accepts_context:
                        handle_args['context'] = self.context

                    # --- Execute handle method ---
                    # Use keyword arguments
                    handle_result = plugin.handle(**handle_args)

                    # --- Process handle result (response and potential context update) ---
                    if isinstance(handle_result, tuple) and len(handle_result) == 2:
                        response, updated_context = handle_result
                        if isinstance(updated_context, dict):
                            self.context.update(updated_context) # Update processor's context
                            logger.info(f"Plugin '{name}' updated context: {updated_context}")
                        else:
                             logger.warning(f"Plugin '{name}' returned non-dict context: {updated_context}")
                    else:
                        response = handle_result # Assume only response was returned

                    intent_label_for_output = name # Set intent label to plugin name

                    break # Stop after finding the first handling plugin
            except Exception as e:
                 logger.error(f"Error checking or handling plugin {name} for text '{text}': {e}", exc_info=True)


        # Fallback handling if no plugin handled explicitly via can_handle
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
                    # TODO: Requires main loop/plugins to store 'last_user_utterance'/'last_assistant_response' in self.context
                    last_user = self.context.get('last_user_utterance', '')
                    last_agent = self.context.get('last_assistant_response', '')
                    base_context = "Contexto General: JARVIS es un asistente virtual que usa Python, spaCy y Transformers. JARVIS is a voice assistant using Python, spaCy, and Transformers."
                    # Combine context, prioritizing recent interaction
                    context_parts = [part for part in [last_user, last_agent, text, base_context] if part] # Filter empty strings
                    context_for_qa = "\n".join(context_parts)

                    question_part = text # Use the full text as the question for now

                    qa_result = self.advanced_nlp_processor.answer_question(question_part, context_for_qa, lang=current_lang)
                    qa_result_for_output = qa_result # Capture for output
                    logger.info(f"Hugging Face QA (lang={current_lang}) attempt for question '{question_part}': {qa_result}")

                    # Use the answer if confidence is high enough
                    confidence_threshold = 0.3 # Adjust as needed
                    if qa_result and not qa_result.get("error") and qa_result.get('score', 0) > confidence_threshold:
                       response = qa_result['answer']
                       plugin_used = f"AdvancedNLP_QA (lang={current_lang})"
                       intent_label_for_output = "qa_fallback" # Set intent label
                       logger.info(f"Using QA answer with score {qa_result.get('score')}")
                    else:
                       logger.info(f"QA score below threshold or error occurred. Score: {qa_result.get('score', 0)}")

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
                                handle_sig = inspect.signature(plugin.handle)
                                handle_accepts_context = 'context' in handle_sig.parameters
                                handle_accepts_doc = 'doc' in handle_sig.parameters
                                handle_args = []
                                if handle_accepts_doc and doc: handle_args.append(doc)
                                else: handle_args.append(processed_text)
                                if handle_accepts_context: handle_args.append(self.context)

                                # Execute handle method
                                handle_result = plugin.handle(*handle_args)

                                # Process result
                                if isinstance(handle_result, tuple) and len(handle_result) == 2:
                                    response, updated_context = handle_result
                                    if isinstance(updated_context, dict): self.context.update(updated_context)
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
                            # TODO: Update weather plugin to handle entities directly
                            handle_sig = inspect.signature(plugin.handle)
                            handle_accepts_context = 'context' in handle_sig.parameters
                            handle_args = (processed_text, self.context) if handle_accepts_context else (processed_text,)
                            handle_result = plugin.handle(*handle_args)
                            if isinstance(handle_result, tuple) and len(handle_result) == 2:
                                response, updated_context = handle_result
                                if isinstance(updated_context, dict):
                                    self.context.update(updated_context)
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
                            handle_sig = inspect.signature(plugin.handle)
                            handle_accepts_context = 'context' in handle_sig.parameters
                            handle_args = (processed_text, self.context) if handle_accepts_context else (processed_text,)
                            handle_result = plugin.handle(*handle_args)
                            if isinstance(handle_result, tuple) and len(handle_result) == 2:
                                response, updated_context = handle_result
                                if isinstance(updated_context, dict):
                                    self.context.update(updated_context)
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
                            handle_sig = inspect.signature(plugin.handle)
                            handle_accepts_context = 'context' in handle_sig.parameters
                            handle_args = (processed_text, self.context) if handle_accepts_context else (processed_text,)
                            handle_result = plugin.handle(*handle_args)
                            if isinstance(handle_result, tuple) and len(handle_result) == 2:
                                response, updated_context = handle_result
                                if isinstance(updated_context, dict):
                                    self.context.update(updated_context)
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
                            handle_sig = inspect.signature(plugin.handle)
                            handle_accepts_context = 'context' in handle_sig.parameters
                            handle_args = (processed_text, self.context) if handle_accepts_context else (processed_text,)
                            handle_result = plugin.handle(*handle_args)
                            if isinstance(handle_result, tuple) and len(handle_result) == 2:
                                response, updated_context = handle_result
                                if isinstance(updated_context, dict):
                                    self.context.update(updated_context)
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
            # Default response if no plugin or fallback worked
            final_response = "Lo siento, no sé cómo ayudarte con eso todavía."
            logger.warning(f"No plugin or fallback found for: '{text}'. Using default response.")

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
                      empathetic_triggered = True

        logger.info(f"Final response for '{text}'. Final Context: {self.context}")
        # TODO: Store current 'text' as 'last_user_utterance' and 'final_response' as 'last_assistant_response' in context for next turn. This should happen in the main loop after calling process().

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
