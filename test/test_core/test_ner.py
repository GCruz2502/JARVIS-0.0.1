# test_core/test_ner.py
import logging
import sys
from pathlib import Path
import spacy # Keep for type hinting or direct model loading if needed for other tests

# Attempt to import project modules.
# This setup assumes tests might be run directly or via a test runner from project root.
try:
    from core.intent_processor import IntentProcessor
    from core.context_manager import ContextManager
    from utils.config_manager import ConfigManager
    from utils.logger import setup_logging # To setup logging if run standalone
except ImportError:
    # Fallback for direct execution from 'test_core' or if PYTHONPATH is not set:
    # Add project root to sys.path.
    # Project root is parent of 'tests', which is parent of 'test_core'.
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from core.intent_processor import IntentProcessor
    from core.context_manager import ContextManager
    from utils.config_manager import ConfigManager
    from utils.logger import setup_logging

logger = logging.getLogger(__name__)

# Mock classes for dependencies not central to NER testing via IntentProcessor's nlp object
class MockConfigManager:
    def get_app_setting(self, key, default=None): return default
    def get_env_variable(self, key, default=None): return default
    def __init__(self, project_root_dir=None): pass # Allow project_root_dir

class MockContextManager:
    def __init__(self, max_history_len=0): pass
    def get_context_for_processing(self): return {}
    def add_utterance(self, speaker, text): pass
    def clear_all_context(self): pass
    def set_current_turn_data(self, key, value): pass
    def get_current_turn_data(self, key, default=None): return default


def evaluate_ner_via_intent_processor(model_lang="es"):
    """
    Evaluates NER by initializing IntentProcessor and using its spaCy instance.
    This tests the combined effect of the spaCy model and the custom entity ruler.
    """
    logger.info(f"--- Evaluating NER via IntentProcessor for language: {model_lang} ---")
    
    mock_config_manager = MockConfigManager()
    mock_context_manager = MockContextManager()

    try:
        # IntentProcessor's __init__ loads spaCy models and applies the custom entity ruler.
        intent_processor = IntentProcessor(
            context_manager=mock_context_manager, 
            config_manager=mock_config_manager
        )
    except Exception as e:
        logger.error(f"Failed to initialize IntentProcessor for NER test: {e}", exc_info=True)
        # Attempt to load spacy and AdvancedNLPProcessor for more detailed error
        try:
            spacy.load("es_core_news_sm")
            from core.nlp_engine import AdvancedNLPProcessor
            AdvancedNLPProcessor() # This might fail if models aren't downloaded
        except Exception as dep_e:
            logger.error(f"Underlying dependency error: {dep_e}", exc_info=True)
        return

    nlp_instance = None
    if model_lang == "es":
        nlp_instance = intent_processor.nlp_es
    elif model_lang == "en":
        nlp_instance = intent_processor.nlp_en
    
    if not nlp_instance:
        logger.error(f"spaCy model for lang '{model_lang}' not loaded in IntentProcessor. Aborting NER test.")
        logger.info("This might be due to spaCy models (e.g., 'es_core_news_sm' or 'en_core_web_sm') not being downloaded.")
        logger.info("Try: python -m spacy download es_core_news_sm")
        logger.info("And: python -m spacy download en_core_web_sm")
        return

    model_meta = nlp_instance.meta if hasattr(nlp_instance, 'meta') else {'name': 'unknown', 'version': 'unknown'}
    logger.info(f"Using spaCy model '{model_meta['name']}' (version {model_meta['version']}) via IntentProcessor.")

    # Test commands (same as before)
    commands = [
        "¿Cuál es el clima en Ciudad de Panamá mañana?",
        "Pon la canción Bohemian Rhapsody de Queen",
        "Recuérdame llamar a mamá a las 5 pm",
        "Toca algo de Rubén Blades",
        "Necesito ir a la farmacia Arrocha de Albrook Mall",
        "Programa una alarma para las 7 de la mañana",
        "Quiero escuchar el álbum Abbey Road",
        "¿Qué noticias hay sobre inteligencia artificial?",
        "Agrega leche a la lista de compras",
        "Llama a Juan Pérez al 6677-8899"
    ]

    print(f"\n--- Evaluating NER Performance on Language: {model_lang} ---")
    for text in commands:
        try:
            doc = nlp_instance(text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            print(f"\nText: {text}")
            print(f"Entities: {entities}")
        except Exception as e:
            logger.error(f"Error processing text '{text}' with spaCy model: {e}")

if __name__ == "__main__":
    # Setup logging for standalone script execution.
    # The main application calls setup_logging() in main.py.
    # Here, we call it directly for test script runs.
    # You might want to configure a different log file or level for tests.
    setup_logging(log_level_str="INFO", log_to_console=True, log_file_path=Path("test_ner_evaluation.log"))

    logger.info("Starting NER evaluation script...")
    
    # Evaluate Spanish NER (using default model configured in IntentProcessor)
    evaluate_ner_via_intent_processor(model_lang="es")
    
    # Optionally, evaluate English NER
    # logger.info("Evaluating English NER...")
    # evaluate_ner_via_intent_processor(model_lang="en")
    
    logger.info("NER evaluation script finished.")
