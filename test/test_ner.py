# test/test_ner.py
import spacy
import logging
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure the project root is in the path if running from outside
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# --- Copied/Adapted from IntentProcessor ---
def add_custom_entity_ruler(nlp): # Renamed function
    """Adds custom rules for various entities (TIME, DATE, PHONE, etc.) to a spaCy pipeline."""
    logger = logging.getLogger(__name__) # Use local logger
    logger.info("Adding custom EntityRuler for custom patterns...")
    ruler_name = "custom_entity_ruler" # Use matching name
    if ruler_name in nlp.pipe_names:
        ruler = nlp.get_pipe(ruler_name)
        logger.info(f"EntityRuler '{ruler_name}' already exists, updating patterns.")
    else:
        ruler = nlp.add_pipe("entity_ruler", name=ruler_name, before="ner")
        logger.info(f"EntityRuler '{ruler_name}' added before NER.")

    patterns = [
        {"label": "TIME", "pattern": [{"IS_DIGIT": True}, {"LOWER": {"IN": ["pm", "am"]}}]},
        {"label": "TIME", "pattern": [{"IS_DIGIT": True}, {"ORTH": ":"}, {"IS_DIGIT": True}]},
        {"label": "TIME", "pattern": [{"IS_DIGIT": True}, {"LOWER": "de"}, {"LOWER": "la"}, {"LOWER": {"IN": ["mañana", "tarde", "noche"]}}]},
        {"label": "DATE", "pattern": [{"LOWER": "mañana"}, {"LOWER": "por"}, {"LOWER": "la"}, {"LOWER": "tarde"}]},
        {"label": "TIME", "pattern": [{"IS_DIGIT": True}, {"LOWER": "en"}, {"LOWER": "punto"}]},
        # Add rule for just 'mañana' as DATE if needed, though model might get it
        {"label": "DATE", "pattern": [{"LOWER": "mañana"}]},
        {"label": "DATE", "pattern": [{"LOWER": "hoy"}]},
        {"label": "DATE", "pattern": [{"LOWER": "ayer"}]},
            {"label": "DATE", "pattern": [{"LOWER": "pasado"}, {"LOWER": "mañana"}]}, # pasado mañana

            # --- PHONE Number Patterns (Using Regex) ---
            # Matches ####-#### or ######## (Panama style)
            {"label": "PHONE", "pattern": [{"TEXT": {"REGEX": r"^\d{4}-?\d{4}$"}}]},

            # --- WORK_OF_ART Patterns (Tentative/Experimental) ---
            {"label": "WORK_OF_ART", "pattern": [{"LOWER": {"IN": ["canción", "album", "álbum"]}}, {"IS_TITLE": True, "OP": "+"}]},
        {"label": "WORK_OF_ART", "pattern": [{"ORTH": '"'}, {"OP": "+"}, {"ORTH": '"'}]},
        {"label": "WORK_OF_ART", "pattern": [{"ORTH": "'"}, {"OP": "+"}, {"ORTH": "'"}]},
    ]
    
    # Overwrite patterns to ensure clean state for testing
    ruler.initialize(lambda: [], nlp=nlp, patterns=patterns)
    
    logger.info(f"EntityRuler '{ruler_name}' initialized/updated with {len(patterns)} patterns.")
    return nlp
# --- End Copied/Adapted Section ---


def evaluate_spacy_ner(model_name="es_core_news_sm"):
    """Loads a spaCy model, adds custom rules, and evaluates NER on sample texts."""
    nlp = None
    logging.info(f"Attempting to load spaCy model: {model_name}")
    try:
        nlp = spacy.load(model_name)
        logging.info(f"Model '{model_name}' loaded successfully.")
    except OSError:
        logging.error(f"Model '{model_name}' not found. Please download it: python -m spacy download {model_name}")
        return
    except Exception as e:
        logging.error(f"An unexpected error occurred loading model {model_name}: {e}")
        return

    # Add custom rules after loading the model
    try:
        nlp = add_custom_entity_ruler(nlp) # Call renamed function
    except Exception as e:
         logging.error(f"Failed to add EntityRuler rules: {e}")
         # Continue without ruler if adding fails, but log error

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
        "Llama a Juan Pérez al 6677-8899" # Example with phone number
    ]

    logging.info("\n--- Evaluating spaCy NER Performance ---")
    for text in commands:
        try:
            doc = nlp(text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            print(f"\nText: {doc.text}")
            print(f"Entities: {entities}")
        except Exception as e:
            logging.error(f"Error processing text '{text}': {e}")

if __name__ == "__main__":
    # Default to Spanish model, can be overridden via command line arg if needed
    model_to_use = sys.argv[1] if len(sys.argv) > 1 else "es_core_news_sm"
    evaluate_spacy_ner(model_name=model_to_use)
