# train_nlp_models.py
import spacy
import random
from spacy.training.example import Example
from pathlib import Path

# --- CONFIGURATION ---
# (Eventually, load this from a config file or command-line args)
MODEL_LANG_ES = "es"
MODEL_LANG_EN = "en"
OUTPUT_DIR_ES = Path("models/nlp_es")
OUTPUT_DIR_EN = Path("models/nlp_en")
NUM_TRAIN_EPOCHS = 10 # Number of training iterations

# --- TRAINING DATA (Import or define here) ---
# (This would be your actual, complete list of utterances and intents)

# Spanish Intents
ALL_INTENT_LABELS_ES = [
    "INTENT_GET_WEATHER",
    "INTENT_PLAY_SONG",
    "INTENT_PLAY_ARTIST",
    "INTENT_PLAY_PLAYLIST",
    "INTENT_PLAY_MUSIC", # Added for generic requests
    "INTENT_SET_REMINDER",
    "INTENT_GET_NEWS",
    "INTENT_GREET",
    "INTENT_TELL_JOKE",
    "INTENT_ANSWER_QUESTION",
    "INTENT_OPEN_URL",
    "INTENT_SEARCH_WEB",
    "INTENT_SET_ALARM",
    "INTENT_GET_TIME",
    "INTENT_GET_DATE",
    "INTENT_STOP",
    "INTENT_CANCEL",
    "INTENT_HELP"
]
def create_cats_dict_es(target_intent):
    cats = {label: 0.0 for label in ALL_INTENT_LABELS_ES}
    if target_intent in cats: cats[target_intent] = 1.0
    return cats

TRAIN_DATA_ES_TEXTCAT = [
    # INTENT_GET_WEATHER
    ("¿Qué tiempo hace hoy?", {"cats": create_cats_dict_es("INTENT_GET_WEATHER")}),
    ("¿Qué tiempo hace en Londres?", {"cats": create_cats_dict_es("INTENT_GET_WEATHER")}),
    ("Dime el pronóstico del tiempo.", {"cats": create_cats_dict_es("INTENT_GET_WEATHER")}),
    ("¿Necesito un paraguas hoy?", {"cats": create_cats_dict_es("INTENT_GET_WEATHER")}),
    ("¿Cómo está el tiempo?", {"cats": create_cats_dict_es("INTENT_GET_WEATHER")}),
    ("¿Cuál es la temperatura exterior?", {"cats": create_cats_dict_es("INTENT_GET_WEATHER")}),
    ("¿Va a llover?", {"cats": create_cats_dict_es("INTENT_GET_WEATHER")}),
    ("¿Cuál es el pronóstico del tiempo para mañana?", {"cats": create_cats_dict_es("INTENT_GET_WEATHER")}),
    ("¿Qué tiempo hace en Nueva York?", {"cats": create_cats_dict_es("INTENT_GET_WEATHER")}),
    ("Dame el informe del tiempo.", {"cats": create_cats_dict_es("INTENT_GET_WEATHER")}),
    ("¿Cuál es la humedad hoy?", {"cats": create_cats_dict_es("INTENT_GET_WEATHER")}),
    ("¿Cuál es la velocidad del viento?", {"cats": create_cats_dict_es("INTENT_GET_WEATHER")}),
    ("¿Hace sol hoy?", {"cats": create_cats_dict_es("INTENT_GET_WEATHER")}),
    ("¿Cuál es la máxima y la mínima para hoy?", {"cats": create_cats_dict_es("INTENT_GET_WEATHER")}),
    ("¿Cómo va a estar el tiempo este fin de semana?", {"cats": create_cats_dict_es("INTENT_GET_WEATHER")}),

    # INTENT_PLAY_MUSIC (and its variants)
    # Example: "Pon algo de música." -> INTENT_PLAY_MUSIC (general)
    ("Pon algo de música.", {"cats": create_cats_dict_es("INTENT_PLAY_MUSIC")}),
    # Example: "Pon Bohemian Rhapsody." -> INTENT_PLAY_SONG (specific song)
    ("Pon Bohemian Rhapsody.", {"cats": create_cats_dict_es("INTENT_PLAY_SONG")}),
    # Example: "Pon algo de Queen." -> INTENT_PLAY_ARTIST (specific artist)
    ("Pon algo de Queen.", {"cats": create_cats_dict_es("INTENT_PLAY_ARTIST")}),
    # Example: "Pon mi lista de reproducción favorita." -> INTENT_PLAY_PLAYLIST (specific playlist)
    ("Pon mi lista de reproducción favorita.", {"cats": create_cats_dict_es("INTENT_PLAY_PLAYLIST")}),

    # INTENT_SET_REMINDER
    ("Recuérdame llamar a Juan a las 5pm.", {"cats": create_cats_dict_es("INTENT_SET_REMINDER")}),
    ("Pon un recordatorio para mañana a las 9am para ir al dentista.", {"cats": create_cats_dict_es("INTENT_SET_REMINDER")}),
    # ... ADD ALL YOUR SPANISH SAMPLE UTTERANCES HERE ...
]

# English Intents (Similar structure)
ALL_INTENT_LABELS_EN = [ # Assuming same for now
    "INTENT_GET_WEATHER",
    "INTENT_PLAY_SONG",
    "INTENT_PLAY_ARTIST",
    "INTENT_PLAY_PLAYLIST",
    "INTENT_PLAY_MUSIC",
    "INTENT_SET_REMINDER",
    "INTENT_GET_NEWS",
    "INTENT_GREET",
    "INTENT_TELL_JOKE",
    "INTENT_ANSWER_QUESTION",
    "INTENT_OPEN_URL",
    "INTENT_SEARCH_WEB",
    "INTENT_SET_ALARM",
    "INTENT_GET_TIME",
    "INTENT_GET_DATE",
    "INTENT_STOP",
    "INTENT_CANCEL",
    "INTENT_HELP"
] 
def create_cats_dict_en(target_intent):
    cats = {label: 0.0 for label in ALL_INTENT_LABELS_EN}
    if target_intent in cats: cats[target_intent] = 1.0
    return cats

TRAIN_DATA_EN_TEXTCAT = [
    ("What's the weather like today?", {"cats": create_cats_dict_en("INTENT_GET_WEATHER")}),
    ("What's the weather in London?", {"cats": create_cats_dict_en("INTENT_GET_WEATHER")}),
    ("Play some music.", {"cats": create_cats_dict_en("INTENT_PLAY_MUSIC")}),
    ("Play Bohemian Rhapsody.", {"cats": create_cats_dict_en("INTENT_PLAY_SONG")}),
    ("Play some Queen.", {"cats": create_cats_dict_en("INTENT_PLAY_ARTIST")}),
    ("Play my favorite playlist.", {"cats": create_cats_dict_en("INTENT_PLAY_PLAYLIST")}),
    ("Remind me to call John at 5pm.", {"cats": create_cats_dict_en("INTENT_SET_REMINDER")}),
    # ... ADD ALL YOUR ENGLISH SAMPLE UTTERANCES HERE ...
]

def train_textcat_model(lang, train_data, all_intent_labels, output_dir, model_name=None, base_model=None, epochs=NUM_TRAIN_EPOCHS):
    """Trains a new spaCy textcat model or updates an existing one."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    if model_name and (output_dir / model_name).exists():
        print(f"Loading existing model for language '{lang}' from {output_dir / model_name} for incremental update.")
        nlp = spacy.load(output_dir / model_name)
    elif base_model:
        print(f"Loading base model '{base_model}' for language '{lang}'.")
        nlp = spacy.load(base_model)
    else:
        print(f"Creating blank model for language '{lang}'.")
        nlp = spacy.blank(lang)

    # Add textcat to the pipeline if it doesn't exist
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.add_pipe("textcat", last=True)
        print(f"Added 'textcat' to pipeline for language '{lang}'.")
    else:
        textcat = nlp.get_pipe("textcat")
        print(f"'textcat' already in pipeline for language '{lang}'.")

    # Add labels to textcat
    for label in all_intent_labels:
        textcat.add_label(label)

    # Convert training data to Example objects
    examples = []
    for text, annotations in train_data:
        doc = nlp.make_doc(text)
        examples.append(Example.from_dict(doc, annotations))

    # Get names of other pipes to disable them during training
    pipe_exceptions = ["textcat", "tok2vec", "transformer"] # tok2vec or transformer might be needed by textcat
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    with nlp.disable_pipes(*other_pipes):  # Only train textcat
        optimizer = nlp.initialize(lambda: examples) # nlp.begin_training() is deprecated
        # optimizer = nlp.resume_training() # if updating

        print(f"Training textcat for language '{lang}' for {epochs} epochs...")
        for i in range(epochs):
            random.shuffle(examples)
            losses = {}
            # batch up the examples using spaCy's minibatch
            for batch in spacy.util.minibatch(examples, size=spacy.util.compounding(4.0, 32.0, 1.001)):
                nlp.update(batch, sgd=optimizer, losses=losses)
            print(f"Epoch {i+1}/{epochs} - Losses: {losses}")

    # Save the trained model
    if model_name:
        nlp.to_disk(output_dir / model_name)
        print(f"Saved trained model for language '{lang}' to {output_dir / model_name}")
    else: # Save to a default name if no specific model name provided (e.g., after initial training)
        nlp.to_disk(output_dir / "trained_textcat_model")
        print(f"Saved trained model for language '{lang}' to {output_dir / 'trained_textcat_model'}")

    return nlp

if __name__ == "__main__":
    # Train Spanish model
    if TRAIN_DATA_ES_TEXTCAT:
        print("\n--- Training Spanish Textcat Model ---")
        nlp_es_trained = train_textcat_model(
            MODEL_LANG_ES,
            TRAIN_DATA_ES_TEXTCAT,
            ALL_INTENT_LABELS_ES,
            OUTPUT_DIR_ES,
            model_name="jarvis_es_intent_model", # Name for the saved model directory
            base_model="es_core_news_sm" # Start with a base model for its tokenizer, vectors etc.
                                         # Or use "es_core_news_lg" for better vectors but slower
        )
        # Test the trained Spanish model (optional)
        if nlp_es_trained:
            test_text_es = "¿Cuál es el tiempo en Barcelona?"
            doc_es = nlp_es_trained(test_text_es)
            print(f"\nTest (ES): '{test_text_es}' -> Intents: {[(label, score) for label, score in doc_es.cats.items() if score > 0.1]}")

    # Train English model
    if TRAIN_DATA_EN_TEXTCAT:
        print("\n--- Training English Textcat Model ---")
        nlp_en_trained = train_textcat_model(
            MODEL_LANG_EN,
            TRAIN_DATA_EN_TEXTCAT,
            ALL_INTENT_LABELS_EN,
            OUTPUT_DIR_EN,
            model_name="jarvis_en_intent_model",
            base_model="en_core_web_sm" # Or "en_core_web_lg"
        )
        # Test the trained English model (optional)
        if nlp_en_trained:
            test_text_en = "What's the weather in Paris?"
            doc_en = nlp_en_trained(test_text_en)
            print(f"\nTest (EN): '{test_text_en}' -> Intents: {[(label, score) for label, score in doc_en.cats.items() if score > 0.1]}")

    print("\nTraining complete.")
