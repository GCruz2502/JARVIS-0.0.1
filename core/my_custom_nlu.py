import re
import logging
from collections import Counter, defaultdict
import math
import json
import os

# --- This is the new import ---
# It assumes your project root is in the Python path.
# When running with `python -m core.my_custom_nlu`, this will work correctly.
try:
    from training.intent_training_data import TRAINING_SAMPLES_ES, TRAINING_SAMPLES_EN
except ImportError:
    # This block allows the file to be run directly for testing IF it can't find the module,
    # but it will print a prominent error. It's a safety measure.
    print("ERROR: Could not import training data. Make sure you are running this script from the project root directory using 'python -m core.my_custom_nlu'")
    TRAINING_SAMPLES_ES = []
    TRAINING_SAMPLES_EN = []


logger = logging.getLogger(__name__)

def tokenize(text: str, lang: str = "es") -> list[str]:
    """
    Simple tokenizer for text.
    - Converts to lowercase.
    - Removes most punctuation, keeping alphanumeric words.
    - Handles some basic language-specific aspects if needed later.
    """
    if not isinstance(text, str):
        logger.error(f"Tokenizer received non-string input: {type(text)}")
        return []

    text = text.lower() # Convert to lowercase

    if lang == "en":
        tokens = re.findall(r"\b[a-z0-9']+\b", text)
    elif lang == "es":
        text = text.strip("¿¡")
        tokens = re.findall(r"\b[a-z0-9ñáéíóúü]+\b", text) # Added Spanish characters
    else:
        logger.warning(f"Unsupported language for tokenization: {lang}. Defaulting to basic word split.")
        tokens = re.findall(r"\b\w+\b", text)

    logger.debug(f"Original text: '{text}' -> Tokens: {tokens} (lang: {lang})")
    return tokens

def build_vocabulary(all_training_tokens: list[list[str]]) -> dict:
    """
    Builds a vocabulary from a list of tokenized training sentences.
    """
    all_words = []
    for sentence_tokens in all_training_tokens:
        all_words.extend(sentence_tokens)

    unique_words = sorted(list(set(all_words)))
    vocabulary = {word: i for i, word in enumerate(unique_words)}

    logger.info(f"Built vocabulary with {len(vocabulary)} unique words.")
    logger.debug(f"Sample of vocabulary: {list(vocabulary.items())[:10]}")
    return vocabulary

class NaiveBayesClassifier:
    def __init__(self, alpha: float = 1.0):
        """
        Initializes the Naive Bayes Classifier.
        """
        self.alpha = alpha
        self.class_priors: dict[str, float] = {}
        self.word_likelihoods: dict[str, dict[str, float]] = {}
        self.vocabulary: dict[str, int] = {}
        self.classes: set[str] = set()
        self.vocab_size: int = 0
        logger.info(f"NaiveBayesClassifier initialized with alpha={self.alpha}.")

    def train(self, training_data: list[tuple[list[str], str]], vocabulary: dict):
        """
        Trains the Naive Bayes classifier.
        """
        self.vocabulary = vocabulary
        self.vocab_size = len(self.vocabulary)
        self.classes = set(intent_label for _, intent_label in training_data)

        class_doc_counts = Counter()
        word_counts_per_class = defaultdict(Counter)
        total_words_per_class = Counter()

        logger.info(f"Starting training with {len(training_data)} samples.")
        logger.info(f"Number of classes (intents): {len(self.classes)}")
        logger.info(f"Vocabulary size: {self.vocab_size}")

        for tokens_list, intent_label in training_data:
            class_doc_counts[intent_label] += 1
            for token in tokens_list:
                if token in self.vocabulary:
                    word_counts_per_class[intent_label][token] += 1
                    total_words_per_class[intent_label] += 1

        total_docs = len(training_data)

        for intent_label in self.classes:
            if total_docs > 0 and class_doc_counts[intent_label] > 0:
                self.class_priors[intent_label] = math.log(class_doc_counts[intent_label]) - math.log(total_docs)
            else:
                self.class_priors[intent_label] = -float('inf')
                logger.warning(f"Could not calculate prior for {intent_label} due to zero counts.")
            logger.debug(f"Prior for {intent_label}: {self.class_priors[intent_label]:.4f}")

        self.word_likelihoods = defaultdict(lambda: defaultdict(float))

        for intent_label in self.classes:
            denominator = total_words_per_class[intent_label] + self.alpha * self.vocab_size
            if denominator == 0:
                logger.warning(f"Denominator is zero for intent {intent_label} during likelihood calculation.")

            for word in self.vocabulary:
                count_word_in_class = word_counts_per_class[intent_label][word]
                numerator = count_word_in_class + self.alpha

                if numerator > 0 and denominator > 0:
                    self.word_likelihoods[intent_label][word] = math.log(numerator) - math.log(denominator)
                else:
                    log_prob_for_unseen_word = math.log(self.alpha) - math.log(denominator if denominator > 0 else 1.0)
                    self.word_likelihoods[intent_label][word] = log_prob_for_unseen_word
                    logger.debug(f"Word '{word}' not in class '{intent_label}', using smoothed log_prob: {log_prob_for_unseen_word:.4f}")

        logger.info("Training completed.")

    def predict(self, tokens: list[str]) -> str | None:
        """
        Predicts the intent label for a given list of tokens.
        """
        if not self.classes:
            logger.error("Classifier has not been trained yet. Cannot predict.")
            return None

        class_scores = {}
        for intent_label in self.classes:
            score = self.class_priors.get(intent_label, -float('inf'))
            for token in tokens:
                if token in self.vocabulary:
                    log_likelihood = self.word_likelihoods[intent_label].get(token)
                    if log_likelihood is not None:
                        score += log_likelihood
                    else:
                        logger.warning(f"Word '{token}' in vocabulary but has no log_likelihood for intent '{intent_label}'.")
            class_scores[intent_label] = score

        if not class_scores:
            logger.warning("No scores calculated for any class. Cannot predict.")
            return None

        best_intent = max(class_scores, key=class_scores.get)
        logger.info(f"Predicted Intent: {best_intent} with score: {class_scores[best_intent]:.4f}")
        return best_intent

    def save_model(self, file_path: str):
        """
        Saves the trained model parameters to a JSON file.
        """
        model_data = {
            "alpha": self.alpha,
            "class_priors": self.class_priors,
            "word_likelihoods": self.word_likelihoods,
            "vocabulary": self.vocabulary,
            "classes": list(self.classes),
            "vocab_size": self.vocab_size
        }
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(model_data, f, ensure_ascii=False, indent=4)
            logger.info(f"Model saved successfully to {file_path}")
        except Exception as e:
            logger.error(f"Error saving model to {file_path}: {e}", exc_info=True)

    @classmethod
    def load_model(cls, file_path: str):
        """
        Loads a trained model from a JSON file.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                model_data = json.load(f)

            classifier = cls(alpha=model_data.get("alpha", 1.0))
            classifier.class_priors = model_data["class_priors"]
            loaded_likelihoods = model_data["word_likelihoods"]
            classifier.word_likelihoods = defaultdict(lambda: defaultdict(float))
            for intent, word_probs in loaded_likelihoods.items():
                for word, prob in word_probs.items():
                    classifier.word_likelihoods[intent][word] = prob

            classifier.vocabulary = model_data["vocabulary"]
            classifier.classes = set(model_data["classes"])
            classifier.vocab_size = model_data["vocab_size"]

            logger.info(f"Model loaded successfully from {file_path}")
            return classifier
        except FileNotFoundError:
            logger.error(f"Model file not found at {file_path}.")
            return None
        except Exception as e:
            logger.error(f"Error loading model from {file_path}: {e}", exc_info=True)
            return None


# --- Main execution block for training ---
if __name__ == '__main__':
    # Setup basic logging to see output in the console
    logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
    logger.info("--- Starting Custom Naive Bayes NLU Training ---")

    custom_model_dir = "custom_jarvis_models"
    MODEL_FILE_ES = os.path.join(custom_model_dir, "naive_bayes_es.json")
    MODEL_FILE_EN = os.path.join(custom_model_dir, "naive_bayes_en.json")
    
    # --- 1. Preprocess Training Data ---
    # The training data is now imported from the training.intent_training_data module
    logger.info("\n--- Preprocessing all training data for vocabulary ---")
    all_tokens_for_vocab_build = []
    processed_training_data_es = []
    processed_training_data_en = []

    for text, intent in TRAINING_SAMPLES_ES:
        tokens = tokenize(text, "es")
        processed_training_data_es.append((tokens, intent))
        all_tokens_for_vocab_build.append(tokens)

    for text, intent in TRAINING_SAMPLES_EN:
        tokens = tokenize(text, "en")
        processed_training_data_en.append((tokens, intent))
        all_tokens_for_vocab_build.append(tokens)

    # --- 2. Build a single, shared vocabulary ---
    vocabulary = build_vocabulary(all_tokens_for_vocab_build)

    # --- 3. Train and Save Classifiers ---
    logger.info("\n--- Training and Saving Spanish Naive Bayes Classifier ---")
    classifier_es = NaiveBayesClassifier(alpha=1.0)
    classifier_es.train(processed_training_data_es, vocabulary)
    classifier_es.save_model(MODEL_FILE_ES)

    logger.info("\n--- Training and Saving English Naive Bayes Classifier ---")
    classifier_en = NaiveBayesClassifier(alpha=1.0)
    classifier_en.train(processed_training_data_en, vocabulary)
    classifier_en.save_model(MODEL_FILE_EN)

    # --- 4. Load classifiers and run a quick test ---
    logger.info("\n--- Loading and testing newly trained models ---")
    loaded_classifier_es = NaiveBayesClassifier.load_model(MODEL_FILE_ES)
    if loaded_classifier_es:
        test_text_es = "dame las noticias de hoy"
        tokens_es = tokenize(test_text_es, "es")
        prediction_es = loaded_classifier_es.predict(tokens_es)
        print(f"ES Test: '{test_text_es}' -> Predicted: {prediction_es}")

    loaded_classifier_en = NaiveBayesClassifier.load_model(MODEL_FILE_EN)
    if loaded_classifier_en:
        test_text_en = "tell me the news for today"
        tokens_en = tokenize(test_text_en, "en")
        prediction_en = loaded_classifier_en.predict(tokens_en)
        print(f"EN Test: '{test_text_en}' -> Predicted: {prediction_en}")
        
    logger.info("\n--- Custom NLU Training Completed Successfully ---")