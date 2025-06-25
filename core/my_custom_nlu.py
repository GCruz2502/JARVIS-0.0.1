import re
import logging
from collections import Counter, defaultdict
import math
import json
import os

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

    # Remove punctuation - this regex keeps words (alphanumeric) and apostrophes within words (for contractions)
    # It will split on spaces and other non-word, non-apostrophe characters.
    # This is a starting point and can be refined.
    if lang == "en":
        # For English, we might want to be careful with apostrophes for contractions like "don't", "it's"
        # This regex tries to find sequences of word characters OR apostrophes within words.
        tokens = re.findall(r"\b[a-z0-9']+\b", text)
    elif lang == "es":
        # For Spanish, apostrophes are less common within words for contractions.
        # Remove initial ¿ and ¡ if present
        text = text.strip("¿¡")
        tokens = re.findall(r"\b[a-z0-9ñáéíóúü]+\b", text) # Added Spanish characters
    else:
        logger.warning(f"Unsupported language for tokenization: {lang}. Defaulting to basic word split.")
        # A very generic fallback, might not be ideal.
        tokens = re.findall(r"\b\w+\b", text) 
        # \w matches alphanumeric characters (letters, numbers, and underscore)

    logger.debug(f"Original text: '{text}' -> Tokens: {tokens} (lang: {lang})")
    return tokens

# +++ NEW FUNCTIONS FOR FEATURE EXTRACTION +++

def build_vocabulary(all_training_tokens: list[list[str]]) -> dict:
    """
    Builds a vocabulary from a list of tokenized training sentences.
    Args:
        all_training_tokens: A list where each item is a list of tokens for one training sentence.
                            Example: [ ['hola', 'jarvis'], ['qué', 'tiempo', 'hace'] ]
    Returns:
        A dictionary mapping each unique word to a unique integer index.
        Example: {'hola': 0, 'jarvis': 1, 'qué': 2, 'tiempo': 3, 'hace': 4}
    """
    all_words = []
    for sentence_tokens in all_training_tokens:
        all_words.extend(sentence_tokens)
    
    unique_words = sorted(list(set(all_words))) # Get unique words and sort for consistent indexing
    vocabulary = {word: i for i, word in enumerate(unique_words)}
    
    logger.info(f"Built vocabulary with {len(vocabulary)} unique words.")
    logger.debug(f"Sample of vocabulary: {list(vocabulary.items())[:10]}") # Log a sample
    return vocabulary

def extract_bow_features(tokens: list[str], vocabulary: dict) -> list[int]:
    """
    Extracts Bag-of-Words features for a list of tokens.
    Args:
        tokens: A list of tokens for a single sentence.
        vocabulary: The vocabulary dictionary (word -> index).
    Returns:
        A list of integers representing the BoW vector (word counts).
    """
    bow_vector = [0] * len(vocabulary) # Initialize vector with zeros
    token_counts = Counter(tokens) # Count occurrences of each token in the input sentence
    
    for token, count in token_counts.items():
        if token in vocabulary: # Only consider words present in our vocabulary
            vocab_index = vocabulary[token]
            bow_vector[vocab_index] = count
            
    logger.debug(f"Tokens: {tokens} -> BoW Vector (first 20 elements if long): {bow_vector[:20]}")
    return bow_vector

# +++ NEW NaiveBayesClassifier CLASS +++
class NaiveBayesClassifier:
    def __init__(self, alpha: float = 1.0):
        """
        Initializes the Naive Bayes Classifier.
        Args:
            alpha (float): Smoothing parameter for Laplace/Lidstone smoothing.
                          Prevents zero probabilities for words not seen during training for a class.
        """
        self.alpha = alpha  # Smoothing parameter
        self.class_priors: dict[str, float] = {}  # P(Intent)
        self.word_likelihoods: dict[str, dict[str, float]] = {} # P(Word | Intent)
        self.vocabulary: dict[str, int] = {} # Word to index mapping
        self.classes: set[str] = set() # Set of all unique intent labels seen during training
        self.vocab_size: int = 0 # Will be len(self.vocabulary)

        logger.info(f"NaiveBayesClassifier initialized with alpha={self.alpha}.")

    def train(self, training_data: list[tuple[list[str], str]], vocabulary: dict):
        """
        Trains the Naive Bayes classifier.
        Args:
            training_data: A list of tuples, where each tuple is (tokens_list, intent_label).
                          Example: [ (['hola', 'jarvis'], 'INTENT_GREET'), 
                                      (['qué', 'tiempo', 'hace'], 'INTENT_GET_WEATHER') ]
            vocabulary: The vocabulary dictionary (word -> index) built from all training data.
        """
        self.vocabulary = vocabulary
        self.vocab_size = len(self.vocabulary)
        self.classes = set(intent_label for _, intent_label in training_data)
        
        # Initialize counts
        class_doc_counts = Counter() # How many documents/sentences per class
        # word_counts_per_class: For each class, a Counter for words in that class
        # e.g., {'INTENT_GREET': Counter({'hola': 1, 'jarvis': 1}), ...}
        word_counts_per_class = defaultdict(Counter)
        # total_words_per_class: For each class, total number of words (tokens)
        # e.g., {'INTENT_GREET': 2, 'INTENT_GET_WEATHER': 3, ...}
        total_words_per_class = Counter()

        logger.info(f"Starting training with {len(training_data)} samples.")
        logger.info(f"Number of classes (intents): {len(self.classes)}")
        logger.info(f"Vocabulary size: {self.vocab_size}")

        # --- Pass 1: Calculate counts ---
        for tokens_list, intent_label in training_data:
            class_doc_counts[intent_label] += 1
            # For Bag-of-Words, we count each unique word in the sentence once per sentence for this calculation step,
            # or we count all words if not using set(tokens_list).
            # Standard Multinomial Naive Bayes uses total word occurrences.
            for token in tokens_list:
                if token in self.vocabulary: # Only consider words in our vocabulary
                    word_counts_per_class[intent_label][token] += 1
                    total_words_per_class[intent_label] += 1
        
        total_docs = len(training_data)

        # --- Pass 2: Calculate probabilities ---
        # Calculate class prior probabilities: P(Intent)
        for intent_label in self.classes:
            if total_docs > 0 and class_doc_counts[intent_label] > 0:
                self.class_priors[intent_label] = math.log(class_doc_counts[intent_label]) - math.log(total_docs)
            else:
                # Handle case where a class might have 0 documents or total_docs is 0 (should not happen with valid training data)
                # Assign a very small log probability (large negative number)
                self.class_priors[intent_label] = -float('inf') # Or some other indicator of impossibility
                logger.warning(f"Could not calculate prior for {intent_label} due to zero counts. Assigning -inf.")
            logger.debug(f"Prior for {intent_label}: {self.class_priors[intent_label]:.4f}")

        # Calculate word likelihoods: P(Word | Intent) with Laplace smoothing
        # P(word_i | class_j) = (count(word_i, class_j) + alpha) / (total_words_in_class_j + alpha * vocab_size)
        self.word_likelihoods = defaultdict(lambda: defaultdict(float)) # Initialize properly

        for intent_label in self.classes:
            denominator = total_words_per_class[intent_label] + self.alpha * self.vocab_size
            if denominator == 0: # Should not happen with alpha > 0 and vocab_size > 0
                logger.warning(f"Denominator is zero for intent {intent_label} during likelihood calculation. This is unexpected.")
                # Handle this case, e.g., by assigning a very small probability or skipping
                # For now, let's just ensure all words get a smoothed default probability
            
            for word in self.vocabulary:
                count_word_in_class = word_counts_per_class[intent_label][word]
                numerator = count_word_in_class + self.alpha
                
                # Store log probabilities for numerical stability and to avoid underflow
                # log P(word_i | class_j) = log(numerator) - log(denominator)
                if numerator > 0 and denominator > 0: # Ensure args to log are positive
                    self.word_likelihoods[intent_label][word] = math.log(numerator) - math.log(denominator)
                else:
                    # This case should ideally be avoided by smoothing and having a vocab.
                    # If it occurs, it means a word from vocab was not seen with this intent,
                    # and smoothing didn't make numerator positive (only if alpha=0 and count=0)
                    # or denominator was zero (very problematic).
                    # With alpha > 0, numerator is always > 0.
                    # Denominator only zero if total_words_in_class_j is 0 AND alpha * vocab_size is 0.
                    # This implies no words for that class AND (alpha=0 OR vocab_size=0).
                    # We'll rely on alpha > 0 and vocab_size > 0.
                    # Fallback to a very small log probability if something unexpected happens.
                    # This is P(word | class) = alpha / (total_words_in_class + alpha * vocab_size)
                    log_prob_for_unseen_word = math.log(self.alpha) - math.log(denominator if denominator > 0 else 1.0) # Avoid log(0)
                    self.word_likelihoods[intent_label][word] = log_prob_for_unseen_word
                    logger.debug(f"Word '{word}' not in class '{intent_label}', using smoothed log_prob: {log_prob_for_unseen_word:.4f}")

            # For debugging, log a few likelihoods for each class
            sample_word_likelihoods = {w: self.word_likelihoods[intent_label][w] for w in list(self.vocabulary.keys())[:5]}
            logger.debug(f"Sample word log-likelihoods for {intent_label}: {sample_word_likelihoods}")

        logger.info("Training completed.")

    def predict(self, tokens: list[str]) -> str | None:
        """
        Predicts the intent label for a given list of tokens.
        Args:
            tokens: A list of tokens for the sentence to classify.
        Returns:
            The predicted intent label (str) or None if no classes are available.
        """

        if not self.classes:
            logger.error("Classifier has not been trained yet (no classes found). Cannot predict.")
            return None

        # Calculate score for each class (intent)
        # Score(Intent) = log P(Intent) + sum(log P(Word_i | Intent)) for each word in the input tokens
        # We use logs to sum instead of multiplying probabilities, which helps with numerical stability.
        
        class_scores = {}
        for intent_label in self.classes:
            # Start with the log prior probability of the class
            score = self.class_priors.get(intent_label, -float('inf')) # Default to very low score if prior is missing

            # Add the log likelihood of each word in the input tokens given this class
            for token in tokens:
                if token in self.vocabulary: # Only consider words known in our vocabulary
                    # Get P(token | intent_label)
                    # If a word was not seen with this intent during training,
                    # its likelihood will be the smoothed "unseen word" probability.
                    log_likelihood = self.word_likelihoods[intent_label].get(token)
                    
                    if log_likelihood is not None:
                        score += log_likelihood
                    else:
                        # This case means the word is in the global vocabulary,
                        # but somehow a log likelihood wasn't computed for it for this class during training.
                        # This might happen if the denominator for smoothing was 0 for that class initially,
                        # and the fallback log_prob_for_unseen_word was used.
                        # Or if self.alpha was 0 and the word had 0 count in class.
                        # We can use a default smoothed probability for words in vocab but missing from likelihoods dict for a class
                        # This is essentially the P(word | class) = alpha / (total_words_in_class + alpha * vocab_size)
                        # which we already calculated during training for unseen words.
                        # A simpler approach here is to just not add to the score if it's truly missing,
                        # or add a very small fixed penalty.
                        # Let's assume our training's word_likelihoods for each class covers all vocab words.
                        # If not, we might need a global "unknown word for this class" likelihood.
                        # For now, if a word from vocab doesn't have a likelihood for a class (shouldn't happen with proper smoothing in train)
                        # we can use the P(unseen_word | class) value which is alpha / (N_class + alpha*V)
                        # This was already stored in self.word_likelihoods[intent_label][word] during training if it was 0 count.
                        # So, if log_likelihood is None, it's truly an issue beyond normal unseen words.
                        logger.warning(f"Word '{token}' is in vocabulary but has no stored log_likelihood for intent '{intent_label}'. Skipping.")
                        pass # Or add a default penalty: score += math.log(self.alpha) - math.log(total_words_per_class.get(intent_label,0) + self.alpha * self.vocab_size)

            class_scores[intent_label] = score
            logger.debug(f"Score for intent '{intent_label}': {score:.4f}")

        if not class_scores:
            logger.warning("No scores calculated for any class. Cannot predict.")
            return None
            
        # Return the class (intent label) with the highest score
        best_intent = max(class_scores, key=class_scores.get)
        logger.info(f"Predicted Intent: {best_intent} with score: {class_scores[best_intent]:.4f}")
        return best_intent
    
    def save_model(self, file_path: str):
        """
        Saves the trained model parameters to a JSON file.
        Args:
            file_path (str): The path to save the model file.
        """
        model_data = {
            "alpha": self.alpha,
            "class_priors": self.class_priors,
            "word_likelihoods": self.word_likelihoods,
            "vocabulary": self.vocabulary,
            "classes": list(self.classes), # Convert set to list for JSON serialization
            "vocab_size": self.vocab_size
        }
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(model_data, f, ensure_ascii=False, indent=4)
            logger.info(f"Model saved successfully to {file_path}")
        except Exception as e:
            logger.error(f"Error saving model to {file_path}: {e}", exc_info=True)

    @classmethod # This method creates an instance from a file, so it's a class method
    
    def load_model(cls, file_path: str):
        """
        Loads a trained model from a JSON file.
        Args:
            file_path (str): The path to the model file.
        Returns:
            An instance of NaiveBayesClassifier with loaded parameters, or None if loading fails.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                model_data = json.load(f)
            
            # Create a new classifier instance
            # The alpha value used during training is saved, so we use it.
            classifier = cls(alpha=model_data.get("alpha", 1.0)) # Default alpha if not found
            
            classifier.class_priors = model_data["class_priors"]
            # defaultdict needs to be reconstructed carefully if it was saved as a plain dict
            # For word_likelihoods, json saves dicts, not defaultdicts. We need to handle this.
            loaded_likelihoods = model_data["word_likelihoods"]
            classifier.word_likelihoods = defaultdict(lambda: defaultdict(float))
            for intent, word_probs in loaded_likelihoods.items():
                for word, prob in word_probs.items():
                    classifier.word_likelihoods[intent][word] = prob
            
            classifier.vocabulary = model_data["vocabulary"]
            classifier.classes = set(model_data["classes"]) # Convert list back to set
            classifier.vocab_size = model_data["vocab_size"]
            
            logger.info(f"Model loaded successfully from {file_path}")
            logger.info(f"Loaded model details: {len(classifier.classes)} classes, {classifier.vocab_size} vocab size.")
            return classifier
        except FileNotFoundError:
            logger.error(f"Model file not found at {file_path}. Cannot load model.")
            return None
        except Exception as e:
            logger.error(f"Error loading model from {file_path}: {e}", exc_info=True)
            return None

    # We will add train(), predict(), save_model(), load_model() methods here later.


# --- Example Usage (for testing this file directly) ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO) 
    logger.info("--- Starting Custom NLU Tests ---")

    # Define model file paths (consider making these configurable later)
    # Create a directory for your custom models if it doesn't exist
    custom_model_dir = "custom_jarvis_models"
    if not os.path.exists(custom_model_dir):
        os.makedirs(custom_model_dir)
        logger.info(f"Created directory: {custom_model_dir}")

    MODEL_FILE_ES = os.path.join(custom_model_dir, "naive_bayes_es.json")
    MODEL_FILE_EN = os.path.join(custom_model_dir, "naive_bayes_en.json")

    # --- 1. Define Sample Training Data ---
    training_samples_es = [
        # INTENT_GET_WEATHER (keep existing, add more if desired)
        ("qué tiempo hace hoy", "INTENT_GET_WEATHER"),
        ("dime el clima en panamá", "INTENT_GET_WEATHER"),
        ("va a llover mañana", "INTENT_GET_WEATHER"),
        ("cómo está el tiempo en bogotá", "INTENT_GET_WEATHER"),
        ("temperatura actual por favor", "INTENT_GET_WEATHER"),

        # INTENT_PLAY_MUSIC / _SONG / _ARTIST (keep existing, add more)
        ("pon música de rock", "INTENT_PLAY_MUSIC"),
        ("reproduce la canción Imagine de John Lennon", "INTENT_PLAY_SONG"),
        ("quiero escuchar a Queen", "INTENT_PLAY_ARTIST"),
        ("pon mi playlist de ejercicio", "INTENT_PLAY_PLAYLIST"),

        # INTENT_GREET (keep existing, add more)
        ("hola jarvis", "INTENT_GREET"),
        ("buenos días", "INTENT_GREET"),
        ("qué tal jarvis", "INTENT_GREET"),

        # INTENT_FAREWELL (keep existing, add more)
        ("adiós", "INTENT_FAREWELL"),
        ("hasta luego", "INTENT_FAREWELL"),
        ("chao", "INTENT_FAREWELL"),
        ("nos vemos", "INTENT_FAREWELL"),

        # +++ NEW EXAMPLES FOR PREVIOUSLY MISCLASSIFIED INTENTS +++
        ("qué hora es", "INTENT_GET_TIME"),
        ("dime la hora por favor", "INTENT_GET_TIME"),
        ("me dices la hora", "INTENT_GET_TIME"),
        ("la hora actual", "INTENT_GET_TIME"),
        ("hora", "INTENT_GET_TIME"), # Short, might be ambiguous  

        ("qué fecha es hoy", "INTENT_GET_DATE"),
        ("dime la fecha", "INTENT_GET_DATE"),
        ("fecha de hoy", "INTENT_GET_DATE"),
        ("en qué fecha estamos", "INTENT_GET_DATE"),
        ("fecha", "INTENT_GET_DATE"), # Short

        ("dame las noticias", "INTENT_GET_NEWS"),
        ("últimas noticias", "INTENT_GET_NEWS"),
        ("noticias de actualidad", "INTENT_GET_NEWS"),
        ("quiero saber las noticias", "INTENT_GET_NEWS"),
        ("qué está pasando", "INTENT_GET_NEWS"), # More conversational

        # Example for INTENT_SET_REMINDER
        ("recuérdame llamar a mamá mañana", "INTENT_SET_REMINDER"),
        ("pon un recordatorio para la reunión de las 3 pm", "INTENT_SET_REMINDER"),

        # Example for INTENT_HELP
        ("ayuda", "INTENT_HELP"),
        ("qué puedes hacer", "INTENT_HELP"),

        # Example for INTENT_CLEAR_CONTEXT
        ("limpia el contexto", "INTENT_CLEAR_CONTEXT"),
        ("olvida nuestra conversación", "INTENT_CLEAR_CONTEXT"),
    ]
    training_samples_en = [
        # INTENT_GET_WEATHER (keep existing, add more if desired)
        ("what is the weather today", "INTENT_GET_WEATHER"),
        ("tell me the forecast for london", "INTENT_GET_WEATHER"),
        ("is it going to rain tomorrow", "INTENT_GET_WEATHER"),
        ("how's the weather in new york", "INTENT_GET_WEATHER"),
        ("current temperature please", "INTENT_GET_WEATHER"),

        # INTENT_PLAY_MUSIC / _SONG / _ARTIST (keep existing, add more)
        ("play some rock music", "INTENT_PLAY_MUSIC"),
        ("play the song Imagine of Jhon Lennon", "INTENT_PLAY_SONG"),
        ("I want to listen to Queen", "INTENT_PLAY_ARTIST"),
        ("play my workout playlist", "INTENT_PLAY_PLAYLIST"),

        # INTENT_GREET (keep existing, add more)
        ("hello jarvis", "INTENT_GREET"),
        ("good morning", "INTENT_GREET"),
        ("how are you jarvis", "INTENT_GREET"),
        ("hi there", "INTENT_GREET"),           

        # INTENT_FAREWELL (keep existing, add more)
        ("goodbye", "INTENT_FAREWELL"),
        ("see you later", "INTENT_FAREWELL"),
        ("bye bye", "INTENT_FAREWELL"),

        # +++ NEW EXAMPLES FOR PREVIOUSLY MISCLASSIFIED INTENTS +++
        ("what hour is it", "INTENT_GET_TIME"),
        ("tell me the time please", "INTENT_GET_TIME"),
        ("can you tell me the time", "INTENT_GET_TIME"),
        ("the current time", "INTENT_GET_TIME"),
        ("time", "INTENT_GET_TIME"), # Short, might be ambiguous  

        ("what date is today", "INTENT_GET_DATE"),
        ("tell me the date", "INTENT_GET_DATE"),
        ("date of today", "INTENT_GET_DATE"),
        ("what's the date today", "INTENT_GET_DATE"),
        ("date", "INTENT_GET_DATE"), # Short

        ("give me the news", "INTENT_GET_NEWS"),
        ("latest news", "INTENT_GET_NEWS"),
        ("current news", "INTENT_GET_NEWS"),
        ("I want to know the news", "INTENT_GET_NEWS"),
        ("what's happening", "INTENT_GET_NEWS"), # More conversational

        # Example for INTENT_SET_REMINDER
        ("remind me to call mom tomorrow", "INTENT_SET_REMINDER"),
        ("set a reminder for the meeting at 3 pm", "INTENT_SET_REMINDER"),

        # Example for INTENT_HELP
        ("help", "INTENT_HELP"),
        ("what can you do", "INTENT_HELP"),

        # Example for INTENT_CLEAR_CONTEXT
        ("clear the context", "INTENT_CLEAR_CONTEXT"),
        ("forget our conversation", "INTENT_CLEAR_CONTEXT"),
    ]

    # --- 2. Preprocess Training Data ---
    logger.info("\n--- Preprocessing Training Data ---")
    all_tokens_for_vocab_build = []
    processed_training_data_es = []
    processed_training_data_en = []
    for text, intent in training_samples_es:
        tokens = tokenize(text, "es")
        processed_training_data_es.append((tokens, intent))
        all_tokens_for_vocab_build.append(tokens)
    for text, intent in training_samples_en:
        tokens = tokenize(text, "en")
        processed_training_data_en.append((tokens, intent))
        all_tokens_for_vocab_build.append(tokens)
        
    vocabulary = build_vocabulary(all_tokens_for_vocab_build)

    # --- 4. Train and Save Classifiers ---
    logger.info("\n--- Training and Saving Spanish Naive Bayes Classifier ---")
    classifier_es = NaiveBayesClassifier(alpha=1.0)
    classifier_es.train(processed_training_data_es, vocabulary)
    classifier_es.save_model(MODEL_FILE_ES)

    logger.info("\n--- Training and Saving English Naive Bayes Classifier ---")
    classifier_en = NaiveBayesClassifier(alpha=1.0)
    classifier_en.train(processed_training_data_en, vocabulary)
    classifier_en.save_model(MODEL_FILE_EN)

    # --- 5. Load Classifiers and Test Predictions ---
    logger.info("\n--- Loading Spanish Classifier and Testing Predictions ---")
    loaded_classifier_es = NaiveBayesClassifier.load_model(MODEL_FILE_ES)
    if loaded_classifier_es:
        test_sentences_es = [
            "qué clima hace", "escuchar Imagine", "hola", "chao"
        ]
        for sentence in test_sentences_es:
            tokens = tokenize(sentence, "es")
            predicted_intent = loaded_classifier_es.predict(tokens)
            logger.info(f"ES (Loaded) - Sentence: '{sentence}' -> Tokens: {tokens} -> Predicted Intent: {predicted_intent}")
    else:
        logger.error("Failed to load Spanish classifier for testing.")

    logger.info("\n--- Loading English Classifier and Testing Predictions ---")
    loaded_classifier_en = NaiveBayesClassifier.load_model(MODEL_FILE_EN)
    if loaded_classifier_en:
        test_sentences_en = [
            "current weather", "listen to some Queen", "hi there", "bye bye"
        ]
        for sentence in test_sentences_en:
            tokens = tokenize(sentence, "en")
            predicted_intent = loaded_classifier_en.predict(tokens)
            logger.info(f"EN (Loaded) - Sentence: '{sentence}' -> Tokens: {tokens} -> Predicted Intent: {predicted_intent}")
    else:
        logger.error("Failed to load English classifier for testing.")
        
    logger.info("\n--- Custom NLU Tests Completed (with save/load) ---")