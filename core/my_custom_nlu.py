import re
import logging
from collections import Counter

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


# --- Example Usage (for testing this file directly) ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    # --- Test Tokenizer ---
    print("\n--- Tokenizer Tests ---")
    test_text_es1 = "¿Hola, cómo estás, amigo?"
    tokens_es1 = tokenize(test_text_es1, 'es')
    print(f"ES1: {tokens_es1}")

    test_text_es2 = "Pon la canción de 'Bohemian Rhapsody' mañana a las 5pm."
    
    test_text_en1 = "Hello, how are you, my friend?"
    
    test_text_en2 = "Don't forget to play the song 'Bohemian Rhapsody' tomorrow at 5pm."
    tokens_en2 = tokenize(test_text_en2, 'en')
    print(f"EN2: {tokens_en2}")

    print(f"ES1: {tokenize(test_text_es1, 'es')}")
    # Expected rough output: ['hola', 'cómo', 'estás', 'amigo']
    
    print(f"ES2: {tokenize(test_text_es2, 'es')}")
    # Expected rough output: ['pon', 'la', 'canción', 'de', 'bohemian', 'rhapsody', 'mañana', 'a', 'las', '5pm'] (Note: '5pm' becomes one token)

    print(f"EN1: {tokenize(test_text_en1, 'en')}")
    # Expected rough output: ['hello', 'how', 'are', 'you', 'my', 'friend']
    
    print(f"EN2: {tokenize(test_text_en2, 'en')}")
    # Expected rough output: ["don't", 'forget', 'to', 'play', 'the', 'song', 'bohemian', 'rhapsody', 'tomorrow', 'at', '5pm']

    # Test edge cases
    print(f"Empty: {tokenize('', 'es')}")
    print(f"Numbers: {tokenize('Orden 123 para mesa 4.', 'es')}")
    # Expected: ['orden', '123', 'para', 'mesa', '4']


    # --- Test Feature Extraction ---
    print("\n--- Feature Extraction Tests ---")
    # Simulate some tokenized training data to build a vocabulary
    sample_training_sentences_tokens = [
        tokenize("hola jarvis", "es"),
        tokenize("qué tiempo hace hoy", "es"),
        tokenize("pon música de rock", "es"),
        tokenize("adiós jarvis", "es"),
        tokenize("play a song", "en"),
        tokenize("what is the weather today", "en"),
        tokenize("set a reminder", "en")
    ]

    # Build vocabulary from ALL sample training sentences (both languages for a combined vocab, or separate)
    # For simplicity here, let's combine. In practice, you might have lang-specific vocabs or a shared one.
    print("\nBuilding combined vocabulary for example...")
    combined_training_tokens = []
    for sentence_list in sample_training_sentences_tokens:
        combined_training_tokens.append(sentence_list)
        
    # In a real scenario, `combined_training_tokens` would come from processing
    # ALL your `TRAIN_DATA_..._TEXTCAT` that you prepared for spaCy.
    
    vocabulary = build_vocabulary(combined_training_tokens)
    print(f"Vocabulary created with {len(vocabulary)} words.")
    # print(f"Full vocabulary: {vocabulary}") # Can be very long

    # Test feature extraction for a new sentence
    new_sentence_es = "qué música te gusta?"
    new_tokens_es = tokenize(new_sentence_es, "es")
    print(f"\nNew ES sentence: '{new_sentence_es}' -> Tokens: {new_tokens_es}")
    bow_features_es = extract_bow_features(new_tokens_es, vocabulary)
    print(f"BoW Features (ES): {bow_features_es}")
    # To make sense of the BoW vector, you'd need to see the vocabulary:
    # Example: if vocabulary is {'gusta': 10, 'música': 15, 'qué': 20}
    # and new_tokens_es is ['qué', 'música', 'te', 'gusta'] (assuming 'te' is not in vocab here)
    # bow_features_es might look like [0,0,..,1 (at index 10 for gusta),0,..,1 (at index 15 for música),0,..,1 (at index 20 for qué),0...]

    new_sentence_en = "play some rock music"
    new_tokens_en = tokenize(new_sentence_en, "en")
    print(f"\nNew EN sentence: '{new_sentence_en}' -> Tokens: {new_tokens_en}")
    bow_features_en = extract_bow_features(new_tokens_en, vocabulary)
    print(f"BoW Features (EN): {bow_features_en}")