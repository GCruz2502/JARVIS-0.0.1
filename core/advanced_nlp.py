# core/advanced_nlp.py
import logging
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

class AdvancedNLPProcessor:
    def __init__(self):
        """
        Initializes the Advanced NLP Processor, loading necessary
        Hugging Face Transformers models for English and Spanish.
        """
        self.sentiment_analyzer_en = None
        self.qa_pipeline_en = None
        self.sentiment_analyzer_en = None
        self.qa_pipeline_en = None
        self.sentiment_analyzer_es = None
        self.qa_pipeline_es = None
        self.zero_shot_classifier = None
        self.ner_pipeline = None # Added for HF NER
        
        # Initialize English Sentiment Analysis Pipeline
        try:
            logger.info("Initializing English Hugging Face sentiment analysis pipeline...")
            self.sentiment_analyzer_en = pipeline(
                "sentiment-analysis", 
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            logger.info("English Hugging Face sentiment analysis pipeline initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize English Hugging Face sentiment analysis pipeline: {e}", exc_info=True)
            
        # Initialize English Question Answering Pipeline
        try:
            logger.info("Initializing English Hugging Face question answering pipeline...")
            self.qa_pipeline_en = pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad"
            )
            logger.info("English Hugging Face question answering pipeline initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize English Hugging Face question answering pipeline: {e}", exc_info=True)

        # Initialize Spanish Sentiment Analysis Pipeline
        try:
            logger.info("Initializing Spanish Hugging Face sentiment analysis pipeline...")
            # Model: pysentimiento/robertuito-sentiment-analysis
            self.sentiment_analyzer_es = pipeline(
                "sentiment-analysis",
                model="pysentimiento/robertuito-sentiment-analysis"
            )
            logger.info("Spanish Hugging Face sentiment analysis pipeline initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Spanish Hugging Face sentiment analysis pipeline: {e}", exc_info=True)

        # Initialize Spanish Question Answering Pipeline
        try:
            logger.info("Initializing Spanish Hugging Face question answering pipeline...")
            # Model: mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es
            self.qa_pipeline_es = pipeline(
                "question-answering",
                model="mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"
            )
            logger.info("Spanish Hugging Face question answering pipeline initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Spanish Hugging Face question answering pipeline: {e}", exc_info=True)

        # Initialize Zero-Shot Classification Pipeline
        try:
            logger.info("Initializing Hugging Face zero-shot classification pipeline...")
            # Model: facebook/bart-large-mnli (Good multilingual zero-shot model)
            # Using cross-encoder/nli-roberta-base might be faster if only EN/ES needed, but BART is robust.
            self.zero_shot_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
                # Consider adding device=0 if GPU is available and configured in torch
            )
            logger.info("Hugging Face zero-shot classification pipeline initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Hugging Face zero-shot classification pipeline: {e}", exc_info=True)

        # Initialize NER Pipeline (Multilingual)
        try:
            logger.info("Initializing Hugging Face NER pipeline...")
            # Model: dslim/bert-base-NER (multilingual, covers PER, ORG, LOC, MISC)
            # Other options exist, e.g., Spanish-specific ones if needed.
            self.ner_pipeline = pipeline(
                "ner",
                model="dslim/bert-base-NER",
                grouped_entities=True # Group subword tokens for entities like "Bohemian Rhapsody"
            )
            logger.info("Hugging Face NER pipeline initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Hugging Face NER pipeline: {e}", exc_info=True)


    def analyze_sentiment(self, text: str, lang: str = "en") -> dict:
        """
        Analyzes the sentiment of a given text.

        Args:
            text (str): The text to analyze.

        Returns:
            dict: A dictionary containing the sentiment label (e.g., 'POSITIVE', 'NEGATIVE')
                  and score, or an error message.
        """
        analyzer = None
        if lang == "es" and self.sentiment_analyzer_es:
            analyzer = self.sentiment_analyzer_es
        elif lang == "en" and self.sentiment_analyzer_en:
            analyzer = self.sentiment_analyzer_en
        else:
            logger.warning(f"Sentiment analyzer for language '{lang}' not available or not loaded.")
            return {"error": f"Sentiment analyzer for language '{lang}' not available."}
        
        try:
            logger.debug(f"Analyzing sentiment for text (lang={lang}): '{text}'")
            result = analyzer(text)
            # The pipeline returns a list of dictionaries, e.g., [{'label': 'POSITIVE', 'score': 0.999}]
            if result and isinstance(result, list):
                return result[0] 
            return {"error": "Failed to get sentiment from pipeline result."}
        except Exception as e:
            logger.error(f"Error during sentiment analysis for text (lang={lang}) '{text}': {e}", exc_info=True)
            return {"error": f"Error during sentiment analysis: {str(e)}"}

    def answer_question(self, question: str, context_text: str, lang: str = "en") -> dict:
        """
        Answers a question based on a given context text.

        Args:
            question (str): The question to answer.
            context_text (str): The text containing the context for answering the question.
            lang (str): Language of the text ('en' or 'es'). Defaults to 'en'.

        Returns:
            dict: A dictionary containing the answer, score, start, and end positions,
                  or an error message.
        """
        q_pipeline = None
        if lang == "es" and self.qa_pipeline_es:
            q_pipeline = self.qa_pipeline_es
        elif lang == "en" and self.qa_pipeline_en:
            q_pipeline = self.qa_pipeline_en
        else:
            logger.warning(f"Question answering pipeline for language '{lang}' not available or not loaded.")
            return {"error": f"Question answering pipeline for language '{lang}' not available."}

        try:
            logger.debug(f"Answering question (lang={lang}): '{question}' with context: '{context_text[:100]}...'")
            result = q_pipeline(question=question, context=context_text)
            return result
        except Exception as e:
            logger.error(f"Error during question answering for question (lang={lang}) '{question}': {e}", exc_info=True)
            return {"error": f"Error during question answering: {str(e)}"}

    def classify_intent(self, text: str, candidate_labels: list, multi_label: bool = False) -> dict:
        """
        Classifies the intent of the text using zero-shot classification.

        Args:
            text (str): The text to classify.
            candidate_labels (list): A list of possible intent labels (e.g., plugin names or actions).
            multi_label (bool): Whether multiple labels can be true. Defaults to False.

        Returns:
            dict: A dictionary containing the sequence, labels, and scores, or an error message.
                  Example: {'sequence': '...', 'labels': ['weather', 'music'], 'scores': [0.9, 0.1]}
        """
        if not self.zero_shot_classifier:
            logger.warning("Zero-shot classifier not available.")
            return {"error": "Zero-shot classifier not available."}
        
        if not candidate_labels:
             logger.warning("No candidate labels provided for zero-shot classification.")
             return {"error": "No candidate labels provided."}

        try:
            logger.debug(f"Classifying intent for text: '{text}' with labels: {candidate_labels}")
            result = self.zero_shot_classifier(text, candidate_labels, multi_label=multi_label)
            return result
        except Exception as e:
            logger.error(f"Error during zero-shot classification for text '{text}': {e}", exc_info=True)
            return {"error": f"Error during zero-shot classification: {str(e)}"}

    def extract_entities_hf(self, text: str) -> list:
        """
        Extracts named entities using the Hugging Face NER pipeline.

        Args:
            text (str): The text to extract entities from.

        Returns:
            list: A list of dictionaries, where each dictionary represents an entity.
                  Example: [{'entity_group': 'PER', 'score': 0.99, 'word': 'Rubén Blades', 'start': 13, 'end': 25}]
                  Returns an empty list if the pipeline is not available or an error occurs.
        """
        if not self.ner_pipeline:
            logger.warning("HF NER pipeline not available.")
            return []

        try:
            logger.debug(f"Extracting entities using HF NER for text: '{text}'")
            entities = self.ner_pipeline(text)
            # The pipeline might return entities split by subwords if grouped_entities=False
            # With grouped_entities=True, it should return combined entities.
            # Format can vary slightly based on model/pipeline version.
            return entities
        except Exception as e:
            logger.error(f"Error during HF NER entity extraction for text '{text}': {e}", exc_info=True)
            return []


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    processor = AdvancedNLPProcessor()

    # Test English Sentiment Analysis
    if processor.sentiment_analyzer_en:
        logger.info("\n--- Testing English Sentiment Analysis ---")
        texts_to_test_en_sentiment = [
            "I love this new feature, it's amazing!",
            "This is terrible, I'm very disappointed.",
        ]
        for t in texts_to_test_en_sentiment:
            sentiment = processor.analyze_sentiment(t, lang="en")
            logger.info(f"Text (en): '{t}' -> Sentiment: {sentiment}")
    else:
        logger.warning("English Sentiment analyzer failed to initialize. Skipping English sentiment tests.")

    # Test Spanish Sentiment Analysis
    if processor.sentiment_analyzer_es:
        logger.info("\n--- Testing Spanish Sentiment Analysis ---")
        texts_to_test_es_sentiment = [
            "¡Me encanta esta nueva característica, es increíble!",
            "Esto es terrible, estoy muy decepcionado.",
            "El clima está bastante neutral hoy."
        ]
        for t in texts_to_test_es_sentiment:
            sentiment = processor.analyze_sentiment(t, lang="es")
            logger.info(f"Text (es): '{t}' -> Sentiment: {sentiment}")
    else:
        logger.warning("Spanish Sentiment analyzer failed to initialize. Skipping Spanish sentiment tests.")

    # Test English Question Answering
    if processor.qa_pipeline_en:
        logger.info("\n--- Testing English Question Answering ---")
        qa_context_en = """
        The JARVIS project is a voice-activated virtual assistant. 
        It uses Python as its core language. SpaCy is used for basic NLP tasks.
        """
        questions_en = [
            "What is JARVIS?",
            "Which programming language does JARVIS use?",
        ]
        for q in questions_en:
            answer = processor.answer_question(question=q, context_text=qa_context_en, lang="en")
            logger.info(f"Context (en): (snippet)\nQuestion: {q}\nAnswer: {answer}\n")
    else:
        logger.warning("English Question answering pipeline failed to initialize. Skipping English QA tests.")

    # Test Spanish Question Answering
    if processor.qa_pipeline_es:
        logger.info("\n--- Testing Spanish Question Answering ---")
        qa_context_es = """
        El proyecto JARVIS es un asistente virtual activado por voz. 
        Utiliza Python como su lenguaje principal. Se usa SpaCy para tareas básicas de PLN,
        y se están integrando Transformers de Hugging Face para capacidades avanzadas.
        """
        questions_es = [
            "¿Qué es JARVIS?",
            "¿Qué lenguaje de programación usa JARVIS?",
            "¿Para qué se usa SpaCy en JARVIS?"
        ]
        for q in questions_es:
            answer = processor.answer_question(question=q, context_text=qa_context_es, lang="es")
            logger.info(f"Context (es): (snippet)\nQuestion: {q}\nAnswer: {answer}\n")
    else:
        logger.warning("Spanish Question answering pipeline failed to initialize. Skipping Spanish QA tests.")

    # Test Zero-Shot Classification
    if processor.zero_shot_classifier:
        logger.info("\n--- Testing Zero-Shot Classification ---")
        intent_labels = ["get weather", "play music", "tell joke", "set reminder", "ask question"]
        texts_to_classify = [
            "what's the temperature in London tomorrow", # EN
            "pon la canción 'Bohemian Rhapsody'", # ES
            "tell me something funny", # EN
            "recuérdame llamar a mamá a las 5 pm", # ES
            "who directed the movie Inception?" # EN
        ]
        for t in texts_to_classify:
            classification = processor.classify_intent(t, intent_labels)
            logger.info(f"Text: '{t}' -> Classification: {classification}")
    else:
        logger.warning("Zero-shot classifier failed to initialize. Skipping classification tests.")
        
    # Test HF NER
    if processor.ner_pipeline:
        logger.info("\n--- Testing HF NER ---")
        texts_for_ner = [
            "Pon la canción Bohemian Rhapsody de Queen",
            "Toca algo de Rubén Blades en Ciudad de Panamá",
            "Apple está buscando comprar una startup del Reino Unido por mil millones de dólares."
        ]
        for t in texts_for_ner:
            entities = processor.extract_entities_hf(t)
            logger.info(f"Text: '{t}' -> HF Entities: {entities}")
    else:
        logger.warning("HF NER pipeline failed to initialize. Skipping NER tests.")
