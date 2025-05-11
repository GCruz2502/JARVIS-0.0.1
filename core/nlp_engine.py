# core/nlp_engine.py
import logging
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Assuming 'load_data' is now in 'utils.database_handler'
# This import path needs to be correct based on the final location of load_data.
try:
    from utils.database_handler import load_data
except ImportError:
    logger = logging.getLogger(__name__) # Define logger early for this message
    logger.error("Failed to import 'load_data' from 'utils.database_handler'. "
                 "Scikit-learn model training/loading might fail if it relies on this.")
    # Define a dummy load_data if it's absolutely critical for the file to load
    # though this means the sklearn model part won't work.
    def load_data():
        logger.error("'load_data' is not available due to import error.")
        return [] 

logger = logging.getLogger(__name__)

# --- Content from core/advanced_nlp.py ---
class AdvancedNLPProcessor:
    def __init__(self):
        """
        Initializes the Advanced NLP Processor, loading necessary
        Hugging Face Transformers models for English and Spanish.
        """
        self.sentiment_analyzer_en = None
        self.qa_pipeline_en = None
        # self.sentiment_analyzer_en = None # Duplicate line removed
        # self.qa_pipeline_en = None # Duplicate line removed
        self.sentiment_analyzer_es = None
        self.qa_pipeline_es = None
        self.zero_shot_classifier = None
        self.ner_pipeline = None
        self.text_generator = None # For DialoGPT
        
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
            self.zero_shot_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
            logger.info("Hugging Face zero-shot classification pipeline initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Hugging Face zero-shot classification pipeline: {e}", exc_info=True)

        # Initialize NER Pipeline (Multilingual)
        try:
            logger.info("Initializing Hugging Face NER pipeline...")
            self.ner_pipeline = pipeline(
                "ner",
                model="dslim/bert-base-NER",
                grouped_entities=True
            )
            logger.info("Hugging Face NER pipeline initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Hugging Face NER pipeline: {e}", exc_info=True)

        # Initialize Text Generation Pipeline (DialoGPT)
        try:
            logger.info("Initializing Hugging Face text generation pipeline (Llama-3-8B-Instruct-hf)...")
            # Ensure you are logged into Hugging Face CLI and have accepted Llama 3 terms
            self.text_generator = pipeline("text-generation", model="meta-llama/Llama-3-8B-Instruct-hf", device_map="auto") # device_map="auto" helps with large models
            logger.info("Hugging Face text generation pipeline (Llama-3-8B-Instruct-hf) initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Hugging Face text generation pipeline (Llama-3-8B-Instruct-hf): {e}", exc_info=True)
            # TRANSFORMERS_AVAILABLE in report_generator.py was a global flag.
            # Here, self.text_generator will remain None if it fails.

    def analyze_sentiment(self, text: str, lang: str = "en") -> dict:
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
            if result and isinstance(result, list):
                return result[0] 
            return {"error": "Failed to get sentiment from pipeline result."}
        except Exception as e:
            logger.error(f"Error during sentiment analysis for text (lang={lang}) '{text}': {e}", exc_info=True)
            return {"error": f"Error during sentiment analysis: {str(e)}"}

    def answer_question(self, question: str, context_text: str, lang: str = "en") -> dict:
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
        if not self.ner_pipeline:
            logger.warning("HF NER pipeline not available.")
            return []

        try:
            logger.debug(f"Extracting entities using HF NER for text: '{text}'")
            entities = self.ner_pipeline(text)
            return entities
        except Exception as e:
            logger.error(f"Error during HF NER entity extraction for text '{text}': {e}", exc_info=True)
            return []

    def generate_chat_response(self, query_text: str, max_length: int = 70) -> str: # Increased max_length slightly
        """
        Generates a conversational response using a text generation model.
        """
        if not self.text_generator:
            logger.warning("Text generation pipeline (DialoGPT) not available.")
            return "Lo siento, mi capacidad de chat general no está disponible en este momento."

        try:
            logger.debug(f"Generating chat response for: '{query_text}'")
            # DialoGPT often includes the input in its output, so we might need to clean it.
            # It also works best as a multi-turn model, but here we use it for single response.
            # For more complex chat, a history/context would be passed.
            response_sequences = self.text_generator(query_text, max_length=max_length, num_return_sequences=1, pad_token_id=50256) # pad_token_id for gpt2-like models
            
            if response_sequences and response_sequences[0]:
                generated_text = response_sequences[0]['generated_text']
                
                # Clean up response: remove the input query if it's prepended
                # This is common with some conversational models.
                if generated_text.startswith(query_text):
                    cleaned_response = generated_text[len(query_text):].strip()
                    # Sometimes it might still have an assistant prefix if it's mimicking dialogue turns
                    # This part might need model-specific fine-tuning for cleaning.
                else:
                    cleaned_response = generated_text
                
                # If after cleaning, the response is empty, or too short, provide a fallback.
                if not cleaned_response or len(cleaned_response) < 5 : # Arbitrary short length
                    logger.info("DialoGPT generated an empty or too short response after cleaning. Using fallback.")
                    return "No estoy seguro de cómo responder a eso." 
                
                logger.info(f"DialoGPT generated response: '{cleaned_response}'")
                return cleaned_response
            else:
                logger.warning("Text generation pipeline returned no sequences.")
                return "No se me ocurre qué decir."
        except Exception as e:
            logger.error(f"Error during text generation for query '{query_text}': {e}", exc_info=True)
            return "Lo siento, tuve un problema al generar una respuesta de chat."

# --- Content from core/ml_models.py ---
def train_sklearn_command_model():
    """
    Trains a scikit-learn model for command-response mapping.
    Uses data loaded via 'load_data()' (expected from utils.database_handler).
    """
    data_for_model = load_data() 
    if not data_for_model:
        logger.warning("No data available to train the scikit-learn command model.")
        return None

    try:
        df = pd.DataFrame(data_for_model)
        if "command" not in df.columns or "response" not in df.columns:
            logger.error("Training data for scikit-learn model is missing 'command' or 'response' columns.")
            return None
        if df.empty:
            logger.warning("Training data DataFrame is empty for scikit-learn model.")
            return None
            
        X = df["command"]
        y = df["response"]

        model = Pipeline([('vect', CountVectorizer()), ('clf', MultinomialNB())])
        model.fit(X, y)
        logger.info("Scikit-learn command model trained successfully.")
        return model
    except Exception as e:
        logger.error(f"Error during scikit-learn model training: {e}", exc_info=True)
        return None

def predict_sklearn_command_response(model, command_text: str) -> str:
    """
    Predicts a response for a command using the trained scikit-learn model.
    """
    if model:
        try:
            prediction = model.predict([command_text])[0]
            return str(prediction)
        except Exception as e:
            logger.error(f"Error during scikit-learn model prediction for '{command_text}': {e}", exc_info=True)
            return "Lo siento, tuve un problema al procesar eso con mi modelo local."
    else:
        logger.warning("Scikit-learn command model not available for prediction.")
        return "No tengo una respuesta predefinida para eso en mi modelo local."

# Example __main__ block for testing (can be uncommented and adapted)
# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
#     logger.info("--- Testing AdvancedNLPProcessor ---")
#     adv_processor = AdvancedNLPProcessor()
#     if adv_processor.sentiment_analyzer_en:
#         logger.info(f"EN Sentiment 'I love this': {adv_processor.analyze_sentiment('I love this', 'en')}")
#     if adv_processor.sentiment_analyzer_es:
#         logger.info(f"ES Sentiment 'odio esto': {adv_processor.analyze_sentiment('odio esto', 'es')}")
    
#     logger.info("\n--- Testing scikit-learn model ---")
#     # This part requires 'load_data' to be functional and data to exist.
#     # You might need to create a dummy utils/database_handler.py with a load_data function
#     # returning sample data for this test to run without the full database setup.
#     # Example:
#     # def load_data(): return [{"command": "hola", "response": "Hola! Cómo estás?"}, 
#     #                          {"command": "adiós", "response": "Hasta luego!"}]
#     sklearn_model = train_sklearn_command_model()
#     if sklearn_model:
#         test_commands = ["hola", "adiós", "qué tal"]
#         for cmd in test_commands:
#             resp = predict_sklearn_command_response(sklearn_model, cmd)
#             logger.info(f"Command: '{cmd}' -> Predicted Response: '{resp}'")
#     else:
#         logger.info("Scikit-learn model training failed or no data.")
