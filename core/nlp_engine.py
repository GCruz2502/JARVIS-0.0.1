# core/nlp_engine.py
import logging
import requests  # Para la llamada a la API de Ollama
import json  # Para la llamada a la API de Ollama
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Asumiendo que 'load_data' ahora está en 'utils.database_handler'
# Esta ruta de importación debe ser correcta según la ubicación final de load_data.
try:
    from utils.database_handler import load_data
except ImportError:
    logger = logging.getLogger(__name__)  # Define el logger temprano para este mensaje
    logger.error("No se pudo importar 'load_data' desde 'utils.database_handler'. "
                 "El entrenamiento/carga del modelo scikit-learn podría fallar si depende de esto.")
    # Define un load_data ficticio si es absolutamente crítico que el archivo se cargue
    # aunque esto significa que la parte del modelo sklearn no funcionará.
    def load_data():
        logger.error("'load_data' no está disponible debido a un error de importación.")
        return []

logger = logging.getLogger(__name__)

# --- Contenido de core/advanced_nlp.py ---
class AdvancedNLPProcessor:
    def __init__(self):
        """
        Inicializa el Procesador NLP Avanzado, cargando los modelos necesarios
        de Hugging Face Transformers para inglés y español.
        """
        self.sentiment_analyzer_en = None
        self.qa_pipeline_en = None
        # self.sentiment_analyzer_en = None # Línea duplicada eliminada
        # self.qa_pipeline_en = None # Línea duplicada eliminada
        self.sentiment_analyzer_es = None
        self.qa_pipeline_es = None
        self.zero_shot_classifier = None
        self.ner_pipeline = None
        # self.text_generator = None # Se eliminará, en su lugar se usará Ollama

        # Inicializar el Pipeline de Análisis de Sentimiento en Inglés
        try:
            logger.info("Inicializando el pipeline de análisis de sentimiento en inglés de Hugging Face...")
            self.sentiment_analyzer_en = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            logger.info("Pipeline de análisis de sentimiento en inglés de Hugging Face inicializado correctamente.")
        except Exception as e:
            logger.error(f"No se pudo inicializar el pipeline de análisis de sentimiento en inglés de Hugging Face: {e}", exc_info=True)

        # Inicializar el Pipeline de Preguntas y Respuestas en Inglés
        try:
            logger.info("Inicializando el pipeline de preguntas y respuestas en inglés de Hugging Face...")
            self.qa_pipeline_en = pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad"
            )
            logger.info("Pipeline de preguntas y respuestas en inglés de Hugging Face inicializado correctamente.")
        except Exception as e:
            logger.error(f"No se pudo inicializar el pipeline de preguntas y respuestas en inglés de Hugging Face: {e}", exc_info=True)

        # Inicializar el Pipeline de Análisis de Sentimiento en Español
        try:
            logger.info("Inicializando el pipeline de análisis de sentimiento en español de Hugging Face...")
            self.sentiment_analyzer_es = pipeline(
                "sentiment-analysis",
                model="pysentimiento/robertuito-sentiment-analysis"
            )
            logger.info("Pipeline de análisis de sentimiento en español de Hugging Face inicializado correctamente.")
        except Exception as e:
            logger.error(f"No se pudo inicializar el pipeline de análisis de sentimiento en español de Hugging Face: {e}", exc_info=True)

        # Inicializar el Pipeline de Preguntas y Respuestas en Español
        try:
            logger.info("Inicializando el pipeline de preguntas y respuestas en español de Hugging Face...")
            self.qa_pipeline_es = pipeline(
                "question-answering",
                model="mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"
            )
            logger.info("Pipeline de preguntas y respuestas en español de Hugging Face inicializado correctamente.")
        except Exception as e:
            logger.error(f"No se pudo inicializar el pipeline de preguntas y respuestas en español de Hugging Face: {e}", exc_info=True)

        # Inicializar el Pipeline de Clasificación Zero-Shot
        try:
            logger.info("Inicializando el pipeline de clasificación zero-shot de Hugging Face...")
            self.zero_shot_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
            logger.info("Pipeline de clasificación zero-shot de Hugging Face inicializado correctamente.")
        except Exception as e:
            logger.error(f"No se pudo inicializar el pipeline de clasificación zero-shot de Hugging Face: {e}", exc_info=True)

        # Inicializar el Pipeline NER (Multilingüe)
        try:
            logger.info("Inicializando el pipeline NER de Hugging Face...")
            self.ner_pipeline = pipeline(
                "ner",
                model="dslim/bert-base-NER",
                grouped_entities=True
            )
            logger.info("Pipeline NER de Hugging Face inicializado correctamente.")
        except Exception as e:
            logger.error(f"No se pudo inicializar el pipeline NER de Hugging Face: {e}", exc_info=True)

        # Inicializar el Pipeline de Generación de Texto (DialoGPT)
        # try:
        #     logger.info("Inicializando el pipeline de generación de texto de Hugging Face (Llama-3-8B-Instruct-hf)...")
        #     # Asegúrate de haber iniciado sesión en Hugging Face CLI y haber aceptado los términos de Llama 3
        #     self.text_generator = pipeline("text-generation", model="meta-llama/Llama-3-8B-Instruct-hf", device_map="auto") # device_map="auto" ayuda con modelos grandes
        #     logger.info("Pipeline de generación de texto de Hugging Face (Llama-3-8B-Instruct-hf) inicializado correctamente.")
        # except Exception as e:
        #     logger.error(f"No se pudo inicializar el pipeline de generación de texto de Hugging Face (Llama-3-8B-Instruct-hf): {e}", exc_info=True)
        #     # TRANSFORMERS_AVAILABLE en report_generator.py era una bandera global.
        #     # Aquí, self.text_generator permanecerá como None si falla.
        # Pipeline de generación de texto eliminado, se usará la API de Ollama en su lugar

    def analyze_sentiment(self, text: str, lang: str = "en") -> dict:
        analyzer = None
        if lang == "es" and self.sentiment_analyzer_es:
            analyzer = self.sentiment_analyzer_es
        elif lang == "en" and self.sentiment_analyzer_en:
            analyzer = self.sentiment_analyzer_en
        else:
            logger.warning(f"Analizador de sentimiento para el idioma '{lang}' no disponible o no cargado.")
            return {"error": f"Analizador de sentimiento para el idioma '{lang}' no disponible."}

        try:
            logger.debug(f"Analizando el sentimiento del texto (idioma={lang}): '{text}'")
            result = analyzer(text)
            if result and isinstance(result, list):
                return result[0]
            return {"error": "No se pudo obtener el sentimiento del resultado del pipeline."}
        except Exception as e:
            logger.error(f"Error durante el análisis de sentimiento para el texto (idioma={lang}) '{text}': {e}", exc_info=True)
            return {"error": f"Error durante el análisis de sentimiento: {str(e)}"}

    def answer_question(self, question: str, context_text: str, lang: str = "en") -> dict:
        q_pipeline = None
        if lang == "es" and self.qa_pipeline_es:
            q_pipeline = self.qa_pipeline_es
        elif lang == "en" and self.qa_pipeline_en:
            q_pipeline = self.qa_pipeline_en
        else:
            logger.warning(f"Pipeline de preguntas y respuestas para el idioma '{lang}' no disponible o no cargado.")
            return {"error": f"Pipeline de preguntas y respuestas para el idioma '{lang}' no disponible."}

        try:
            logger.debug(f"Respondiendo a la pregunta (idioma={lang}): '{question}' con el contexto: '{context_text[:100]}...'")
            result = q_pipeline(question=question, context=context_text)
            return result
        except Exception as e:
            logger.error(f"Error durante la respuesta a la pregunta (idioma={lang}) '{question}': {e}", exc_info=True)
            return {"error": f"Error durante la respuesta a la pregunta: {str(e)}"}

    def classify_intent(self, text: str, candidate_labels: list, multi_label: bool = False) -> dict:
        if not self.zero_shot_classifier:
            logger.warning("Clasificador zero-shot no disponible.")
            return {"error": "Clasificador zero-shot no disponible."}
        
        if not candidate_labels:
             logger.warning("No se proporcionaron etiquetas candidatas para la clasificación zero-shot.")
             return {"error": "No se proporcionaron etiquetas candidatas."}

        try:
            logger.debug(f"Clasificando la intención del texto: '{text}' con las etiquetas: {candidate_labels}")
            result = self.zero_shot_classifier(text, candidate_labels, multi_label=multi_label)
            return result
        except Exception as e:
            logger.error(f"Error durante la clasificación zero-shot para el texto '{text}': {e}", exc_info=True)
            return {"error": f"Error durante la clasificación zero-shot: {str(e)}"}

    def extract_entities_hf(self, text: str) -> list:
        if not self.ner_pipeline:
            logger.warning("Pipeline NER de HF no disponible.")
            return []

        try:
            logger.debug(f"Extrayendo entidades usando NER de HF para el texto: '{text}'")
            entities = self.ner_pipeline(text)
            return entities
        except Exception as e:
            logger.error(f"Error durante la extracción de entidades NER de HF para el texto '{text}': {e}", exc_info=True)
            return []

    def generate_chat_response(self, query_text: str, model_tag: str = "llama3.1:8b") -> str:  # Etiqueta de modelo predeterminada actualizada
        """
        Genera una respuesta conversacional utilizando Llama 3 a través de la API de Ollama.
        Args:
            query_text (str): La entrada del usuario.
            model_tag (str): La etiqueta del modelo Ollama a utilizar (por ejemplo, "llama3:8b-instruct").
        Returns:
            str: La respuesta generada o un mensaje de error.
        """
        ollama_api_url = "http://localhost:11434/api/chat"
        logger.debug(f"Enviando consulta a Ollama ({model_tag}): '{query_text}'")

        payload = {
            "model": model_tag,
            "messages": [
                {"role": "user", "content": query_text}
            ],
            "stream": False  # Queremos la respuesta completa de una vez
        }

        try:
            response = requests.post(ollama_api_url, json=payload, timeout=180)  # Aumenta el tiempo de espera a 180 segundos
            response.raise_for_status()  # Lanza una excepción para errores HTTP

            response_data = response.json()

            if response_data and "message" in response_data and "content" in response_data["message"]:
                assistant_response = response_data["message"]["content"].strip()
                logger.info(f"Ollama ({model_tag}) generó la respuesta: '{assistant_response}'")
                if not assistant_response or len(assistant_response) < 5:
                    logger.info(f"Ollama ({model_tag}) generó una respuesta vacía o demasiado corta. Usando fallback.")
                    return "No estoy seguro de cómo responder a eso."
                return assistant_response
            else:
                logger.warning(f"Formato de respuesta de Ollama ({model_tag}) inesperado: {response_data}")
                return "No se me ocurre qué decir."

        except requests.exceptions.Timeout:
            logger.error(f"Tiempo de espera agotado al conectar con la API de Ollama en {ollama_api_url} para el modelo {model_tag}.")
            return "Lo siento, el servicio de chat tardó demasiado en responder."
        except requests.exceptions.ConnectionError:
            logger.error(f"Error de conexión con la API de Ollama en {ollama_api_url}. ¿Está Ollama en ejecución?")
            return "Lo siento, no pude conectarme al servicio de chat. Asegúrate de que Ollama esté en ejecución."
        except requests.exceptions.RequestException as e:
            logger.error(f"Error durante la solicitud a la API de Ollama para el modelo {model_tag}: {e}", exc_info=True)
            return "Lo siento, tuve un problema al comunicarme con el servicio de chat."
        except Exception as e:
            logger.error(f"Error inesperado durante la generación de respuesta de chat de Ollama (modelo {model_tag}): {e}", exc_info=True)
            return "Lo siento, tuve un problema inesperado al generar una respuesta de chat."

# --- Contenido de core/ml_models.py ---
def train_sklearn_command_model():
    """
    Entrena un modelo scikit-learn para el mapeo comando-respuesta.
    Utiliza los datos cargados a través de 'load_data()' (esperado desde utils.database_handler).
    """
    data_for_model = load_data() 
    if not data_for_model:
        logger.warning("No hay datos disponibles para entrenar el modelo de comando scikit-learn.")
        return None

    try:
        df = pd.DataFrame(data_for_model)
        if "command" not in df.columns or "response" not in df.columns:
            logger.error("Faltan las columnas 'command' o 'response' en los datos de entrenamiento para el modelo scikit-learn.")
            return None
        if df.empty:
            logger.warning("El DataFrame de datos de entrenamiento está vacío para el modelo scikit-learn.")
            return None
        
        X = df["command"]
        y = df["response"]

        model = Pipeline([('vect', CountVectorizer()), ('clf', MultinomialNB())])
        model.fit(X, y)
        logger.info("Modelo de comando scikit-learn entrenado correctamente.")
        return model
    except Exception as e:
        logger.error(f"Error durante el entrenamiento del modelo scikit-learn: {e}", exc_info=True)
        return None

def predict_sklearn_command_response(model, command_text: str) -> str:
    """
    Predice una respuesta para un comando utilizando el modelo scikit-learn entrenado.
    """
    if model:
        try:
            prediction = model.predict([command_text])[0]
            return str(prediction)
        except Exception as e:
            logger.error(f"Error durante la predicción del modelo scikit-learn para '{command_text}': {e}", exc_info=True)
            return "Lo siento, tuve un problema al procesar eso con mi modelo local."
    else:
        logger.warning("Modelo de comando scikit-learn no disponible para la predicción.")
        return "No tengo una respuesta predefinida para eso en mi modelo local."

# Bloque __main__ de ejemplo para pruebas (se puede descomentar y adaptar)
# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
#     logger.info("--- Probando AdvancedNLPProcessor ---")
#     adv_processor = AdvancedNLPProcessor()
#     if adv_processor.sentiment_analyzer_en:
#         logger.info(f"Sentimiento EN 'I love this': {adv_processor.analyze_sentiment('I love this', 'en')}")
#     if adv_processor.sentiment_analyzer_es:
#         logger.info(f"ES Sentiment 'odio esto': {adv_processor.analyze_sentiment('odio esto', 'es')}")
    
#     logger.info("\n--- Testing scikit-learn model ---")
#     # Esta parte requiere que 'load_data' sea funcional y que existan datos.
#     # Es posible que necesites crear un utils/database_handler.py ficticio con una función load_data
#     # que devuelva datos de muestra para que esta prueba se ejecute sin la configuración completa de la base de datos.
#     # Ejemplo:
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
