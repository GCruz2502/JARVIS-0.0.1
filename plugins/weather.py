# jarvis/plugins/weather.py
import requests
import os
import logging
from spacy.tokens import Doc # Added for type hinting

logger = logging.getLogger(__name__)

class Plugin:
    def __init__(self):
        """Inicializa el plugin de clima"""
        self.api_key = os.getenv("WEATHER_API_KEY")
        if not self.api_key:
            logger.warning("No se ha configurado la API key para el clima")

    # Keep can_handle simple for now, or update to check entities if needed
    def can_handle(self, text: str, doc: Doc = None, context: dict = None, entities: list = None) -> bool: # Added entities
        """Determina si este plugin puede manejar el texto de entrada"""
        # Option 1: Simple keyword check (current approach)
        keywords = ["tiempo", "clima", "lluvia", "temperatura", "sol", "pronóstico"] 
        if any(keyword in text.lower() for keyword in keywords):
             return True

        # Option 2: Check doc lemmas (more robust)
        if doc:
            lemmas = ["tiempo", "clima", "llover", "temperatura", "sol", "pronosticar"]
            if any(token.lemma_ in lemmas for token in doc):
                return True
        
        # Option 3: Check merged entities (if passed)
        # if entities:
        #    if any(ent['label'] in ["GPE", "LOC"] for ent in entities): # Example: trigger if location found
        #        return True

        return False # Default if no checks pass

    def handle(self, text: str, doc: Doc = None, context: dict = None, entities: list = None) -> str: # Added entities
        """
        Procesa la solicitud de clima y retorna la información, prioritizing merged entities.
        
        Args:
            text (str): Texto del comando (ej: "¿Cuál es el clima en Madrid?")
            doc (spacy.tokens.Doc, optional): Documento spaCy procesado. Defaults to None.
            context (dict, optional): Contexto adicional. Defaults to None.
            entities (list, optional): Merged list of entities from IntentProcessor. Defaults to None.
            
        Returns:
            str: Información sobre el clima o un mensaje de error
        """
        if not self.api_key:
            return "Lo siento, no tengo configurada la API del clima."

        city = None
        
        # 1. Prioritize merged entities if available
        if entities:
            for ent in entities:
                # Look for GPE (GeoPolitical Entity) or LOC (Location)
                if ent['label'] in ["GPE", "LOC"]:
                    city = ent['text']
                    logger.info(f"Ciudad extraída de merged_entities: {city} (Label: {ent['label']}, Source: {ent['source']})")
                    break 
        
        # 2. Fallback to spaCy doc.ents if merged_entities didn't yield a city
        if not city and doc:
            logger.info("No city found in merged_entities, checking doc.ents...")
            for ent in doc.ents:
                if ent.label_ in ["GPE", "LOC"]:
                    city = ent.text
                    logger.info(f"Ciudad extraída de spaCy Doc: {city} (Entidad: {ent.label_})")
                    break
        
        # 3. Fallback to regex if still no city found
        if not city:
            logger.info("No se encontró ciudad en spaCy Doc, intentando regex sobre el texto.")
            import re
            # Regex mejorada para capturar ciudades con múltiples palabras y después de "en", "de", "para"
            city_match = re.search(r"(?:clima|tiempo|temperatura)\s+(?:en|de|para)\s+([A-Za-záéíóúñÁÉÍÓÚÑ\s]+)", text, re.IGNORECASE)
            if city_match:
                city = city_match.group(1).strip()
                logger.info(f"Ciudad extraída con regex: {city}")

        if not city:
            # Si no se detecta ciudad, usar una por defecto o pedir aclaración
            logger.info("No se pudo extraer la ciudad ni con spaCy ni con regex.")
            return "¿En qué ciudad quieres saber el clima?"
        
        try:
            # Llamada a la API de OpenWeatherMap
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={self.api_key}&units=metric&lang=es"
            response = requests.get(url)
            
            if response.status_code == 200:
                weather_data = response.json()
                temp = weather_data["main"]["temp"]
                condition = weather_data["weather"][0]["description"]
                
                return f"En {city} hay {condition} con una temperatura de {temp}°C."
            else:
                logger.error(f"Error en API del clima: {response.status_code}")
                return f"Lo siento, no pude obtener el clima para {city}."
                
        except Exception as e:
            logger.error(f"Error al consultar el clima: {str(e)}")
            return "Hubo un problema al obtener la información del clima."
