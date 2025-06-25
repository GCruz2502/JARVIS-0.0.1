"""
Módulo para gestionar el contexto de la conversación.
"""
import logging
from collections import deque
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class ContextManager:
    def __init__(self, max_history_len: int = 20): # Increased default history length
        """
        Inicializa el Gestor de Contexto.

        Args:
            max_history_len (int): Número máximo de turnos de conversación (objetos de turno enriquecidos) a almacenar.
        """
        # Almacena diccionarios de turno enriquecidos
        self.history = deque(maxlen=max_history_len)
        # Datos específicos del turno actual y los detalles del último turno del usuario/asistente
        self.current_turn_data = {}
        logger.info(f"ContextManager inicializado con longitud máxima de historial: {max_history_len} turnos.")

    def add_utterance(self, speaker: str, text: str, **kwargs):
        """
        Añade una elocución enriquecida al historial de la conversación.

        Args:
            speaker (str): 'user' o 'assistant'.
            text (str): El texto de la elocución.
            **kwargs: Datos adicionales para el turno.
                Para 'user': intent, entities, sentiment, language, etc.
                Para 'assistant': plugin_triggered, action_details, etc.
        """
        if speaker not in ['user', 'assistant']:
            logger.warning(f"Hablante inválido '{speaker}' para elocución. Debe ser 'user' o 'assistant'.")
            return

        turn_details = {
            "speaker": speaker,
            "text": text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        turn_details.update(kwargs) # Incorpora todos los datos adicionales

        self.history.append(turn_details)
        logger.debug(f"Añadido al historial: {turn_details}")

        if speaker == 'user':
            self.current_turn_data['last_user_turn_details'] = turn_details
            # Cuando llega una nueva elocución del usuario, el 'last_assistant_turn_details'
            # sigue siendo el de la respuesta anterior a este nuevo input.
            # No limpiamos 'last_assistant_turn_details' aquí.
        elif speaker == 'assistant':
            self.current_turn_data['last_assistant_turn_details'] = turn_details
            # 'last_user_turn_details' sigue siendo el input que llevó a esta respuesta.

    def get_context_for_processing(self) -> dict:
        """
        Retorna el contexto actual para ser usado por IntentProcessor y plugins.
        Incluye el historial enriquecido y los detalles de los últimos turnos.
        """
        ctx = {
            "history": list(self.history), # Lista de diccionarios de turno enriquecidos
            "last_user_turn_details": self.current_turn_data.get('last_user_turn_details'),
            "last_assistant_turn_details": self.current_turn_data.get('last_assistant_turn_details'),
        }
        # Añadir cualquier otro dato de self.current_turn_data que no esté ya explícitamente en ctx
        # (ej. qa_context_override)
        for key, value in self.current_turn_data.items():
            if key not in ctx: # Evitar sobrescribir las claves ya establecidas
                ctx[key] = value
        return ctx

    def clear_all_context(self):
        """Limpia todo el historial de conversación y datos del turno."""
        self.history.clear()
        self.current_turn_data.clear() # Esto limpiará last_user_turn_details, last_assistant_turn_details y todo lo demás.
        logger.info("Historial de conversación y datos del turno actual limpiados completamente.")

    def set_current_turn_data(self, key: str, value):
        """
        Establece un dato específico para el turno de procesamiento actual.
        Ejemplo: 'qa_context_override' para pruebas, o datos derivados por otros módulos.
        Estos datos se incluirán en `get_context_for_processing`.
        """
        self.current_turn_data[key] = value
        logger.debug(f"Dato del turno establecido: {key} = {value}")

    def get_current_turn_data(self, key: str, default=None):
        """
        Obtiene un dato específico del turno de procesamiento actual.
        Busca en self.current_turn_data.
        """
        return self.current_turn_data.get(key, default)

# Ejemplo de uso (conceptual):
# context_mngr = ContextManager()
#
# # Nuevo input del usuario
# user_input_text = "Qué tiempo hace en Londres?"
# # Asumimos que el IntentProcessor ha procesado esto y tenemos:
# user_intent = {"name": "get_weather", "confidence": 0.9}
# user_entities = [{"text": "Londres", "type": "GPE"}]
# user_lang = "es"
#
# context_mngr.add_utterance(
#     'user',
#     user_input_text,
#     intent=user_intent,
#     entities=user_entities,
#     language=user_lang
# )
#
# current_processing_context = context_mngr.get_context_for_processing()
# print(f"Contexto para procesamiento: {current_processing_context['last_user_turn_details']}")
#
# # ... IntentProcessor usa current_processing_context para decidir la acción ...
# # Supongamos que el plugin de clima genera una respuesta
# assistant_response_text = "El tiempo en Londres es soleado."
# plugin_used = "weather_plugin"
# action_details_info = {"location": "Londres", "condition": "soleado"}
#
# context_mngr.add_utterance(
#     'assistant',
#     assistant_response_text,
#     plugin_triggered=plugin_used,
#     action_details=action_details_info
# )
#
# # Ver el historial
# # for turn in context_mngr.get_context_for_processing()["history"]:
# # print(turn)
#
# # Si el usuario dice "limpiar contexto"
# # context_mngr.clear_all_context()
