"""
Módulo para gestionar el contexto de la conversación.
"""
import logging
from collections import deque

logger = logging.getLogger(__name__)

class ContextManager:
    def __init__(self, max_history_len: int = 10):
        """
        Inicializa el Gestor de Contexto.

        Args:
            max_history_len (int): Número máximo de turnos de conversación (usuario + asistente) a almacenar.
        """
        # Almacena tuplas (hablante, texto), ej: ('user', 'hola JARVIS')
        self.history = deque(maxlen=max_history_len) 
        # Datos específicos del turno actual, ej: qa_context_override para pruebas
        self.current_turn_data = {} 
        logger.info(f"ContextManager inicializado con longitud máxima de historial: {max_history_len}.")

    def add_utterance(self, speaker: str, text: str):
        """
        Añade una elocución al historial de la conversación.

        Args:
            speaker (str): 'user' o 'assistant'.
            text (str): El texto de la elocución.
        """
        if speaker not in ['user', 'assistant']:
            logger.warning(f"Hablante inválido '{speaker}' para elocución. Debe ser 'user' o 'assistant'.")
            return
        
        turn = (speaker, text)
        self.history.append(turn)
        logger.debug(f"Añadido al historial: {turn}")

        # Actualizar datos del turno actual basados en el hablante
        if speaker == 'user':
            # Cuando llega una nueva elocución del usuario, es el inicio de un nuevo "turno" de procesamiento.
            # Limpiamos los datos del turno anterior, excepto el último input del usuario y la última respuesta del asistente,
            # que pueden ser útiles para el contexto del turno actual.
            previous_user_utterance = self.current_turn_data.get('last_user_utterance')
            previous_assistant_response = self.current_turn_data.get('last_assistant_response')
            
            self.current_turn_data.clear() # Limpiar datos del turno anterior
            
            if previous_user_utterance: # Restaurar si existían
                 self.current_turn_data['previous_user_utterance'] = previous_user_utterance
            if previous_assistant_response: # Restaurar si existían
                 self.current_turn_data['previous_assistant_response'] = previous_assistant_response

            self.current_turn_data['last_user_utterance'] = text

        elif speaker == 'assistant':
            self.current_turn_data['last_assistant_response'] = text

    def get_context_for_processing(self) -> dict:
        """
        Retorna el contexto actual para ser usado por IntentProcessor y plugins.
        """
        # Esto puede expandirse para incluir contexto derivado más complejo.
        ctx = {
            "history": list(self.history), # Convertir deque a lista para consumo más fácil
            "last_user_utterance": self.current_turn_data.get('last_user_utterance'),
            "last_assistant_response": self.current_turn_data.get('last_assistant_response'),
            "previous_user_utterance": self.current_turn_data.get('previous_user_utterance'),
            "previous_assistant_response": self.current_turn_data.get('previous_assistant_response'),
            # Incluir otros datos relevantes del turno actual
        }
        # Añadir cualquier otro dato de self.current_turn_data que no sea uno de los anteriores
        for key, value in self.current_turn_data.items():
            if key not in ctx:
                ctx[key] = value
        return ctx

    def clear_all_context(self):
        """Limpia todo el historial de conversación y datos del turno."""
        self.history.clear()
        self.current_turn_data.clear()
        logger.info("Historial de conversación y datos del turno actual limpiados completamente.")

    def set_current_turn_data(self, key: str, value):
        """
        Establece un dato específico para el turno de procesamiento actual.
        Ejemplo: 'qa_context_override' para pruebas.
        """
        self.current_turn_data[key] = value
        logger.debug(f"Dato del turno establecido: {key} = {value}")

    def get_current_turn_data(self, key: str, default=None):
        """Obtiene un dato específico del turno de procesamiento actual."""
        return self.current_turn_data.get(key, default)

# Ejemplo de uso (conceptual, estaría en IntentProcessor o el bucle principal de main.py):
# context_mngr = ContextManager()
# 
# # Nuevo input del usuario
# user_input = "Hola JARVIS"
# context_mngr.add_utterance('user', user_input)
# current_processing_context = context_mngr.get_context_for_processing()
# # ... IntentProcessor usa current_processing_context ...
# assistant_output = "¡Hola! ¿Cómo puedo ayudarte hoy?"
# context_mngr.add_utterance('assistant', assistant_output)
#
# # Si el usuario dice "limpiar contexto"
# # context_mngr.clear_all_context()
