"""
Módulo para la interfaz de línea de comandos (CLI) de JARVIS.
"""
import logging
from core.speech_recognition import escuchar, entrada_texto
from core.text_to_speech import hablar
# IntentProcessor y ContextManager se pasarán como argumentos a start_cli
# from core.intent_processor import IntentProcessor 
# from core.context_manager import ContextManager
from utils.database_handler import collect_data # Para guardar interacciones

logger = logging.getLogger(__name__)

def start_cli(intent_processor, context_manager, config_manager):
    """
    Inicia la interfaz de línea de comandos para interactuar con JARVIS.

    Args:
        intent_processor: Instancia de IntentProcessor.
        context_manager: Instancia de ContextManager.
        config_manager: Instancia de ConfigManager.
    """
    logger.info("Iniciando CLI de JARVIS...")
    
    # Saludo inicial
    initial_greeting = config_manager.get_app_setting("initial_greeting", "Hola, soy JARVIS. ¿Cómo puedo ayudarte hoy?")
    print(f"JARVIS: {initial_greeting}")
    hablar(initial_greeting)

    # El modelo scikit-learn (si se usa) ahora es parte de la lógica interna del IntentProcessor o NLPEngine.
    # Ya no se entrena ni se llama directamente desde aquí.

    while True:
        # Preferir entrada de voz, con fallback a texto
        # Esto podría ser configurable.
        use_voice_input = config_manager.get_app_setting("use_voice_input_by_default", True)
        user_input = None

        if use_voice_input:
            # print("Di tu comando o 'texto' para cambiar a entrada de texto:") # Prompt para el usuario
            # raw_voice_input = escuchar() # Asumiendo que escuchar() imprime "Escuchando..."
            # if raw_voice_input and raw_voice_input.strip().lower() == "texto":
            #     print("Cambiando a entrada de texto.")
            #     hablar("Entendido, puedes escribir tu comando.")
            #     user_input = entrada_texto()
            # elif raw_voice_input:
            #     user_input = raw_voice_input
            # else: # Fallo en reconocimiento de voz o timeout
            #     print("No se detectó voz. Puedes escribir tu comando:")
            #     user_input = entrada_texto()
            # Simplificado por ahora:
            print("Escuchando para comando de voz (o escribe directamente y presiona Enter):")
            # Esta es una simplificación. Una CLI real podría tener un modo de escucha más explícito.
            # O podríamos intentar escuchar y si falla, pedir input.
            # Por ahora, vamos a usar input() y el usuario puede hablar si su sistema lo permite,
            # o escribir. La función `escuchar()` es más para un flujo de solo voz.
            # Para una CLI mixta, `input()` es más directo.
            # Si se quiere priorizar voz:
            # user_input = escuchar()
            # if user_input is None: # Si falla la voz
            #     print("No pude escuchar. Por favor, escribe tu comando:")
            #     user_input = entrada_texto()
            # Por ahora, para la CLI, usaremos entrada_texto directamente.
            # La lógica de voz/texto se puede refinar en main.py o aquí.
            
            # Para una CLI, es más común que el usuario escriba.
            # Si se quiere un modo "escucha continua", sería diferente.
            # Vamos a asumir que el usuario escribe en la CLI.
            # Si se quiere usar voz, el main.py podría tener un loop diferente.
            prompt = config_manager.get_app_setting("cli_prompt", "Tú: ")
            user_input_raw = input(prompt)
            if user_input_raw is None: # EOF, e.g. Ctrl+D
                logger.info("Entrada de usuario es None (EOF), saliendo.")
                break
            user_input = user_input_raw.strip().lower()

        else: # Solo entrada de texto
            prompt = config_manager.get_app_setting("cli_prompt", "Tú: ")
            user_input_raw = input(prompt)
            if user_input_raw is None:
                logger.info("Entrada de usuario es None (EOF), saliendo.")
                break
            user_input = user_input_raw.strip().lower()


        if not user_input: # Si la entrada está vacía después de strip()
            logger.info("Entrada de usuario vacía, continuando.")
            continue

        logger.info(f"Input de usuario: '{user_input}'")
        context_manager.add_utterance('user', user_input)

        if user_input in ["exit", "quit", "bye", "salir", "adiós"]:
            farewell_message = config_manager.get_app_setting("farewell_message", "¡Adiós!")
            print(f"JARVIS: {farewell_message}")
            hablar(farewell_message)
            logger.info("Saliendo de la CLI de JARVIS.")
            break

        # Procesar el comando usando IntentProcessor
        # El lang_hint podría obtenerse de config_manager o ser detectado
        lang_hint = config_manager.get_app_setting("default_language_hint", "es") 
        
        # Pasar el contexto actual al IntentProcessor
        # El IntentProcessor ya no usa self.context directamente de esta manera,
        # sino que el contexto se gestiona a través del ContextManager.
        # El IntentProcessor debería tomar el ContextManager o su estado.
        # Por ahora, el IntentProcessor tiene su propio self.context que actualiza.
        # Esto necesita ser armonizado: IntentProcessor debería usar el ContextManager pasado.
        # (Esta es una simplificación para este paso, la integración real es más profunda)
        
        # Actualización: IntentProcessor.process ahora toma el texto y lang_hint.
        # El contexto se maneja internamente por IntentProcessor si usa el ContextManager
        # o si su propio self.context se actualiza correctamente.
        # La llamada a process en IntentProcessor ya tiene acceso a self.context.
        # Lo que sí es importante es que el ContextManager que pasamos aquí
        # sea el mismo que usa el IntentProcessor, o que se sincronicen.
        # Para este ejemplo, asumimos que IntentProcessor usa el context_manager que se le pasó
        # o que main.py se encarga de que sea la misma instancia.

        # El método process de IntentProcessor ya tiene acceso a su propio self.context.
        # Si queremos que use el context_manager que pasamos aquí, IntentProcessor
        # necesitaría ser modificado para aceptar y usar una instancia de ContextManager.
        # Por ahora, la lógica de contexto en IntentProcessor usa self.context.
        # El main.py deberá asegurarse de que el IntentProcessor y el ContextManager
        # estén coordinados (e.g. IntentProcessor tiene una referencia al ContextManager).

        # Para que el IntentProcessor use el context_manager, necesitaríamos algo como:
        # intent_processor.set_context_manager(context_manager) o pasarlo en process.
        # La versión actual de IntentProcessor.process no toma context_manager.
        # Vamos a asumir que el IntentProcessor ya está configurado con el ContextManager correcto.

        result_dict = intent_processor.process(user_input, lang_hint=lang_hint)
        response_text = result_dict.get("final_response", "No pude procesar eso.")
        
        context_manager.add_utterance('assistant', response_text)

        print(f"JARVIS: {response_text}")
        hablar(response_text)

        # Guardar la interacción (opcional)
        if config_manager.get_app_setting("collect_interaction_data", False):
            try:
                collect_data(user_input, response_text) # De utils.database_handler
            except Exception as e:
                logger.error(f"Error guardando datos de interacción: {e}")

if __name__ == '__main__':
    # Esto es solo para pruebas directas de este módulo.
    # En la aplicación real, main.py configurará e iniciará todo.
    print("Modo de prueba para cli_interface.py")
    
    # Crear instancias dummy/mock para probar
    # Esto requeriría importar las clases y posiblemente configurar mucho.
    # Por simplicidad, este __main__ no ejecutará start_cli sin un setup adecuado.
    
    # class DummyIntentProcessor:
    #     def process(self, text, lang_hint=None):
    #         return {"final_response": f"Procesado (dummy): {text}"}
    # class DummyContextManager:
    #     def add_utterance(self, speaker, text): pass
    # class DummyConfigManager:
    #     def get_app_setting(self, key, default=None): return default

    # intent_proc = DummyIntentProcessor()
    # ctx_mngr = DummyContextManager()
    # cfg_mngr = DummyConfigManager()
    
    # print("Para probar, necesitarías un entorno JARVIS completamente configurado.")
    # print("Ejecuta main.py para la funcionalidad completa.")
    # # start_cli(intent_proc, ctx_mngr, cfg_mngr) # No se puede ejecutar sin más setup
    pass
