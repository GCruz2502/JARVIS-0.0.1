"""
JARVIS Virtual Assistant
Main entry point for the aplication
"""
import sys
import os
import logging

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("jarvis.log"),
        logging.StreamHandler()
    ]   
)

logger=logging.getLogger(__name__)

def setup_environment():
    """Configura el entorno de la aplicación."""
    # Añadir directorios al path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(base_dir)

    # Comprobar archivos necesarios
    required_dirs = ['config', 'core', 'data', 'plugins', 'src']
    for directory in required_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Directorio creado: {directory}")

    # Comprobar si existe el archivo .env
    if not os.path.exists(os.path.join("config", ".env")):
        logger.warning("Archivo .env no encontrado. Las APIs pueden no funcionar correctamente.")

def main():
    """Función principal de JARVIS"""
    try:
        # Configurar el entorno
        setup_environment()

        # Importar después de configurar el entorno
        from src.voice import hablar, escuchar
        from src.commands import ejecutar_comando
        from src.utils import saludo

        # Iniciar el asistente
        logger.info("Iniciando JARVIS...")
        mensaje_bienvenida = f"{saludo()}, soy JARVIS. ¿En qué puedo ayudarte?"
        print(mensaje_bienvenida)
        hablar(mensaje_bienvenida)

        # Bucle principal
        while True:
            texto = escuchar()
            if texto:
                logger.info(f"Comando recibido: {texto}")

                if "Terminar" in texto or "Adios" in texto:
                    hablar("Hasta luego!")
                    break

                if not ejecutar_comando(texto):
                    hablar("Comando no reconocido")
                    # Intentar usar el modelo de ML para responder
                    try:
                        from core.ml_models import train_model, predict_response
                        from core.reportes import chat_with_jarvis

                        model = train_model()
                        response = predict_response(model, texto)

                        if response == "I don't know the answer to that yet.":
                            response = chat_with_jarvis(texto)

                        hablar(response)
                        logger.info(f"Respuesta generada: {response}")

                    except Exception as e:
                        logger.error(f"Error al procesar el ML: {e}")
                        hablar("Lo siento, no puedo procesar la solicitud en este momento.")

    except KeyboardInterrupt:
        logger.info("Interrupción del teclado detectada. Cerrando JARVIS.")
        print("\nJarvis se ha cerrado.")
    except Exception as e:
        logger.error(f"Error círitico: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())