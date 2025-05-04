import json
import os
import logging

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def collect_data(command, response):
    """
    Guarda los datos de los comandos y respuestas en un archivo JSON."""
    data = {"command": command, "response": response}

    try:
        if not os.path.exists("data/commands.json"):
           os.makedirs("data")

        with open("data/commands.json", "a") as f:
            json.dump(data, f)
            f.write("\n")
        logger.info(f"Data saved: {command}")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        
        
def load_data():
    """Carga datos históricos de las interacciones"""
    data = []

    try:
        if os.path.exists("data/commands.json"):
            with open("data/commands.json", "r") as f:
                for line in f:
                    try:
                        data.append(json.loads(line.strip()))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON line: {e}")

        else:
            logger.info("No command history found. Starting fresh.")
    except Exception as e:
        logger.error(f"Error loading data: {e}")

    return data