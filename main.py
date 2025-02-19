import sys
import os

# Añade las rutas necesarias al sistema
sys.path.append(os.path.abspath("src"))      # Para módulos en src/
sys.path.append(os.path.abspath("core"))     # Para módulos en core/
sys.path.append(os.path.abspath("plugins"))  # Para módulos en plugins/

# Ahora importa los módulos
from src.voice import hablar, escuchar
from src.commands import ejecutar_comando
from src.utils import saludo

def main():
    print(f"{saludo()}, soy JARVIS. ¿En qué puedo ayudarte?")
    hablar(f"{saludo()}, soy JARVIS. ¿En qué puedo ayudarte?")

    while True:
        texto = escuchar()
        if texto:
            if "terminar" in texto:
                hablar("Hasta luego")
                break
            if not ejecutar_comando(texto):
                hablar("Comando no reconocido")

if __name__ == "__main__":
    main()