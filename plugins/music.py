# Ejemplo en plugins/music.py
from src.utils import guardar_dato

def reproducir_playlist():
    # Lógica para Spotify
    print("Reproduciendo playlist...")

def register(comandos):
    comandos["reproducir playlist"] = reproducir_playlist