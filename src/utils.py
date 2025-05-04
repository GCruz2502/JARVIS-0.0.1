import json
from datetime import datetime

def saludo():
    hora = datetime.now().hour
    if 5 <= hora < 12:
        return "Buenos días"
    elif 12 <= hora < 19:
        return "Buenas tardes"
    else:
        return "Buenas noches"

def guardar_dato(ruta, clave, valor):
    try:
        with open(ruta, "r+") as f:
            datos = json.load(f)
            datos[clave] = valor
            f.seek(0)
            json.dump(datos, f, indent=4)
    except Exception as e:
        print(f"Error guardando datos: {e}")