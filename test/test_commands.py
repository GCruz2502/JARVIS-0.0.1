
from commands import ejecutar_comando

def test_comando_reconocido():
    assert ejecutar_comando("abrir navegador") == True

def test_comando_no_reconocido():
    assert ejecutar_comando("comando falso") == False