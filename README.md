# JARVIS Virtual Assistant

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Descripción

JARVIS es un asistente virtual inteligente diseñado para facilitar tareas diarias a través de comandos de voz y texto. Utiliza técnicas de procesamiento del lenguaje natural y aprendizaje automático para entender y responder a las solicitudes del usuario, mejorando con el tiempo a medida que interactúa.

## ✨ Características Principales

- **Reconocimiento de voz**: Interactúa con JARVIS usando comandos hablados
- **Integración con APIs**: Obtiene información en tiempo real (clima, noticias, etc.)
- **Sistema de plugins**: Arquitectura extensible para añadir fácilmente nuevas funcionalidades
- **Comandos contextuales**: Interpreta solicitudes complejas basadas en conversaciones previas
- **Aprendizaje automático**: Mejora sus respuestas con el tiempo mediante técnicas de ML

## 🚀 Inicio Rápido

### Requisitos previos

- Python 3.8 o superior
- Conexión a Internet para funcionalidades basadas en API
- Micrófono (para interacción por voz)
- **macOS**: PortAudio (`brew install portaudio`) - Necesario para `PyAudio`.

### Instalación

1. **Clonar el repositorio**:
   ```bash
   git clone https://github.com/yourusername/jarvis-assistant.git
   cd jarvis-assistant
   ```

2. **Configurar entorno virtual**:
   ```bash
   python3 -m venv jarvis-env 
   ```
   *(Nota: Se recomienda usar `python3` explícitamente)*

   **Para activar el entorno virtual**:
   - En Windows:
     ```bash
     jarvis-env\Scripts\activate
     ```
   - En macOS/Linux:
     ```bash
     source jarvis-env/bin/activate
     ```

3. **Instalar dependencias**:
   ```bash
   pip install -r reduced-requirements-0.5.1.txt
   ```
   *(Asegúrate de que el entorno virtual esté activado antes de ejecutar este comando)*

4. **Configurar claves de API**:
   - Crea un archivo `.env` en la raíz del proyecto con el siguiente formato:
     ```env
     OPENWEATHER_API_KEY=tu_clave_openweather
     NEWSAPI_KEY=tu_clave_newsapi
     # Añade otras API keys según necesites
     ```

### Uso

1. **Ejecutar JARVIS**:
   ```bash
   python main.py
   ```

2. **Comandos básicos**:
   - "Abrir navegador"
   - "Reproducir música"
   - "¿Cómo está el clima en [ciudad]?"
   - "Dime las noticias de hoy"
   - "Ayuda" (muestra todos los comandos disponibles)
   - "Terminar" o "Adiós" (cierra la aplicación)

## 🧩 Estructura del Proyecto

```
My_Project/
├── .gitattributes
├── .gitignore
├── .python-version
├── main.py                 # Punto de entrada principal
├── README.md               # Este archivo
├── Recomendaciones.pdf
├── reduced-requirements-0.5.1.txt # Dependencias del proyecto
├── .env                    # Variables de entorno (API Keys, etc. - No incluir en Git)
├── config/                 # Configuraciones específicas
│   ├── config.json
│   ├── data.json
│   └── settings.py
├── core/                   # Componentes centrales de JARVIS
│   ├── __init__.py
│   ├── cli.py
│   ├── database.py
│   ├── intent_processor.py
│   ├── ml_models.py
│   └── reportes.py
├── memory-bank/            # Documentación interna y contexto del proyecto
│   ├── activeContext.md
│   ├── productContext.md
│   ├── progress.md
│   ├── projectbrief.md
│   ├── systemPatterns.md
│   └── techContext.md
├── models/                 # Modelos entrenados (ej: ML, Vosk)
├── plugins/                # Sistema de plugins extensible
│   ├── __init__.py
│   ├── music.py
│   └── weather.py
├── src/                    # Código fuente principal (utilidades, voz, comandos)
│   ├── __init__.py
│   ├── commands.py
│   ├── utils.py
│   └── voice.py
├── test/                   # Tests unitarios y de integración
│   └── test_commands.py
└── jarvis-env/             # Entorno virtual (creado por `python3 -m venv jarvis-env`)
```

## 🔌 Sistema de Plugins

JARVIS está diseñado con una arquitectura modular que permite añadir fácilmente nuevas funcionalidades a través de plugins.

### Crear un nuevo plugin

1. Crea un nuevo archivo Python en la carpeta `plugins/` (ej: `mi_plugin.py`)
2. Implementa la función `register` que añade comandos al diccionario principal:

```python
# plugins/mi_plugin.py
def mi_funcion():
    # Implementa tu funcionalidad aquí
    return "Resultado para el usuario"

def register(comandos):
    comandos["activar mi función"] = mi_funcion
```

## 🛠️ Desarrollo

### Requisitos para desarrollo

*(Asegúrate de tener activado el entorno virtual `jarvis-env`)*
```bash
# Si existe un archivo requirements-dev.txt o similar para desarrollo:
# pip install -r requirements-dev.txt 

# O instala dependencias adicionales manualmente:
# pip install pytest flake8 black ... 
```
*(Nota: Actualmente solo existe `reduced-requirements-0.5.1.txt`. Si se necesita un archivo completo para desarrollo, debería crearse)*

### Ejecutar tests

```bash
python -m unittest discover test
```

## 📝 Contribuir

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/amazing-feature`)
3. Haz commit de tus cambios (`git commit -m 'Add some amazing feature'`)
4. Push a la rama (`git push origin feature/amazing-feature`)
5. Abre un Pull Request

## 📜 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## 🔄 Futuras Mejoras

- [ ] Interfaz gráfica de usuario (GUI)
- [ ] Soporte multilenguaje
- [ ] Reconocimiento facial para personalización
- [ ] Integración con dispositivos IoT
- [ ] API REST para acceso remoto
- [ ] Soporte para asistentes como Alexa/Google Home
