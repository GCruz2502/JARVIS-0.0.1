# JARVIS Virtual Assistant

[![Python Version](https://img.shields.io/badge/python-3.12.4-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Descripción

JARVIS es un asistente virtual inteligente diseñado para facilitar tareas diarias a través de comandos de voz y texto. Utiliza avanzadas técnicas de procesamiento del lenguaje natural (PLN) y modelos de aprendizaje automático para entender y responder a las solicitudes del usuario en español e inglés.

## ✨ Características Principales

- **Reconocimiento de Voz**: Interacción mediante comandos hablados (soporta modo offline con Vosk).
- **Procesamiento Avanzado de Lenguaje Natural (PLN)**:
    - **Comprensión Multilingüe (Español/Inglés)**: Procesamiento adaptado al idioma detectado.
    - **Reconocimiento de Intenciones**: Identifica el propósito del usuario usando spaCy y modelos Zero-Shot de Hugging Face Transformers.
    - **Extracción de Entidades**: Detecta información clave (fechas, horas, lugares, nombres) usando spaCy EntityRuler y modelos NER de Hugging Face.
    - **Análisis de Sentimiento**: Evalúa el tono del usuario para respuestas más empáticas.
    - **Respuesta a Preguntas (QA)**: Capacidad de responder preguntas basadas en contexto.
- **Integración con APIs Externas**: Obtiene información en tiempo real (ej. clima con OpenWeatherMap).
- **Sistema de Plugins Extensible**: Arquitectura modular para añadir nuevas funcionalidades fácilmente.
- **Entrada de Texto como Fallback**: Permite comandos por texto si el reconocimiento de voz falla.

## 🚀 Inicio Rápido

### Requisitos previos

- Python 3.12.4 (según `.python-version`)
- Conexión a Internet para funcionalidades basadas en API y descarga inicial de modelos.
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
├── .python-version         # Especifica la versión de Python del proyecto
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
│   ├── advanced_nlp.py     # Procesador para NLP avanzado (Hugging Face Transformers)
│   ├── cli.py
│   ├── database.py
│   ├── intent_processor.py # Procesador principal de intenciones y NLP (spaCy)
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
│   ├── music.py            # Plugin para control de música
│   ├── reminders.py        # Plugin para gestión de recordatorios
│   └── weather.py          # Plugin para información meteorológica
├── src/                    # Código fuente principal (utilidades, voz, comandos)
│   ├── __init__.py
│   ├── commands.py
│   ├── utils.py
│   └── voice.py
├── test/                   # Tests unitarios y de integración
│   ├── test_commands.py
│   └── test_nlp_suite.py   # Suite de tests para NLP
└── jarvis-env/             # Entorno virtual (creado por `python3 -m venv jarvis-env`)
```

## 🔌 Sistema de Plugins

JARVIS utiliza una arquitectura de plugins para extender su funcionalidad. Cada plugin es una clase que reside en un archivo Python dentro del directorio `plugins/`.

### Estructura de un Plugin

Un plugin típico debe implementar una clase llamada `Plugin` con al menos dos métodos:

- **`can_handle(self, text: str, doc: spacy.tokens.Doc = None, context: dict = None, entities: list = None) -> bool`**:
  Determina si el plugin es capaz de manejar la entrada del usuario.
  - `text`: El texto crudo del usuario.
  - `doc`: El documento procesado por spaCy.
  - `context`: El contexto actual de la conversación.
  - `entities`: Una lista de entidades fusionadas (provenientes de spaCy Ruler, HF NER, y spaCy NER).
  Debe devolver `True` si el plugin puede manejar la solicitud, `False` en caso contrario.

- **`handle(self, text: str, doc: spacy.tokens.Doc = None, context: dict = None, entities: list = None) -> str`**:
  Procesa la solicitud y devuelve una respuesta en formato string.
  Los parámetros son los mismos que en `can_handle`.
  Puede devolver una tupla `(response_string, updated_context_dict)` si necesita actualizar el contexto de la conversación.

### Ejemplo de Plugin Básico

```python
# plugins/mi_plugin_ejemplo.py
import logging
from spacy.tokens import Doc # Para type hinting

logger = logging.getLogger(__name__)

class Plugin:
    def __init__(self):
        logger.info("Mi Plugin de Ejemplo inicializado.")

    def can_handle(self, text: str, doc: Doc = None, context: dict = None, entities: list = None) -> bool:
        return "mi comando de ejemplo" in text.lower()

    def handle(self, text: str, doc: Doc = None, context: dict = None, entities: list = None) -> str:
        # Lógica del plugin aquí
        # Puede usar 'text', 'doc', 'context', y 'entities'
        logger.info(f"Mi Plugin de Ejemplo manejando: {text}")
        if entities:
            logger.info(f"Entidades recibidas: {entities}")
        return "¡Mi plugin de ejemplo ha respondido!"
```

JARVIS cargará automáticamente cualquier archivo `.py` (que no sea `__init__.py` ni empiece con `.` ) en el directorio `plugins/` que contenga una clase `Plugin`.

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
- [ ] Mejorar y expandir el soporte multilenguaje (actualmente EN/ES para NLP avanzado)
- [ ] Reconocimiento facial para personalización
- [ ] Integración con dispositivos IoT
- [ ] API REST para acceso remoto
- [ ] Soporte para asistentes como Alexa/Google Home
