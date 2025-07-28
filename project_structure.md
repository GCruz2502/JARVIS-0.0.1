My_Project/root

├── core/                   				# Componentes centrales de JARVIS
│   ├── __init__.py
│   ├── context_manager.py
│   ├── intent_processor.py
│   ├── my_custom_nlu.py
│   ├── nlp_engine.py 
│   ├── speech_recognition.py
│   └── text_to_speech.py
├── custom_jarvis_models/                   
│   ├── native_bayes_en.json
│   └── native_bayes_es.json
├── data/                   
│   ├── application_config.json
│   ├── jarvis_interactions.db
│   └── runtime_data.json
├── jarvis-env/ 						# Entorno virtual (creado por `python3 -m venv jarvis-env`)
├── memory-bank/           			# Documentación interna y contexto del proyecto
│   ├── activeContext.md
│   ├── productContext.md
│   ├── progress.md
│   ├── projectbrief.md
│   ├── systemPatterns.md
│   └── techContext.md
├── models/       
│   ├── nlp_en/
│   └── nlp_es/
├── Node_modules/       
│   ├── bin/
│   ├── @playwright/
│   ├── playwright/
│   └── playwright-core/
├── plugins/               				# Sistema de plugins extensible
│   ├── __init__.py
│   ├── browser_control.py 			
│   ├── date_plugin.py 				# Plugin para control de fecha
│   ├── music.py 					# Plugin para control de música
│   ├── news_plugin.py 				
│   ├── reminders.py        				# Plugin para gestión de recordatorios
│   ├── time_plugin.py 				# Plugin para gestión de hora
│   └── weather.py          				# Plugin para información meteorológica
├── test/ 							# Tests unitarios y de integración
│   ├── test_commands/
│   ├── test_core/
│    │   ├── __init__.py
│    │   ├── test_ner.py
│    │   ├── test_nlp_suite.py
│   ├── test_plugins/
│   ├── test_ui/
│   └── test_utils/
├── training/ 
│   └── train_nlp_models.py
├── ui/       
│   ├── __init__.py
│   └── cli_interface.py
├── utils/ 
│   ├── __init__.py
│   ├── config_manager.py
│   ├── database_handler.py
│   ├── general_utils.py
│   └── logger.py
├── .env 							# Variables de entorno (API Keys, etc. - No incluir en Git)
├── .gitattributes
├── .gitignore
├── .python-version 				# Especifica la versión de Python del proyecto
├── get-pip.py
├── jarvis_app.log
├── main.py 						# Punto de entrada principal
├── package-lock.json
├── package.json
├── README.md 
├── Recomendaciones.pdf
├── reduced-requirements-0.5.1.txt 	# Dependencias del proyecto