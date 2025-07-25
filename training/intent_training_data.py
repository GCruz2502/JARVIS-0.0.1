"""
Central repository for intent recognition training data.
This file is the single source of truth for training the custom Naive Bayes classifiers.
"""

# The format is a list of tuples, where each tuple is (sentence, intent_label).
TRAINING_SAMPLES_ES = [
    # --- INTENT_GET_WEATHER ---
    ("qué tiempo hace hoy", "INTENT_GET_WEATHER"),
    ("dime el clima en panamá", "INTENT_GET_WEATHER"),
    ("va a llover mañana", "INTENT_GET_WEATHER"),
    ("cómo está el tiempo en bogotá", "INTENT_GET_WEATHER"),
    ("temperatura actual por favor", "INTENT_GET_WEATHER"),
    ("dime el pronóstico", "INTENT_GET_WEATHER"),

    # --- INTENT_GET_TIME ---
    ("qué hora es", "INTENT_GET_TIME"),
    ("dime la hora por favor", "INTENT_GET_TIME"),
    ("me dices la hora", "INTENT_GET_TIME"),
    ("la hora actual", "INTENT_GET_TIME"),

    # --- INTENT_GET_DATE ---
    ("qué fecha es hoy", "INTENT_GET_DATE"),
    ("dime la fecha", "INTENT_GET_DATE"),
    ("fecha de hoy", "INTENT_GET_DATE"),
    ("en qué fecha estamos", "INTENT_GET_DATE"),

    # --- INTENT_PLAY_MUSIC (and related) ---
    ("pon música de rock", "INTENT_PLAY_MUSIC"),
    ("quiero escuchar algo de música", "INTENT_PLAY_MUSIC"),
    ("reproduce la canción Imagine de John Lennon", "INTENT_PLAY_SONG"),
    ("pon la canción Bohemian Rhapsody", "INTENT_PLAY_SONG"),
    ("quiero escuchar a Queen", "INTENT_PLAY_ARTIST"),
    ("reproduce algo de Metallica", "INTENT_PLAY_ARTIST"),
    ("pon mi playlist de ejercicio", "INTENT_PLAY_PLAYLIST"),
    ("reproduce la lista de relajación", "INTENT_PLAY_PLAYLIST"),

    # --- INTENT_STOP ---
    ("para la música", "INTENT_STOP"),
    ("detente", "INTENT_STOP"),
    ("para ya", "INTENT_STOP"),
    ("silencio", "INTENT_STOP"),

    # --- INTENT_SET_REMINDER / INTENT_SET_ALARM ---
    ("recuérdame llamar a mamá mañana", "INTENT_SET_REMINDER"),
    ("pon un recordatorio para la reunión de las 3 pm", "INTENT_SET_REMINDER"),
    ("avísame en 15 minutos que revise el horno", "INTENT_SET_REMINDER"),
    ("pon una alarma a las 7 de la mañana", "INTENT_SET_ALARM"),
    ("despiértame a las 6 am", "INTENT_SET_ALARM"),

    # --- INTENT_CANCEL ---
    ("cancela el recordatorio", "INTENT_CANCEL"),
    ("borra la alarma", "INTENT_CANCEL"),
    ("no me lo recuerdes", "INTENT_CANCEL"),

    # --- INTENT_GET_NEWS ---
    ("dame las noticias de hoy", "INTENT_GET_NEWS"),
    ("cuáles son los titulares", "INTENT_GET_NEWS"),
    ("noticias de actualidad", "INTENT_GET_NEWS"),
    ("quiero saber las noticias de tecnología", "INTENT_GET_NEWS"),

    # --- INTENT_BROWSER_CONTROL ---
    ("abre la página de google.com", "INTENT_OPEN_URL"),
    ("muéstrame wikipedia.org", "INTENT_OPEN_URL"),
    ("busca en la web información sobre python", "INTENT_SEARCH_WEB"),
    ("googlea la capital de Francia", "INTENT_SEARCH_WEB"),

    # --- INTERNAL: GREET ---
    ("hola jarvis", "INTENT_GREET"),
    ("buenos días", "INTENT_GREET"),
    ("qué tal", "INTENT_GREET"),
    ("hola", "INTENT_GREET"),

    # --- INTERNAL: FAREWELL ---
    ("adiós", "INTENT_FAREWELL"),
    ("hasta luego", "INTENT_FAREWELL"),
    ("chao", "INTENT_FAREWELL"),
    ("nos vemos", "INTENT_FAREWELL"),

    # --- INTERNAL: HELP ---
    ("ayuda", "INTENT_HELP"),
    ("qué puedes hacer", "INTENT_HELP"),
    ("necesito ayuda", "INTENT_HELP"),
    ("menú de opciones", "INTENT_HELP"),

    # --- INTERNAL: CLEAR_CONTEXT ---
    ("limpia el contexto", "INTENT_CLEAR_CONTEXT"),
    ("olvida nuestra conversación", "INTENT_CLEAR_CONTEXT"),
    ("borra todo", "INTENT_CLEAR_CONTEXT"),
    ("empecemos de nuevo", "INTENT_CLEAR_CONTEXT"),
]


TRAINING_SAMPLES_EN = [
    # --- INTENT_GET_WEATHER ---
    ("what is the weather today", "INTENT_GET_WEATHER"),
    ("tell me the forecast for london", "INTENT_GET_WEATHER"),
    ("is it going to rain tomorrow", "INTENT_GET_WEATHER"),
    ("how's the weather in new york", "INTENT_GET_WEATHER"),
    ("current temperature please", "INTENT_GET_WEATHER"),

    # --- INTENT_GET_TIME ---
    ("what time is it", "INTENT_GET_TIME"),
    ("tell me the time please", "INTENT_GET_TIME"),
    ("can you tell me the current time", "INTENT_GET_TIME"),
    ("what's the time", "INTENT_GET_TIME"),

    # --- INTENT_GET_DATE ---
    ("what is today's date", "INTENT_GET_DATE"),
    ("tell me the date", "INTENT_GET_DATE"),
    ("what date is it", "INTENT_GET_DATE"),
    ("current date please", "INTENT_GET_DATE"),

    # --- INTENT_PLAY_MUSIC (and related) ---
    ("play some rock music", "INTENT_PLAY_MUSIC"),
    ("I want to listen to some music", "INTENT_PLAY_MUSIC"),
    ("play the song Imagine by John Lennon", "INTENT_PLAY_SONG"),
    ("put on the song Bohemian Rhapsody", "INTENT_PLAY_SONG"),
    ("I want to listen to Queen", "INTENT_PLAY_ARTIST"),
    ("play something by Metallica", "INTENT_PLAY_ARTIST"),
    ("play my workout playlist", "INTENT_PLAY_PLAYLIST"),
    ("put on the chill mix", "INTENT_PLAY_PLAYLIST"),

    # --- INTENT_STOP ---
    ("stop the music", "INTENT_STOP"),
    ("stop playing", "INTENT_STOP"),
    ("silence", "INTENT_STOP"),
    ("stop", "INTENT_STOP"),

    # --- INTENT_SET_REMINDER / INTENT_SET_ALARM ---
    ("remind me to call mom tomorrow", "INTENT_SET_REMINDER"),
    ("set a reminder for the 3 pm meeting", "INTENT_SET_REMINDER"),
    ("remind me in 15 minutes to check the oven", "INTENT_SET_REMINDER"),
    ("set an alarm for 7 am", "INTENT_SET_ALARM"),
    ("wake me up at 6 in the morning", "INTENT_SET_ALARM"),

    # --- INTENT_CANCEL ---
    ("cancel the reminder", "INTENT_CANCEL"),
    ("delete my alarm", "INTENT_CANCEL"),
    ("nevermind don't remind me", "INTENT_CANCEL"),

    # --- INTENT_GET_NEWS ---
    ("give me the news for today", "INTENT_GET_NEWS"),
    ("what are the headlines", "INTENT_GET_NEWS"),
    ("show me the latest news", "INTENT_GET_NEWS"),
    ("I want to know the tech news", "INTENT_GET_NEWS"),

    # --- INTENT_BROWSER_CONTROL ---
    ("open the page google.com", "INTENT_OPEN_URL"),
    ("show me wikipedia.org", "INTENT_OPEN_URL"),
    ("search the web for information on python", "INTENT_SEARCH_WEB"),
    ("google the capital of France", "INTENT_SEARCH_WEB"),

    # --- INTERNAL: GREET ---
    ("hello jarvis", "INTENT_GREET"),
    ("good morning", "INTENT_GREET"),
    ("what's up", "INTENT_GREET"),
    ("hi", "INTENT_GREET"),

    # --- INTERNAL: FAREWELL ---
    ("goodbye", "INTENT_FAREWELL"),
    ("see you later", "INTENT_FAREWELL"),
    ("bye bye", "INTENT_FAREWELL"),
    ("talk to you later", "INTENT_FAREWELL"),

    # --- INTERNAL: HELP ---
    ("help", "INTENT_HELP"),
    ("what can you do", "INTENT_HELP"),
    ("I need help", "INTENT_HELP"),
    ("show me the options menu", "INTENT_HELP"),

    # --- INTERNAL: CLEAR_CONTEXT ---
    ("clear the context", "INTENT_CLEAR_CONTEXT"),
    ("forget our conversation", "INTENT_CLEAR_CONTEXT"),
    ("erase everything", "INTENT_CLEAR_CONTEXT"),
    ("let's start over", "INTENT_CLEAR_CONTEXT"),
]