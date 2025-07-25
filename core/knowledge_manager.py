"""
Módulo para la gestión de la base de conocimientos local de JARVIS.
"""
import json
import logging
from pathlib import Path
import sqlite3
import time

logger = logging.getLogger(__name__)

class KnowledgeManager:
    """Gestiona el almacenamiento y recuperación de conocimiento para JARVIS."""
    
    def __init__(self, config_manager):
        """Inicializa el gestor de conocimiento."""
        self.config_manager = config_manager
        self.project_root = Path(config_manager.project_root)
        
        # Definir rutas de almacenamiento
        self.knowledge_base_path = self.project_root / "data" / "knowledge_base"
        self.memory_path = self.project_root / "data" / "memory"
        self.indexes_path = self.project_root / "data" / "indexes"
        
        # Inicializar almacenamiento
        self._initialize_storage()
        
        # Conexión a la base de datos
        self.db_path = self.project_root / "data" / "jarvis_knowledge.db"
        self.conn = self._create_connection()
        if self.conn:
            self._create_tables()
    
    def _initialize_storage(self):
        """Crea la estructura de directorios si no existe."""
        # Crear directorios principales
        self.knowledge_base_path.mkdir(parents=True, exist_ok=True)
        self.memory_path.mkdir(parents=True, exist_ok=True)
        self.indexes_path.mkdir(parents=True, exist_ok=True)
        
        # Crear subdirectorios de knowledge_base
        (self.knowledge_base_path / "facts").mkdir(exist_ok=True)
        (self.knowledge_base_path / "concepts").mkdir(exist_ok=True)
        (self.knowledge_base_path / "procedures").mkdir(exist_ok=True)
        (self.knowledge_base_path / "temporal").mkdir(exist_ok=True)
        
        # Crear subdirectorios de memory
        (self.memory_path / "short_term").mkdir(exist_ok=True)
        (self.memory_path / "medium_term").mkdir(exist_ok=True)
        (self.memory_path / "long_term").mkdir(exist_ok=True)
        
        # Crear subdirectorios de indexes
        (self.indexes_path / "vector_index").mkdir(exist_ok=True)
        (self.indexes_path / "keyword_index").mkdir(exist_ok=True)
        
        logger.info("Estructura de almacenamiento de conocimiento inicializada.")
    
    def _create_connection(self):
        """Crea una conexión a la base de datos SQLite."""
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path))
            logger.info(f"Conexión a la base de datos de conocimiento exitosa: {self.db_path}")
            return conn
        except sqlite3.Error as e:
            logger.error(f"Error al conectar a la base de datos de conocimiento: {e}")
            return None
    
    def _create_tables(self):
        """Crea las tablas necesarias en la base de datos."""
        try:
            cursor = self.conn.cursor()
            
            # Tabla de hechos
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact TEXT NOT NULL,
                category TEXT NOT NULL,
                source TEXT,
                confidence REAL,
                timestamp TEXT,
                expiration TEXT,
                is_permanent INTEGER
            );
            ''')
            
            # Tabla de conceptos
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS concepts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                definition TEXT NOT NULL,
                related_concepts TEXT,
                examples TEXT,
                timestamp TEXT
            );
            ''')
            
            # Tabla de memoria a largo plazo
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS long_term_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                category TEXT,
                importance REAL,
                last_accessed TEXT,
                creation_date TEXT
            );
            ''')
            
            self.conn.commit()
            logger.info("Tablas de la base de datos de conocimiento creadas o verificadas.")
        except sqlite3.Error as e:
            logger.error(f"Error al crear tablas en la base de datos: {e}")
    
    def add_fact(self, fact, category, source=None, confidence=1.0, expiration=None, is_permanent=True):
        """Añade un nuevo hecho a la base de conocimientos."""
        try:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            
            cursor = self.conn.cursor()
            cursor.execute('''
            INSERT INTO facts (fact, category, source, confidence, timestamp, expiration, is_permanent)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (fact, category, source, confidence, timestamp, expiration, 1 if is_permanent else 0))
            
            self.conn.commit()
            logger.info(f"Hecho añadido a la base de conocimientos: {fact}")
            return True
        except Exception as e:
            logger.error(f"Error al añadir hecho a la base de conocimientos: {e}")
            return False
    
    def query_facts(self, category=None, search_term=None, limit=10):
        """Busca hechos en la base de conocimientos."""
        try:
            cursor = self.conn.cursor()
            query = "SELECT * FROM facts WHERE 1=1"
            params = []
            
            if category:
                query += " AND category = ?"
                params.append(category)
            
            if search_term:
                query += " AND fact LIKE ?"
                params.append(f"%{search_term}%")
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            facts = []
            for row in results:
                facts.append({
                    "id": row[0],
                    "fact": row[1],
                    "category": row[2],
                    "source": row[3],
                    "confidence": row[4],
                    "timestamp": row[5],
                    "expiration": row[6],
                    "is_permanent": bool(row[7])
                })
            
            return facts
        except Exception as e:
            logger.error(f"Error al consultar hechos: {e}")
            return []
    
    def add_to_long_term_memory(self, key, value, category=None, importance=0.5):
        """Añade o actualiza un elemento en la memoria a largo plazo."""
        try:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            
            cursor = self.conn.cursor()
            cursor.execute("SELECT id FROM long_term_memory WHERE key = ?", (key,))
            existing = cursor.fetchone()
            
            if existing:
                cursor.execute('''
                UPDATE long_term_memory 
                SET value = ?, category = ?, importance = ?, last_accessed = ?
                WHERE key = ?
                ''', (value, category, importance, timestamp, key))
            else:
                cursor.execute('''
                INSERT INTO long_term_memory (key, value, category, importance, last_accessed, creation_date)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (key, value, category, importance, timestamp, timestamp))
            
            self.conn.commit()
            logger.info(f"Elemento añadido/actualizado en memoria a largo plazo: {key}")
            return True
        except Exception as e:
            logger.error(f"Error al añadir a memoria a largo plazo: {e}")
            return False
    
    def get_from_long_term_memory(self, key):
        """Recupera un elemento de la memoria a largo plazo."""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
            SELECT value, category, importance, last_accessed, creation_date 
            FROM long_term_memory 
            WHERE key = ?
            ''', (key,))
            
            result = cursor.fetchone()
            if result:
                # Actualizar último acceso
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                cursor.execute("UPDATE long_term_memory SET last_accessed = ? WHERE key = ?", (timestamp, key))
                self.conn.commit()
                
                return {
                    "key": key,
                    "value": result[0],
                    "category": result[1],
                    "importance": result[2],
                    "last_accessed": result[3],
                    "creation_date": result[4]
                }
            return None
        except Exception as e:
            logger.error(f"Error al recuperar de memoria a largo plazo: {e}")
            return None
    
    def save_session_memory(self, session_data):
        """Guarda la memoria de sesión actual."""
        try:
            session_file = self.memory_path / "short_term" / "session_memory.json"
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=4)
            logger.info("Memoria de sesión guardada correctamente.")
            return True
        except Exception as e:
            logger.error(f"Error al guardar memoria de sesión: {e}")
            return False
    
    def load_session_memory(self):
        """Carga la memoria de sesión."""
        try:
            session_file = self.memory_path / "short_term" / "session_memory.json"
            if session_file.exists():
                with open(session_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error al cargar memoria de sesión: {e}")
            return {}
    
    def close(self):
        """Cierra la conexión a la base de datos."""
        if self.conn:
            self.conn.close()
            logger.info("Conexión a la base de datos de conocimiento cerrada.")