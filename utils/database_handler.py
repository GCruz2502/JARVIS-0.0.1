import sqlite3
import json
import os
import logging
from pathlib import Path

# Define the database path
DATABASE_PATH = "data/jarvis_interactions.db"

# logging.basicConfig(level=logging.INFO, 
#                    format='%(asctime)s - %(levelname)s - %(message)s') # Removed: Logging should be configured by main.py
logger = logging.getLogger(__name__)

def create_connection():
    """Creates a database connection to the SQLite database."""
    conn = None
    try:
        # Ensure the 'data' directory exists
        Path("data").mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DATABASE_PATH)
        logger.info(f"Connection to SQLite database successful: {DATABASE_PATH}")
    except sqlite3.Error as e:
        logger.error(f"Error connecting to SQLite database: {e}")
    return conn

def create_table(conn):
    """Creates the 'interactions' table if it doesn't exist."""
    try:
        sql_create_interactions_table = """
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            user_input TEXT,
            intent TEXT,
            entities TEXT,
            sentiment TEXT,
            plugin_used TEXT,
            response TEXT,
            success INTEGER,
            language TEXT
        );
        """
        cursor = conn.cursor()
        cursor.execute(sql_create_interactions_table)
        conn.commit()
        logger.info("Interactions table created successfully or already exists.")
    except sqlite3.Error as e:
        logger.error(f"Error creating interactions table: {e}")

def collect_data(timestamp, user_input, intent, entities, sentiment, plugin_used, response, success, language):
    """
    Saves interaction data to the SQLite database.
    """
    conn = create_connection()
    if conn is None:
        return False

    try:
        sql = """
        INSERT INTO interactions (timestamp, user_input, intent, entities, sentiment, plugin_used, response, success, language)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor = conn.cursor()
        cursor.execute(sql, (timestamp, user_input, intent, entities, sentiment, plugin_used, response, success, language))
        conn.commit()
        logger.info(f"Interaction data saved to database: {user_input}")
        return True
    except sqlite3.Error as e:
        logger.error(f"Error saving interaction data to database: {e}")
        return False
    finally:
        if conn:
            conn.commit()
            conn.close()

def load_data():
    """Loads interaction data from the SQLite database (example)."""
    conn = create_connection()
    if conn is None:
        return []

    data = []
    try:
        sql = "SELECT * FROM interactions"
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()

        for row in rows:
            # Convert row to a dictionary (adjust column indices as needed)
            interaction = {
                "id": row[0],
                "timestamp": row[1],
                "user_input": row[2],
                "intent": row[3],
                "entities": row[4],
                "sentiment": row[5],
                "plugin_used": row[6],
                "response": row[7],
                "success": row[8],
                "language": row[9]
            }
            data.append(interaction)
        logger.info(f"Loaded {len(data)} interactions from the database.")
    except sqlite3.Error as e:
        logger.error(f"Error loading data from database: {e}")
    finally:
        if conn:
            conn.close()
    return data

# Initialize the database and table on import
conn = create_connection()
if conn:
    create_table(conn)
    conn.close()
