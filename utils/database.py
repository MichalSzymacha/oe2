import sqlite3

def create_table():
    conn = sqlite3.connect("results.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ga_results (
            iteration INTEGER,
            best_value REAL,
            mean_value REAL,
            std_value REAL
        )
    """)
    conn.commit()
    conn.close()

def insert_result(iteration, best_val, mean_val, std_val):
    conn = sqlite3.connect("results.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO ga_results (iteration, best_value, mean_value, std_value)
        VALUES (?, ?, ?, ?)
    """, (iteration, best_val, mean_val, std_val))
    conn.commit()
    conn.close()
