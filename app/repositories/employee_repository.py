"""
Employee Data Repository
Handles all employee database operations.
"""
import csv
import os
import sqlite3
import threading
from typing import List, Tuple, Optional
from app.config.settings import EMPLOYEE_CSV_PATH, EMPLOYEE_DB_PATH
from dotenv import load_dotenv

load_dotenv()

Employee_Columns: List[Tuple[str, str]] = [
    ("employee_index_id", "TEXT"),
    ("full_name", "TEXT"),
    ("first_name", "TEXT"),
    ("last_name", "TEXT"),
    ("date_of_birth", "TEXT"),
    ("age", "INTEGER"),
    ("gender", "TEXT"),
    ("marital_status", "TEXT"),
    ("nationality", "TEXT"),
    ("race", "TEXT"),
    ("work_pass_type", "TEXT"),
    ("mobile_number", "TEXT"),
    ("personal_email", "TEXT"),
    ("work_email", "TEXT"),
    ("industry", "TEXT"),
    ("job_title", "TEXT"),
    ("employment_type", "TEXT"),
    ("employment_status", "TEXT"),
    ("employment_start_date", "TEXT"),
    ("leave_taken", "INTEGER"),
    ("department", "TEXT"),
]


def ensure_employee_db(
    csv_path: str = EMPLOYEE_CSV_PATH, db_path: str = EMPLOYEE_DB_PATH
) -> str:
    """
    Ensure a SQLite DB exists for employee data. If not present, build it from CSV.
    Creates indexes for performance optimization.
    Returns the db_path.
    """
    if os.path.exists(db_path):
        # Check if indexes exist, create them if missing (for existing databases)
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%';")
        existing_indexes = [row[0] for row in cur.fetchall()]
        
        required_indexes = ['idx_department', 'idx_full_name', 'idx_leave_taken', 'idx_employment_status']
        missing_indexes = [idx for idx in required_indexes if idx not in existing_indexes]
        
        if missing_indexes:
            print(f"[DB] Creating missing indexes: {missing_indexes}")
            if 'idx_department' in missing_indexes:
                cur.execute("CREATE INDEX idx_department ON employees(department);")
            if 'idx_full_name' in missing_indexes:
                cur.execute("CREATE INDEX idx_full_name ON employees(full_name);")
            if 'idx_leave_taken' in missing_indexes:
                cur.execute("CREATE INDEX idx_leave_taken ON employees(leave_taken);")
            if 'idx_employment_status' in missing_indexes:
                cur.execute("CREATE INDEX idx_employment_status ON employees(employment_status);")
            conn.commit()
        
        conn.close()
        return db_path

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Employee CSV not found: {csv_path}")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Create table
    columns_sql = ", ".join([f"{name} {ctype}" for name, ctype in Employee_Columns])
    cur.execute(f"CREATE TABLE employees ({columns_sql});")

    # Load CSV
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [
            tuple(row.get(col, "").strip() or None for col, _ in Employee_Columns)
            for row in reader
        ]

    placeholders = ", ".join(["?"] * len(Employee_Columns))
    cur.executemany(
        f"INSERT INTO employees VALUES ({placeholders});",
        rows,
    )

    # Create indexes for frequently queried columns
    print("[DB] Creating indexes for performance optimization...")
    cur.execute("CREATE INDEX idx_department ON employees(department);")
    cur.execute("CREATE INDEX idx_full_name ON employees(full_name);")
    cur.execute("CREATE INDEX idx_leave_taken ON employees(leave_taken);")
    cur.execute("CREATE INDEX idx_employment_status ON employees(employment_status);")
    
    conn.commit()
    conn.close()
    print(f"[DB] Database created with indexes: {db_path}")
    return db_path


# Ensure database exists when module is imported
EMPLOYEE_DB_PATH = ensure_employee_db(EMPLOYEE_CSV_PATH, EMPLOYEE_DB_PATH)

# Thread-local storage for connection reuse (thread-safe)
_thread_local = threading.local()

def _get_connection(db_path: str = EMPLOYEE_DB_PATH) -> sqlite3.Connection:
    """
    Get a thread-local database connection. Reuses connection within the same thread.
    Thread-safe for parallel execution.
    """
    if not hasattr(_thread_local, 'connection') or _thread_local.connection is None:
        _thread_local.connection = sqlite3.connect(db_path, check_same_thread=False)
        # Enable row factory for better result handling
        _thread_local.connection.row_factory = sqlite3.Row
    return _thread_local.connection

def _close_connection():
    """Close the thread-local connection (called on thread exit or cleanup)."""
    if hasattr(_thread_local, 'connection') and _thread_local.connection is not None:
        _thread_local.connection.close()
        _thread_local.connection = None


def run_employee_sql(
    sql_query: str, db_path: str = EMPLOYEE_DB_PATH, row_limit: int = 50
) -> str:
    """
    Execute a read-only SQL query against the employee database.
    Only SELECT statements are allowed. Returns rows as a simple table string.
    Uses connection pooling for better performance.
    """
    sql_lower = sql_query.strip().lower()
    if not sql_lower.startswith("select"):
        return "Only SELECT queries are allowed."

    # Basic safety: block mutation keywords
    forbidden = ["update ", "insert ", "delete ", "drop ", "alter ", "create "]
    if any(word in sql_lower for word in forbidden):
        return "Only read-only SELECT queries are permitted."

    # Get thread-local connection (reused if available)
    conn = _get_connection(db_path)
    cur = conn.cursor()

    try:
        cur.execute(sql_query)
        rows = cur.fetchall()
        # Convert Row objects to tuples for compatibility
        rows = [tuple(row) for row in rows]
        col_names = [desc[0] for desc in cur.description] if cur.description else []
    except Exception as e:
        return f"SQL execution error: {e}"

    if not rows:
        return "No results."

    # Apply row limit for display
    rows = rows[:row_limit]

    # Format as markdown table
    header = " | ".join(col_names)
    separator = " | ".join(["---"] * len(col_names))
    lines = [header, separator]
    for row in rows:
        lines.append(" | ".join([str(val) if val is not None else "" for val in row]))
    return "\n".join(lines)