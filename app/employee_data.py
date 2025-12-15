import csv
import os
import sqlite3
from typing import List, Tuple
from dotenv import load_dotenv

load_dotenv()

Employee_CSV_Path = os.getenv("CSV_PATH")
Employee_DB_Path = os.getenv("DB_PATH", "employee_data.db")


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
    csv_path: str = Employee_CSV_Path, db_path: str = Employee_DB_Path
) -> str:
    """
    Ensure a SQLite DB exists for employee data. If not present, build it from CSV.
    Returns the db_path.
    """
    if os.path.exists(db_path):
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

    conn.commit()
    conn.close()
    return db_path


# Ensure database exists when module is imported
EMPLOYEE_DB_PATH = ensure_employee_db(Employee_CSV_Path, Employee_DB_Path)


def run_employee_sql(
    sql_query: str, db_path: str = EMPLOYEE_DB_PATH, row_limit: int = 50
) -> str:
    """
    Execute a read-only SQL query against the employee database.
    Only SELECT statements are allowed. Returns rows as a simple table string.
    """
    sql_lower = sql_query.strip().lower()
    if not sql_lower.startswith("select"):
        return "Only SELECT queries are allowed."

    # Basic safety: block mutation keywords
    forbidden = ["update ", "insert ", "delete ", "drop ", "alter ", "create "]
    if any(word in sql_lower for word in forbidden):
        return "Only read-only SELECT queries are permitted."

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    try:
        cur.execute(sql_query)
        rows = cur.fetchall()
        col_names = [desc[0] for desc in cur.description] if cur.description else []
    except Exception as e:
        conn.close()
        return f"SQL execution error: {e}"

    conn.close()

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