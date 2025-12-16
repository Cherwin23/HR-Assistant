"""
Employee Data SQL Tool
LangChain tool for querying employee database.
"""
from langchain_core.tools import tool
from app.repositories.employee_repository import EMPLOYEE_DB_PATH, run_employee_sql
from app.models.employee_schema import Employee_Schema_Description


@tool
def employee_data_sql_tool(sql_query: str) -> str:
    """
    Execute a read-only SQL query against the employee data (SQLite).
    Only SELECT statements are allowed. Table name: employees
    Schema is injected at runtime.
    """
    return run_employee_sql(sql_query, db_path=EMPLOYEE_DB_PATH)

# Inject schema description into tool
employee_data_sql_tool.description = f"""
Execute a read-only SQL query against the employee data (SQLite).
- Only SELECT statements are allowed.
- Table name: employees
- Use exact SQL, not natural language.

Schema:
{Employee_Schema_Description}
"""