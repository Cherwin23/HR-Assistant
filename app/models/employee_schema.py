"""
Employee Data Schema Definition
Single source of truth for employee database schema.
Used by RAG tools, SQL queries, and documentation.
"""

Employee_Schema_Description = """
Table: employees
Columns:
- employee_index_id (TEXT)
- full_name (TEXT)
- first_name (TEXT)
- last_name (TEXT)
- date_of_birth (TEXT)
- age (INTEGER)
- gender (TEXT)
- marital_status (TEXT)
- nationality (TEXT)
- race (TEXT)
- work_pass_type (TEXT)
- mobile_number (TEXT)
- personal_email (TEXT)
- work_email (TEXT)
- industry (TEXT)
- job_title (TEXT)
- employment_type (TEXT)
- employment_status (TEXT)
- employment_start_date (TEXT)
- leave_taken (INTEGER)
- department (TEXT)

Usage guidance:
- For headcount: SELECT COUNT(*) FROM employees WHERE department = 'Engineering';
- For leave taken: SELECT leave_taken FROM employees WHERE full_name LIKE '%John%';
- For employee lookup: SELECT * FROM employees WHERE full_name LIKE '%Jane Doe%';
"""