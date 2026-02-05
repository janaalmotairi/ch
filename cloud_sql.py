from db import execute_sql
from sql_utils import normalize_sql, pretty_answer


def ask_cloud_sql(client, question: str) -> str:
    # Prompt sent to the LLM to generate SQL only
    prompt = f"""
You are a SQL generator.
Rules:
- Output ONLY a valid SQLite SELECT query.
- Table name: employees
- No explanations.
- Use ONLY these exact column names:
  Age, Attrition, Department, JobRole, MonthlyIncome, OverTime, EducationField, JobSatisfaction, WorkLifeBalance
- Department values are EXACTLY:
  'Human Resources', 'Research & Development', 'Sales'
Question:
{question}
SQL:
""".strip()

    # Call Groq / LLM to generate SQL
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Return SQL only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=200,
    )

    # Raw SQL output from the model
    raw = completion.choices[0].message.content

    # Normalize and clean SQL (fix column names, formatting, etc.)
    sql = normalize_sql(raw)

    # Basic validation to ensure we only run SELECT queries
    if not sql.lower().startswith("select"):
        return f"Invalid SQL generated.\nRaw: {str(raw).strip()[:200]}"

    # Execute SQL against the database
    try:
        cols, rows = execute_sql(sql)
    except Exception as e:
        return f"SQL execution error: {e}\nSQL was: {sql}"

    # Handle empty result sets
    if not rows:
        return "No results found."

    # Convert results into a user-friendly answer
    return pretty_answer(question, cols, rows)
