import openai
import duckdb
import os
import pandas as pd

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")  # Or hardcode: 'your-api-key-here'
if not openai.api_key:
    raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

# DuckDB connection (in-memory database)
con = duckdb.connect()

# CSV file path
csv_file = 'data.csv'

# Check if CSV file exists
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"CSV file '{csv_file}' not found. Please ensure it exists in the current directory.")

# Create table
try:
    con.sql("""
    CREATE TABLE IF NOT EXISTS transactions (
        date DATE,
        description VARCHAR,
        amount DECIMAL
    )
    """)
except duckdb.CatalogException as e:
    print(f"Error creating table: {e}")

# Load CSV data into the table
try:
    con.sql(f"""
    INSERT INTO transactions
    SELECT * FROM read_csv_auto('{csv_file}', header=true, dateformat='%Y-%m-%d')
    """)
    print("CSV data loaded successfully.")
except duckdb.IOException as e:
    print(f"Error loading CSV: {e}")
    print("Ensure the CSV has columns 'date', 'description', 'amount' and valid data.")
    exit(1)

def english_to_sql(command):
    """
    Use OpenAI to convert English CRUD command to SQL for DuckDB.
    """
    prompt = f"""
    You are a SQL expert for DuckDB. The table is named 'transactions' with columns:
    - date: DATE
    - description: VARCHAR
    - amount: DECIMAL

    Convert the following English command to valid SQL (INSERT, SELECT, UPDATE, or DELETE).
    Do not add explanations, just the SQL statement.
    For SELECT, return all matching rows.
    Ensure the SQL is executable in DuckDB.

    Command: {command}
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Or 'gpt-3.5-turbo'
            messages=[
                {"role": "system", "content": "You are a helpful SQL translator."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.0
        )
        sql = response.choices[0].message['content'].strip()
        return sql
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return None

def execute_sql(sql):
    """
    Execute the SQL in DuckDB and handle results.
    """
    if not sql:
        print("No SQL generated.")
        return
    
    try:
        result = con.sql(sql)
        if sql.strip().upper().startswith("SELECT"):
            # For SELECT queries, fetch and print results
            df = result.df()
            if df.empty:
                print("No results found.")
            else:
                print("Query Results:")
                print(df)
        else:
            # For INSERT/UPDATE/DELETE
            print("Operation executed successfully.")
            # Save changes back to CSV
            con.sql(f"COPY transactions TO '{csv_file}' (FORMAT CSV, HEADER)")
            print("Changes saved to CSV.")
    except duckdb.Error as e:
        print(f"Error executing SQL: {e}")

# Interactive loop
if __name__ == "__main__":
    print("Enter CRUD commands in English (e.g., 'Add a new entry with date 2023-01-01, description Groceries, amount 50.00')")
    print("Type 'exit' to quit.")
    
    while True:
        command = input("Command: ").strip()
        if command.lower() == 'exit':
            break
        if not command:
            continue
        
        sql = english_to_sql(command)
        print(f"Generated SQL: {sql}")
        execute_sql(sql)

    # Close the DuckDB connection
    con.close()