import polars as pl
import duckdb
from datetime import datetime

# Load the CSV file into a Polars DataFrame
df = pl.read_csv("sample_data.csv")

# Define the filter parameters
description_filter = "Grocery"
start_date = "2025-01-01"
end_date = "2025-01-10"

# Create a DuckDB connection
con = duckdb.connect()

# Register the Polars DataFrame as a table named 'data'
con.register("data", df)

# Define the parameterized SQL query
query = """
    SELECT SUM(amount) as total_amount
    FROM data
    WHERE description ILIKE '%' || ? || '%'
    AND CAST(date AS DATE) BETWEEN ? AND ?
"""

# Execute the query with parameters
result = con.execute(query, [description_filter, start_date, end_date])

# Fetch the result (fetchone() returns a tuple with the sum)
total_amount = result.fetchone()[0] or 0.0

# Print the result
print(f"Sum of amounts for transactions containing '{description_filter}' "
      f"between {start_date} and {end_date}: {total_amount:.2f}")

# Close the DuckDB connection
con.close()