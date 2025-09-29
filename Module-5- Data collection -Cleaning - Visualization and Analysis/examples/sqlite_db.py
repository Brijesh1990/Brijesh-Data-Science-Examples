import pandas as pd
from sqlalchemy import create_engine

# 1. Define the connection string (DB-specific format)
# This example is for PostgreSQL: 'dialect+driver://user:password@host:port/database'
DATABASE_URL = "postgresql+psycopg2://user:pass@localhost:5432/analytics_db"

# 2. Create the database engine
engine = create_engine(DATABASE_URL)

# 3. Define the SQL query
sql_query = "SELECT product_name, price FROM products WHERE stock > 0;"

# 4. Use pandas to execute the query and load results into a DataFrame
df = pd.read_sql(sql_query, engine)

print(df.head())