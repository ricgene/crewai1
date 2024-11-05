import pandas as pd
import sqlite3

# Load the dataset
df = pd.read_csv("ds_salaries.csv")

# Create a SQLite database and write the data to a table
connection = sqlite3.connect("salaries.db")
df.to_sql(name="salaries", con=connection, if_exists='replace', index=False)
