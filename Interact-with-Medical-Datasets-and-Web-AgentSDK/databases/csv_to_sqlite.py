import pandas as pd
import sqlite3
import os

# File paths
datasets = {
    "heart_disease": "F:\Courses\Ostad\Assignments\Multi-Tool AI Agent to Interact with Medical Datasets and Web\medical_ai_agent\data\heart disease dataset.csv",
    "cancer": "F:\Courses\Ostad\Assignments\Multi-Tool AI Agent to Interact with Medical Datasets and Web\medical_ai_agent\data\The Cancer data.csv",
    "diabetes": "F:\Courses\Ostad\Assignments\Multi-Tool AI Agent to Interact with Medical Datasets and Web\medical_ai_agent\data\diabetes.csv",
}

# Output folder
# os.makedirs("databases", exist_ok=True)

for name, path in datasets.items():
    print(f"Processing {name}...")

    # Load CSV
    df = pd.read_csv(path)

    # Show quick info
    print(df.head())
    print(df.dtypes)

    # Create DB connection
    db_path = f"F:\Courses\Ostad\Assignments\Multi-Tool AI Agent to Interact with Medical Datasets and Web\medical_ai_agent\databases\{name}.db"
    conn = sqlite3.connect(db_path)

    # Store into SQLite (table name = records)
    df.to_sql(f"{name}_records", conn, if_exists="replace", index=False)

    conn.close()
    print(f"✅ Saved {name}.db with table {name}_records\n")
