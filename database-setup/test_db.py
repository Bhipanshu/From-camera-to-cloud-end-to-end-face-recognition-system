import psycopg2

conn = psycopg2.connect(
    dbname="face_db",
    user="face_user",
    password="face_pass",
    host="localhost",
    port=5432
)
print("Connected to PostgreSQL successfully!")
conn.close()
