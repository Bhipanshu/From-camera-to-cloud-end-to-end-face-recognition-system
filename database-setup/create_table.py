import psycopg2

conn = psycopg2.connect(
    dbname="face_db",
    user="face_user",
    password="face_pass",
    host="localhost",
    port=5432
)

cur = conn.cursor()


create_table_query = """
CREATE TABLE IF NOT EXISTS person_info (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    dob DATE,
    address TEXT,
    phone VARCHAR(15),
    image_url TEXT
);
"""

cur.execute(create_table_query)
print("Table 'person_info' created.")

insert_query = """
INSERT INTO person_info (name, dob, address, phone, image_url)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (name) DO NOTHING;
"""

data = [
    # Example data entry
    ("Mindly Kaling", "1958-08-16", "Bay City, Michigan", "987-654-3210", r"image/path/defined/here/for/reference/image.jpg"),
]

cur.executemany(insert_query, data)
conn.commit()
print("Data inserted successfully.")

cur.execute("SELECT name, dob, address FROM person_info")
rows = cur.fetchall()
print("Inserted Records:")
for row in rows:
    print(row)

# Close connections
cur.close()
conn.close()