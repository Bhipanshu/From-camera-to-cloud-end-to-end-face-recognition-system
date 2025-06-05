import psycopg2

def query_db(name):
    conn = psycopg2.connect(
        dbname="face_db",
        user="face_user",
        password="face_pass",
        host="localhost",
        port="5432"
    )
    cur = conn.cursor()
    cur.execute("SELECT dob, address, phone, image_url FROM person_info WHERE name = %s", (name,))
    row = cur.fetchone()
    conn.close()
    return row if row else ("-", "-", "-", "-")
