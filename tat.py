from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3

app = FastAPI()

# SQLite setup
def create_db():
    conn = sqlite3.connect('queries_responses.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS queries_responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_query TEXT NOT NULL,
            bot_response TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

create_db()

class QueryRequest(BaseModel):
    user_query: str
    bot_response: str

@app.post("/save/")
async def save_query_response(request: QueryRequest):
    conn = sqlite3.connect('queries_responses.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO queries_responses (user_query, bot_response)
        VALUES (?, ?)
    ''', (request.user_query, request.bot_response))
    conn.commit()
    conn.close()
    return {"status": "Query and response saved"}

@app.get("/queries/")
async def get_queries():
    conn = sqlite3.connect('queries_responses.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM queries_responses')
    rows = cursor.fetchall()
    conn.close()
    return {"queries": rows}
