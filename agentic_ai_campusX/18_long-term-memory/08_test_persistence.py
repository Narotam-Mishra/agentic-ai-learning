
from dotenv import load_dotenv
from langgraph.store.postgres import PostgresStore
import os

load_dotenv(override=True)

DB_URI = os.getenv("DB_URI")

with PostgresStore.from_conn_string(DB_URI) as store:
    ns = ("user", "u1", "details")
    items = store.search(ns)

for it in items:
    print(f"db_data: {it.value["data"]}")
