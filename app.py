import os
import re
import sqlite3
from datetime import datetime
import streamlit as st
import bcrypt
import secrets
import random

# -----------------------------
# CONFIG
# -----------------------------
APP_TITLE = "PsyCare (Secure Suggestion Demo)"
DB_PATH = "psycare.db"

WEEKDAY_KEY = {
    "MONDAY": 2, "TUESDAY": 4, "WEDNESDAY": 6, "THURSDAY": 8,
    "FRIDAY": 10, "SATURDAY": 12, "SUNDAY": 14,
}

ADMIN_USERNAME = os.getenv("PSY_ADMIN_USER", "psychologist")
ADMIN_PASSWORD = os.getenv("PSY_ADMIN_PASS", "admin123")  # hashed on first run

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# -----------------------------
# DB HELPERS
# -----------------------------
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash BLOB NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('USER','ADMIN'))
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS messages(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        username TEXT NOT NULL,
        weekday TEXT NOT NULL,
        user_message TEXT NOT NULL,
        llm_suggestion_plain TEXT NOT NULL,
        suggestion_encrypted TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )""")
    conn.commit()
    return conn

def bootstrap_admin(conn):
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE username = ?", (ADMIN_USERNAME,))
    if not cur.fetchone():
        pw_hash = bcrypt.hashpw(ADMIN_PASSWORD.encode(), bcrypt.gensalt())
        cur.execute(
            "INSERT INTO users(username, password_hash, role) VALUES(?,?,?)",
            (ADMIN_USERNAME, pw_
