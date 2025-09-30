import os
import re
import sqlite3
from datetime import datetime
import streamlit as st
import bcrypt
import secrets
import random
from math import gcd

# -----------------------------
# CONFIG
# -----------------------------
APP_TITLE = "PsyCare (Your personal End-to-End Secure Suggestion System)"
DB_PATH = "psycare.db"

# Weekday -> additive key b for Affine cipher
WEEKDAY_B = {
    "MONDAY": 2, "TUESDAY": 4, "WEDNESDAY": 6, "THURSDAY": 8,
    "FRIDAY": 10, "SATURDAY": 12, "SUNDAY": 14,
}

# Admin bootstrap (override via env if desired)
ADMIN_USERNAME = os.getenv("PSY_ADMIN_USER", "psychologist")
ADMIN_PASSWORD = os.getenv("PSY_ADMIN_PASS", "admin123")  # hashed on first run

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# -----------------------------
# DB HELPERS (with migration)
# -----------------------------
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)

    # Base tables
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
        a_param INTEGER,
        b_param INTEGER,
        user_message TEXT NOT NULL,
        llm_suggestion_plain TEXT NOT NULL,
        suggestion_encrypted TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )""")
    conn.commit()

    # Migration: ensure a_param, b_param exist (older DBs)
    def col_missing(table, col):
        cur = conn.execute(f"PRAGMA table_info({table})")
        return col.upper() not in {row[1].upper() for row in cur.fetchall()}

    if col_missing("messages", "a_param"):
        conn.execute("ALTER TABLE messages ADD COLUMN a_param INTEGER")
    if col_missing("messages", "b_param"):
        conn.execute("ALTER TABLE messages ADD COLUMN b_param INTEGER")
    conn.commit()

    # Backfill any NULL a/b
    cur = conn.cursor()
    cur.execute("SELECT id, b_param FROM messages WHERE a_param IS NULL OR b_param IS NULL")
    rows = cur.fetchall()
    for mid, b in rows:
        if b is None:
            b = 2
        a = next_coprime_after(b, 26)
        conn.execute("UPDATE messages SET a_param=?, b_param=? WHERE id=?", (a, b, mid))
    conn.commit()

    return conn


def bootstrap_admin(conn):
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE username = ?", (ADMIN_USERNAME,))
    if not cur.fetchone():
        pw_hash = bcrypt.hashpw(ADMIN_PASSWORD.encode(), bcrypt.gensalt())
        cur.execute(
            "INSERT INTO users(username, password_hash, role) VALUES(?,?,?)",
            (ADMIN_USERNAME, pw_hash, "ADMIN")
        )
        conn.commit()


def create_user(conn, username, password, role="USER"):
    cur = conn.cursor()
    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    cur.execute("INSERT INTO users(username, password_hash, role) VALUES(?,?,?)",
                (username, pw_hash, role))
    conn.commit()


def get_user(conn, username):
    cur = conn.cursor()
    cur.execute("SELECT id, username, password_hash, role FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    if not row:
        return None
    return {"id": row[0], "username": row[1], "password_hash": row[2], "role": row[3]}

# -----------------------------
# CRYPTO (AFFINE)
# -----------------------------
def next_coprime_after(b: int, modulus: int = 26) -> int:
    """Smallest integer a such that a > b and gcd(a, modulus) == 1."""
    a = b + 1
    while gcd(a, modulus) != 1:
        a += 1
    return a

def affine_encrypt_char(ch: str, a: int, b: int) -> str:
    if ch == " ":
        return " "
    if ch in ALPHABET:
        x = ord(ch) - ord('A')
        y = (a * x + b) % 26
        return chr(y + ord('A'))
    return ""

def affine_encrypt(text: str, a: int, b: int) -> str:
    # Cipher operates on A–Z and spaces only; everything else is dropped
    text = text.upper()
    return "".join(affine_encrypt_char(c, a, b) for c in text if c == " " or c in ALPHABET)

# Helpers to reason about ciphertext length before encrypting
def letters_spaces_only_upper(s: str) -> str:
    """Uppercase and keep only A–Z and spaces (mirrors affine_encrypt filter)."""
    s = s.upper()
    return "".join(ch for ch in s if ch == " " or ("A" <= ch <= "Z"))

def trim_to_word_boundary(s: str, max_chars: int) -> str:
    """Trim string s (already UPPER) to <= max_chars at a space if possible."""
    if len(s) <= max_chars:
        return s
    cut = s.rfind(" ", 0, max_chars + 1)
    return s[: (cut if cut != -1 else max_chars)].rstrip()

# -----------------------------
# SMALL LOCAL "LLM" (natural English, only uppercasing at end)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_small_llm():
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        model_name = os.getenv("SMALL_LLM_NAME", "google/flan-t5-small")
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return tok, mdl
    except Exception:
        return None, None

def tiny_llm_generate_length_aware(prompt_text: str, min_cipher_chars: int = 500, max_cipher_chars: int = 700) -> str:
    """
    Generate supportive text (~550 chars target) in natural English, then UPPERCASE,
    ensuring that the final ciphertext length (A–Z + spaces only) falls within [min,max].
    Achieved by optional short continuations or word-boundary trimming before encryption.
    """
    tok, mdl = load_small_llm()
    seed = int.from_bytes(secrets.token_bytes(4), "big")
    rnd = random.Random(seed)

    def gen_once(context: str) -> str:
        sys_hint = (
            "You are a concise psychologist. Write a supportive and practical suggestion "
            "for the case below in 4–8 short sentences. Avoid lists. Keep it warm and actionable."
        )
        user_case = re.sub(r"\s+", " ", context).strip()
        inp = f"{sys_hint}\nCASE: {user_case}"
        if tok and mdl:
            try:
                ids = tok(inp, return_tensors="pt", truncation=True, max_length=640).input_ids
                out = mdl.generate(
                    ids,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.8,
                    max_new_tokens=360,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=3,
                )
                return tok.decode(out[0], skip_special_tokens=True)
            except Exception:
                pass
        # Fallback natural templates (English sentences)
        templates = [
            "Take a few minutes to breathe slowly and notice how your body feels. "
            "Give yourself permission to pause. Progress does not need to be perfect. "
            "Reach out to a trusted friend if the day feels heavy. Rest is also action.",
            "When your mind races, gently write down what worries you. "
            "Break it into one small step you can try today. "
            "Your feelings are valid, but they do not define your worth. "
            "You are allowed to ask for support.",
            "Step outside for a moment of fresh air and let your senses ground you. "
            "Notice three things you see, two things you hear, one thing you feel. "
            "Healing often begins with small choices repeated with patience."
        ]
        return rnd.choice(templates)

    # initial text
    text = gen_once(prompt_text)

    # try to hit the cipher-length band via continuations
    ATTEMPTS = 3
    for _ in range(ATTEMPTS + 1):
        caps = text.upper()
        filtered = letters_spaces_only_upper(caps)
        if len(filtered) > max_cipher_chars:
            # trim at word boundary, then ensure under max after filtering
            caps_trimmed = trim_to_word_boundary(caps, max_cipher_chars + 20)
            filtered = letters_spaces_only_upper(caps_trimmed)
            while len(filtered) > max_cipher_chars and len(caps_trimmed) > 0:
                caps_trimmed = caps_trimmed[:-1]
                filtered = letters_spaces_only_upper(caps_trimmed)
            return caps_trimmed
        if len(filtered) >= min_cipher_chars:
            return caps

        # too short → generate a short continuation and append
        cont_prompt = f"{text}\nContinue with two brief, supportive sentences that build on the same tone."
        continuation = gen_once(cont_prompt)
        text = (text.rstrip() + " " + continuation.lstrip()).strip()

    # last resort: nudge with one brief sentence
    tail = " Keep your plans simple and gentle for now."
    return (text + tail).upper()

# -----------------------------
# UI HELPERS
# -----------------------------
def login_box(conn):
    st.subheader("Login")
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
    if submit:
        user = get_user(conn, username.strip())
        if not user or not bcrypt.checkpw(password.encode(), user["password_hash"]):
            st.error("Invalid credentials.")
            return None
        st.success(f"Welcome, {user['username']}!")
        return {"id": user["id"], "username": user["username"], "role": user["role"]}
    return None

def signup_box(conn):
    st.subheader("Sign up")
    with st.form("signup_form", clear_on_submit=True):
        username = st.text_input("Choose a username").strip()
        password = st.text_input("Choose a password", type="password")
        password2 = st.text_input("Confirm password", type="password")
        submit = st.form_submit_button("Create account")
    if submit:
        if not username or not password:
            st.error("Username and password required.")
            return
        if password != password2:
            st.error("Passwords do not match.")
            return
        if get_user(conn, username):
            st.error("Username already exists.")
            return
        try:
            create_user(conn, username, password, role="USER")
            st.success("Account created. Please log in.")
        except sqlite3.IntegrityError:
            st.error("Username already exists.")

def render_user_portal(conn, current_user):
    st.header("User Portal")
    st.caption("Share your case details. You will receive a secure response.")

    weekdays = list(WEEKDAY_B.keys())
    weekday = st.selectbox("Select weekday", weekdays, index=0)
    case_text = st.text_area("Your message (case details)")

    if st.button("Send to Psychologist"):
        if not case_text.strip():
            st.warning("Please enter your message.")
            return

        # Generate plaintext (ALL CAPS) with ciphertext length targeting 500–700
        suggestion_plain = tiny_llm_generate_length_aware(case_text, min_cipher_chars=500, max_cipher_chars=700)

        # Affine parameters: b from weekday, a is smallest coprime > b
        b = WEEKDAY_B[weekday]
        a = next_coprime_after(b, 26)

        # Encrypt
        suggestion_cipher = affine_encrypt(suggestion_plain, a, b)

        # Persist
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO messages(user_id, username, weekday, a_param, b_param, user_message, llm_suggestion_plain, suggestion_encrypted, created_at)
            VALUES(?,?,?,?,?,?,?,?,?)
        """, (
            current_user["id"], current_user["username"], weekday, a, b, case_text.strip(),
            suggestion_plain, suggestion_cipher, datetime.utcnow().isoformat()
        ))
        conn.commit()

        st.success("Encrypted suggestion received.")
        wrapped = "\n".join(suggestion_cipher[i:i+80] for i in range(0, len(suggestion_cipher), 80))
        st.code(wrapped, language="text")

    st.divider()
    st.subheader("Your History")
    cur = conn.cursor()
    cur.execute("""
        SELECT weekday, suggestion_encrypted, created_at
        FROM messages WHERE user_id = ?
        ORDER BY id DESC LIMIT 50
    """, (current_user["id"],))
    rows = cur.fetchall()
    if rows:
        for wd, enc, ts in rows:
            st.markdown(f"**{wd.title()}** — {ts}")
            wrapped = "\n".join(enc[i:i+80] for i in range(0, len(enc), 80))
            st.code(wrapped, language="text")
            st.markdown("---")
    else:
        st.info("No messages yet.")

def render_admin_portal(conn, current_user):
    st.header("Admin Portal — Psychologist")
    st.caption("View incoming cases and plaintext suggestions (user sees only the encrypted text).")

    who = st.text_input("Filter by username (optional)").strip()
    cur = conn.cursor()
    if who:
        cur.execute("""
            SELECT username, weekday, a_param, b_param, user_message, llm_suggestion_plain, suggestion_encrypted, created_at
            FROM messages WHERE username = ? ORDER BY id DESC
        """, (who,))
    else:
        cur.execute("""
            SELECT username, weekday, a_param, b_param, user_message, llm_suggestion_plain, suggestion_encrypted, created_at
            FROM messages ORDER BY id DESC
        """)

    rows = cur.fetchall()
    if not rows:
        st.info("No messages yet.")
        return

    for uname, wd, a, b, msg, plain, enc, ts in rows:
        st.markdown(f"**{uname}** • {wd.title()} • {ts}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**User Message**")
            st.write(msg)
            st.markdown("**Encrypted Sent to User**")
            wrapped = "\n".join(enc[i:i+80] for i in range(0, len(enc), 80))
            st.code(wrapped, language="text")
            st.caption(f"Ciphertext length: {len(enc)} chars")
        with c2:
            st.markdown("**Suggestion (Plaintext for Admin, ALL CAPS)**")
            wrapped_plain = "\n".join(plain[i:i+80] for i in range(0, len(plain), 80))
            st.code(wrapped_plain, language="text")
            st.caption(f"AFFINE PARAMS → a = {a}, b = {b}")
        st.markdown("---")

# -----------------------------
# APP
# -----------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="centered")
    st.title(APP_TITLE)
    st.write(
        "This system routes user messages to a psychologist agent, generates a suggestion and returns it back to the user, due to some bug in the system the cipher text is not decrypted for the user, we are sorry for the inconvenience, It will be resovled soon."
    )

    conn = get_db()
    bootstrap_admin(conn)

    if "user" not in st.session_state:
        st.session_state.user = None

    with st.sidebar:
        st.subheader("Account")
        if st.session_state.user:
            st.write(f"Logged in as **{st.session_state.user['username']}** ({st.session_state.user['role']})")
            if st.button("Logout"):
                st.session_state.user = None
                st.rerun()
        else:
            choice = st.radio("Select", ["Login", "Sign up"], index=0, horizontal=True)
            if choice == "Login":
                user = login_box(conn)
                if user:
                    st.session_state.user = user
                    st.rerun()
            else:
                signup_box(conn)

    if not st.session_state.user:
        st.info("Please log in or create an account to continue.")
        return

    if st.session_state.user["role"] == "ADMIN":
        render_admin_portal(conn, st.session_state.user)
    else:
        render_user_portal(conn, st.session_state.user)

if __name__ == "__main__":
    main()
