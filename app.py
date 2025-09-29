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
    if not row: return None
    return {"id": row[0], "username": row[1], "password_hash": row[2], "role": row[3]}

# -----------------------------
# CRYPTO (CAESAR)
# -----------------------------
def caesar_shift_char(ch, k):
    if ch == " ":
        return " "
    if ch in ALPHABET:
        idx = (ALPHABET.index(ch) + k) % 26
        return ALPHABET[idx]
    return ""

def caesar_encrypt(text, k):
    text = text.upper()
    return "".join(caesar_shift_char(c, k) for c in text)

# -----------------------------
# LOCAL “SMALL LLM” GENERATOR (FLAN-T5-SMALL) + SANITIZER
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

def sanitize_upper_space_len(text: str) -> str:
    text = text.upper()
    text = re.sub(r"[^A-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 200:
        text = text[:200].rstrip()
    if len(text) < 150:
        # deterministic padding that still reads natural-ish
        pad = " BREATHE IN BREATHE OUT NOTICE YOUR BODY AND SOFTEN YOUR SHOULDERS "
        while len(text) < 150:
            text += pad
            text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 200:
            text = text[:200].rstrip()
    return text

def tiny_llm_generate(prompt_text: str) -> str:
    """
    Try flan-t5-small on CPU with sampling; sanitize and enforce length.
    Fallback to template generator if model unavailable.
    """
    tok, mdl = load_small_llm()
    seed = int.from_bytes(secrets.token_bytes(4), "big")
    rnd = random.Random(seed)

    sys_hint = (
        "You are a concise psychologist. Write a supportive, practical suggestion "
        "for the case below in 1-2 sentences, ~180 characters. Avoid lists. "
        "Keep it simple and warm."
    )
    user_case = re.sub(r"\s+", " ", prompt_text).strip()
    inp = f"{sys_hint}\nCASE: {user_case}"

    if tok and mdl:
        try:
            ids = tok(inp, return_tensors="pt", truncation=True, max_length=256).input_ids
            out = mdl.generate(
                ids,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
                max_new_tokens=120,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
            )
            text = tok.decode(out[0], skip_special_tokens=True)
            return sanitize_upper_space_len(text)
        except Exception:
            pass

    # --- Fallback: varied template generator (still natural-ish) ---
    actions = [
        "TAKE TEN SLOW BREATHS AND NAME WHAT YOU FEEL",
        "WRITE ONE CLEAR SENTENCE ABOUT YOUR NEEDS",
        "PLAN A TINY STEP YOU CAN DO TODAY",
        "RELAX YOUR JAW NECK AND SHOULDERS",
        "STEP OUTSIDE FOR TWO MINUTES OF AIR",
        "SILENCE NOTIFICATIONS FOR A SHORT WHILE",
    ]
    reframes = [
        "PROGRESS HAPPENS IN SMALL HONEST MOVES",
        "YOUR FEELINGS MAKE SENSE AND WILL EASE",
        "YOU CAN ASK FOR HELP BEFORE IT FEELS URGENT",
        "REST IS PART OF HEALING NOT FAILURE",
        "CURIOSITY BEATS SELF JUDGMENT",
    ]
    supports = [
        "MESSAGE A TRUSTED PERSON FOR GENTLE SUPPORT",
        "SCHEDULE A BRIEF CHECK IN WITH YOUR THERAPIST",
        "SET ONE KIND BOUNDARY AND KEEP IT SIMPLE",
        "DRINK WATER AND EAT SOMETHING STEADY",
    ]
    cue = re.sub(r"[^A-Za-z ]+", " ", prompt_text).upper().strip()
    cue = "REGARDING " + " ".join(cue.split()[:5]) if cue else "FOCUS ON KIND ACTION"
    parts = [cue, rnd.choice(actions), rnd.choice(reframes), rnd.choice(supports)]
    rnd.shuffle(parts)
    text = " ".join(parts)
    return sanitize_upper_space_len(text)

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

    weekdays = ["MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY","SUNDAY"]
    weekday = st.selectbox("Select weekday", weekdays, index=0)
    case_text = st.text_area("Your message (case details)")

    if st.button("Send to Psychologist"):
        if not case_text.strip():
            st.warning("Please enter your message.")
            return

        suggestion_plain = tiny_llm_generate(case_text)
        key = WEEKDAY_KEY[weekday]
        suggestion_cipher = caesar_encrypt(suggestion_plain, key)

        cur = conn.cursor()
        cur.execute("""
            INSERT INTO messages(user_id, username, weekday, user_message, llm_suggestion_plain, suggestion_encrypted, created_at)
            VALUES(?,?,?,?,?,?,?)
        """, (
            current_user["id"], current_user["username"], weekday, case_text.strip(),
            suggestion_plain, suggestion_cipher, datetime.utcnow().isoformat()
        ))
        conn.commit()

        st.success("Encrypted suggestion received.")
        st.code(suggestion_cipher, language="text")  # encrypted only

    st.divider()
    st.subheader("Your History")
    cur = conn.cursor()
    cur.execute("""
        SELECT weekday, user_message, suggestion_encrypted, created_at
        FROM messages WHERE user_id = ?
        ORDER BY id DESC LIMIT 50
    """, (current_user["id"],))
    rows = cur.fetchall()
    if rows:
        for wd, umsg, enc, ts in rows:
            st.markdown(f"**{wd.title()}** — {ts}")
            st.write(umsg)
            st.code(enc, language="text")
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
            SELECT username, weekday, user_message, llm_suggestion_plain, suggestion_encrypted, created_at
            FROM messages WHERE username = ? ORDER BY id DESC
        """, (who,))
    else:
        cur.execute("""
            SELECT username, weekday, user_message, llm_suggestion_plain, suggestion_encrypted, created_at
            FROM messages ORDER BY id DESC
        """)

    rows = cur.fetchall()
    if not rows:
        st.info("No messages yet.")
        return

    for uname, wd, msg, plain, enc, ts in rows:
        st.markdown(f"**{uname}** • {wd.title()} • {ts}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**User Message**")
            st.write(msg)
            st.markdown("**Encrypted Sent to User**")
            st.code(enc, language="text")
        with c2:
            st.markdown("**Suggestion (Plaintext for Admin)**")
            st.code(plain, language="text")
        st.markdown("---")

# -----------------------------
# APP
# -----------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="centered")
    st.title(APP_TITLE)
    st.write(
        "This demo generates a SHORT, PRACTICAL SUGGESTION with a tiny local model, "
        "sanitizes it to UPPERCASE AND SPACES, constrains it to 150–200 CHARS, then "
        "encrypts it per weekday before returning to the user."
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
