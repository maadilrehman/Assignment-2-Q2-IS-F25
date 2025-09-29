import os
import re
import sqlite3
from datetime import datetime
import streamlit as st
import bcrypt

# -----------------------------
# CONFIG
# -----------------------------
APP_TITLE = "PsyCare (Your personal End-to-End Secure Suggestion System)"
DB_PATH = "psycare.db"

# Caesar shift per weekday (step of 2)
WEEKDAY_KEY = {
    "MONDAY": 2,
    "TUESDAY": 4,
    "WEDNESDAY": 6,
    "THURSDAY": 8,
    "FRIDAY": 10,
    "SATURDAY": 12,
    "SUNDAY": 14,
}

# Admin bootstrap (you can override via Streamlit secrets or env)
ADMIN_USERNAME = os.getenv("PSY_ADMIN_USER", "psychologist")
ADMIN_PASSWORD = os.getenv("PSY_ADMIN_PASS", "admin123")  # will be hashed on first run


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
    )
    """)
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
    )
    """)
    conn.commit()
    return conn

def bootstrap_admin(conn):
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE username = ?", (ADMIN_USERNAME,))
    if not cur.fetchone():
        pw_hash = bcrypt.hashpw(ADMIN_PASSWORD.encode(), bcrypt.gensalt())
        cur.execute("INSERT INTO users(username, password_hash, role) VALUES(?,?,?)",
                    (ADMIN_USERNAME, pw_hash, "ADMIN"))
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
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def caesar_shift_char(ch, k):
    if ch == " ":
        return " "
    if ch in ALPHABET:
        idx = (ALPHABET.index(ch) + k) % 26
        return ALPHABET[idx]
    # non A–Z gets stripped from the LLM suggestion before encryption; safe-guard:
    return ""

def caesar_encrypt(text, k):
    text = text.upper()
    return "".join(caesar_shift_char(c, k) for c in text)

def caesar_decrypt(text, k):
    return caesar_encrypt(text, -k)


# -----------------------------
# LLM SUGGESTION (150–200 chars, UPPERCASE + SPACES ONLY)
# -----------------------------
def llm_generate_suggestion(prompt_text: str) -> str:
    """
    Tries OpenAI if OPENAI_API_KEY is set; otherwise falls back to a deterministic, safe local generator.
    Ensures: 150–200 chars, only [A–Z ].
    """
    suggestion = None

    # --- Optional: OpenAI path (safe to deploy without it) ---
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if api_key:
        try:
            # Lazy import to avoid hard dependency if not used
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            sys = ("YOU ARE A CONCISE, EMPATHIC PSYCHOLOGIST. "
                   "WRITE A 150 TO 200 CHARACTER PRACTICAL SUGGESTION IN UPPERCASE AND SPACES ONLY. "
                   "NO PUNCTUATION OR DIGITS.")
            user = f"CASE DETAILS: {prompt_text}"
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": sys},
                          {"role": "user", "content": user}],
                temperature=0.4,
                max_tokens=150
            )
            suggestion = resp.choices[0].message.content if resp.choices else ""
        except Exception:
            suggestion = None

    # --- Fallback: local generator ---
    if not suggestion:
        base = (
            "FOCUS ON ONE SMALL STEP TODAY PRACTICE SLOW BREATHING "
            "NOTICE YOUR THOUGHTS WITHOUT JUDGMENT AND WRITE THEM DOWN "
            "REACH OUT TO A TRUSTED FRIEND FOR SUPPORT AND TAKE A SHORT WALK "
        )

        # lightly adapt to input to avoid feeling too canned
        gist = re.sub(r"[^A-Za-z ]+", " ", prompt_text).upper()
        gist = re.sub(r"\s+", " ", gist).strip()
        # prepend a tailored cue if we have a hint
        if len(gist) > 0:
            cue = f"REGARDING {gist[:30]} TRY GENTLE COPING "
        else:
            cue = "TRY GENTLE COPING "

        suggestion = (cue + base)

    # sanitize: UPPERCASE + SPACE, 150–200 chars
    suggestion = suggestion.upper()
    suggestion = re.sub(r"[^A-Z ]", " ", suggestion)
    suggestion = re.sub(r"\s+", " ", suggestion).strip()
    if len(suggestion) < 150:
        # pad with a neutral mantra until >=150
        pad = " BREATHE IN BREATHE OUT "
        while len(suggestion) < 150:
            suggestion += pad
            suggestion = suggestion[:210]
    # trim to ≤200 while keeping words intact
    if len(suggestion) > 200:
        suggestion = suggestion[:200].rstrip()
    return suggestion


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
    st.caption("Share your case details with the psychologist and receive an encrypted suggestion.")

    weekdays = ["MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY","SUNDAY"]
    weekday = st.selectbox("Select weekday", weekdays, index=0)
    case_text = st.text_area("Your message (case details)")

    if st.button("Send to Psychologist"):
        if not case_text.strip():
            st.warning("Please enter your message.")
            return
        # Generate plaintext suggestion via LLM, sanitize, then Caesar-encrypt using weekday key.
        suggestion_plain = llm_generate_suggestion(case_text)
        key = WEEKDAY_KEY[weekday]
        suggestion_cipher = caesar_encrypt(suggestion_plain, key)

        # save record
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO messages(user_id, username, weekday, user_message, llm_suggestion_plain, suggestion_encrypted, created_at)
            VALUES(?,?,?,?,?,?,?)
        """, (
            current_user["id"], current_user["username"], weekday, case_text.strip(),
            suggestion_plain, suggestion_cipher, datetime.utcnow().isoformat()
        ))
        conn.commit()

        st.success("Suggestion received (encrypted). Share it safely or keep for your records.")
        st.code(suggestion_cipher, language="text")

        # Optional local self-check (not returned to admin): allow user to decrypt to verify correctness
        #with st.expander("Decrypt locally to verify (client-side)"):
        #    st.caption(f"Uses the weekday key = {key}")
        #    st.code(caesar_decrypt(suggestion_cipher, key), language="text")

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
    st.caption("View incoming cases and plaintext suggestions (the user receives only the encrypted version).")

    # simple filter
    who = st.text_input("Filter by username (optional)").strip()
    cur = conn.cursor()
    if who:
        cur.execute("""
            SELECT username, weekday, user_message, llm_suggestion_plain, suggestion_encrypted, created_at
            FROM messages
            WHERE username = ?
            ORDER BY id DESC
        """, (who,))
    else:
        cur.execute("""
            SELECT username, weekday, user_message, llm_suggestion_plain, suggestion_encrypted, created_at
            FROM messages
            ORDER BY id DESC
        """)
    rows = cur.fetchall()
    if not rows:
        st.info("No messages yet.")
        return

    for uname, wd, msg, plain, enc, ts in rows:
        st.markdown(f"**{uname}** • {wd.title()} • {ts}")
        cols = st.columns(2)
        with cols[0]:
            st.markdown("**User Message**")
            st.write(msg)
            st.markdown("**Encrypted Returned to User**")
            st.code(enc, language="text")
        with cols[1]:
            st.markdown("**LLM Suggestion (Plaintext)**")
            st.code(plain, language="text")
            # quick check decrypt == plain
            key = WEEKDAY_KEY[wd]
            dec = caesar_decrypt(enc, key)
            ok = "✅" if dec == plain else "❗"
            st.caption(f"Decrypt check with key {key}: {ok}")
        st.markdown("---")


# -----------------------------
# APP
# -----------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="centered")
    st.title(APP_TITLE)
    st.write("This system routes user messages to a psychologist agent, generates a suggestion"
             "and returns it back to the user, due to some bug in the system the cipher text is not decrypted for the user, we are sorry for the inconvenience, It will be resovled soon.")

    conn = get_db()
    bootstrap_admin(conn)

    # Session
    if "user" not in st.session_state:
        st.session_state.user = None

    # Auth UI
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

        st.divider()
       

    # Main panels
    if not st.session_state.user:
        st.info("Please log in or create an account to continue.")
        return

    # Role-based portals
    if st.session_state.user["role"] == "ADMIN":
        render_admin_portal(conn, st.session_state.user)
    else:
        render_user_portal(conn, st.session_state.user)


if __name__ == "__main__":
    main()
