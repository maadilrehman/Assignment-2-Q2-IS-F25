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
APP_TITLE = "PsyCare (Secure Suggestion Demo)"
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
        a_param INTEGER NOT NULL,
        b_param INTEGER NOT NULL,
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
    if not row:
        return None
    return {"id": row[0], "username": row[1], "password_hash": row[2], "role": row[3]}

# -----------------------------
# CRYPTO (AFFINE)
# -----------------------------
def next_coprime_after(b: int, modulus: int = 26) -> int:
    """
    Return the smallest integer a such that a > b and gcd(a, modulus) == 1.
    For modulus 26, valid a are odd and not 13 mod 26.
    """
    a = b + 1
    while gcd(a, modulus) != 1:
        a += 1
    return a

def affine_encrypt_char(ch: str, a: int, b: int) -> str:
    """
    Affine encryption of single uppercase char: E(x) = (a*x + b) mod 26
    Spaces pass through; all other chars should have been stripped prior.
    """
    if ch == " ":
        return " "
    if ch in ALPHABET:
        x = ord(ch) - ord('A')
        y = (a * x + b) % 26
        return chr(y + ord('A'))
    return ""

def affine_encrypt(text: str, a: int, b: int) -> str:
    text = text.upper()
    return "".join(affine_encrypt_char(c, a, b) for c in text if c == " " or c in ALPHABET)

# -----------------------------
# SMALL LOCAL "LLM" + SANITIZER
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_small_llm():
    """
    Load a tiny instruction-tuned model (CPU OK). Cached across reruns.
    If loading fails, caller falls back to a template generator.
    """
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        model_name = os.getenv("SMALL_LLM_NAME", "google/flan-t5-small")
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return tok, mdl
    except Exception:
        return None, None

def sanitize_upper_space_len(text: str) -> str:
    """
    Enforce: only A–Z and spaces, length between 500–600 characters (inclusive).
    Pads or trims with neutral supportive phrases to stay within bounds.
    """
    text = text.upper()
    text = re.sub(r"[^A-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    MAX_LEN, MIN_LEN = 600, 500
    if len(text) > MAX_LEN:
        text = text[:MAX_LEN].rstrip()

    if len(text) < MIN_LEN:
        pad_bank = [
            " BREATHE IN BREATHE OUT NOTICE YOUR BODY AND SOFTEN YOUR SHOULDERS ",
            " YOU ARE SAFE RIGHT NOW CHOOSE ONE KIND SMALL STEP AND HONOR YOUR LIMITS ",
            " PROGRESS HAPPENS IN QUIET MOMENTS KEEP YOUR PACE STEADY AND COMPASSIONATE ",
            " RETURN TO YOUR BREATH AND NAME WHAT YOU FEEL WITHOUT JUDGMENT THEN LET IT PASS "
        ]
        i = 0
        while len(text) < MIN_LEN:
            text += pad_bank[i % len(pad_bank)]
            text = re.sub(r"\s+", " ", text).strip()
            i += 1
        if len(text) > MAX_LEN:
            text = text[:MAX_LEN].rstrip()
    return text

def tiny_llm_generate(prompt_text: str) -> str:
    """
    Try flan-t5-small to generate a supportive response aimed at ~550 chars (pre-sanitize).
    Falls back to a varied template if model unavailable.
    """
    tok, mdl = load_small_llm()
    seed = int.from_bytes(secrets.token_bytes(4), "big")
    rnd = random.Random(seed)

    sys_hint = (
        "You are a concise psychologist. Write a supportive and practical suggestion "
        "for the case below in 5–8 short sentences, about 550 characters total. "
        "Avoid lists, keep language simple, warm, and actionable."
    )
    user_case = re.sub(r"\s+", " ", prompt_text).strip()
    inp = f"{sys_hint}\nCASE: {user_case}"

    if tok and mdl:
        try:
            ids = tok(inp, return_tensors="pt", truncation=True, max_length=640).input_ids
            out = mdl.generate(
                ids,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
                max_new_tokens=420,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
            )
            text = tok.decode(out[0], skip_special_tokens=True)
            return sanitize_upper_space_len(text)
        except Exception:
            pass  # fallback below

    # Fallback: natural-ish varied template (uppercase + spaces enforced later)
    actions = [
        "TAKE TEN SLOW BREATHS WHILE YOU RELAX YOUR JAW AND SHOULDERS",
        "WRITE ONE SENTENCE ABOUT WHAT YOU NEED AND ONE TINY STEP YOU CAN TAKE",
        "DRINK WATER AND EAT SOMETHING STEADY TO SUPPORT YOUR ENERGY",
        "WALK OUTSIDE BRIEFLY AND COUNT YOUR STEPS TO GROUND YOUR ATTENTION",
        "SILENCE NOTIFICATIONS FOR A SHORT WINDOW TO CREATE QUIET SPACE",
        "PLACE A HAND ON YOUR CHEST AND MATCH YOUR INHALE AND EXHALE GENTLY",
        "NOTICE THREE THINGS YOU SEE AND TWO THINGS YOU HEAR AND ONE THING YOU FEEL"
    ]
    reframes = [
        "FEELINGS SURGE AND FADE YOU CAN RIDE THE WAVE SAFELY",
        "REST IS PART OF HEALING NOT A FAILURE OF WILLPOWER",
        "SMALL CONSISTENT ACTIONS BEAT PERFECT PLANS THAT NEVER START",
        "YOUR EXPERIENCE MAKES SENSE GIVEN YOUR CONTEXT SHOW YOURSELF KINDNESS",
        "YOU CAN ASK FOR HELP EARLY BEFORE IT FEELS OVERWHELMING",
        "PROGRESS ARRIVES THROUGH PATIENCE AND GENTLE PRACTICE EACH DAY"
    ]
    supports = [
        "MESSAGE A TRUSTED PERSON FOR A BRIEF CHECK IN",
        "NOTE ONE QUESTION TO BRING TO YOUR NEXT SESSION",
        "SET A SIMPLE BOUNDARY USING CLEAR AND KIND WORDS",
        "CREATE A CALM EVENING ROUTINE AND PROTECT YOUR WIND DOWN TIME",
        "PLAN A SHORT WALK OR STRETCH TO LOOSEN TENSION"
    ]
    cue = re.sub(r"[^A-Za-z ]+", " ", prompt_text).upper().strip()
    cue = "REGARDING " + " ".join(cue.split()[:12]) if cue else "FOCUS ON STEADY KIND ACTIONS"

    paragraphs = [
        f"{cue} {rnd.choice(actions)} {rnd.choice(reframes)}",
        f"{rnd.choice(actions)} {rnd.choice(supports)} {rnd.choice(reframes)}",
        f"{rnd.choice(supports)} {rnd.choice(actions)} {rnd.choice(reframes)}"
    ]
    rnd.shuffle(paragraphs)
    text = " ".join(paragraphs)
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

    weekdays = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]
    weekday = st.selectbox("Select weekday", weekdays, index=0)
    case_text = st.text_area("Your message (case details)")

    if st.button("Send to Psychologist"):
        if not case_text.strip():
            st.warning("Please enter your message.")
            return

        # Generate plaintext suggestion
        suggestion_plain = tiny_llm_generate(case_text)

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
        st.code(wrapped, language="text")  # encrypted only

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
        with c2:
            st.markdown("**Suggestion (Plaintext for Admin)**")
            wrapped_plain = "\n".join(plain[i:i+80] for i in range(0, len(plain), 80))
            st.code(wrapped_plain, language="text")
            st.caption(f"AFFINE PARAMS USED → a = {a}, b = {b}")
        st.markdown("---")

# -----------------------------
# APP
# -----------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="centered")
    st.title(APP_TITLE)
    st.write(
        "This demo generates a PRACTICAL SUGGESTION with a tiny local model, "
        "sanitizes it to UPPERCASE AND SPACES, constrains it to 500–600 CHARS, then "
        "encrypts it using an AFFINE CIPHER where b depends on the weekday and a is the next coprime greater than b."
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
