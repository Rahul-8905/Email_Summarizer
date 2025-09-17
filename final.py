import imaplib
import email
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
from dotenv import load_dotenv
from email.header import decode_header
from email.utils import parsedate_to_datetime
import html


# Load credentials
env_path = r".env"
load_dotenv(dotenv_path=env_path)
EMAIL_ACCOUNT = os.getenv("EMAIL")
APP_PASSWORD = os.getenv("APPPASSWORD")


# Load Fine-tuned Model
MODEL_PATH = r"fine_tuned_model"  # path where train.py saved the model


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device


tokenizer, model, device = load_model()


# Gmail IMAP Login
IMAP_SERVER = "imap.gmail.com"


def decode_header_value(raw_value):
    """Helper to decode any MIME encoded header value safely."""
    if raw_value:
        parts = decode_header(raw_value)
        decoded_strings = []
        for decoded, encoding in parts:
            if isinstance(decoded, bytes):
                decoded_strings.append(decoded.decode(encoding or "utf-8", errors="ignore"))
            else:
                decoded_strings.append(decoded)
        return ''.join(decoded_strings)
    return "(No Value)"


def decode_subject(subject_raw):
    if subject_raw:
        decoded, encoding = decode_header(subject_raw)[0]
        if isinstance(decoded, bytes):
            return decoded.decode(encoding or "utf-8", errors="ignore")
        return decoded
    return "(No Subject)"


def fetch_emails(n=5):
    mail = imaplib.IMAP4_SSL(IMAP_SERVER)
    mail.login(EMAIL_ACCOUNT, APP_PASSWORD)
    mail.select("inbox")

    result, data = mail.search(None, "ALL")
    email_ids = data[0].split()[-n:]  # last n emails

    emails = []
    for e_id in reversed(email_ids):
        result, msg_data = mail.fetch(e_id, "(RFC822)")
        raw_msg = msg_data[0][1]
        msg = email.message_from_bytes(raw_msg)

        subject = decode_subject(msg["subject"])
        sender = msg["from"]
        body = ""

        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True).decode(errors="ignore")
                    break
        else:
            body = msg.get_payload(decode=True).decode(errors="ignore")

        # Extract and parse Date header
        date_raw = msg["Date"]
        try:
            dt = parsedate_to_datetime(date_raw)
            date_decoded = dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            date_decoded = decode_header_value(date_raw)

        emails.append({
            "subject": subject,
            "from": sender,
            "body": body,
            "date": date_decoded
        })
    mail.logout()
    return emails


# Summarization
def summarize(text, max_len=200):
    inputs = tokenizer(
        "Summarize this email:\n" + text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    summary_ids = model.generate(
        inputs["input_ids"],
        max_new_tokens=max_len,
        num_beams=6,
        length_penalty=2.0,
        no_repeat_ngram_size=3,
        repetition_penalty=2.0,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# Streamlit UI
def main():
    st.set_page_config(page_title="Email Summarizer", page_icon="✉️", layout="centered")

    # Custom CSS for aesthetic improvements
    st.markdown(
        """
        <style>
        /* Background and fonts */
        .main {
            background-color: #f9fbfc;
            color: #333333;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        /* Title styling */
        .title {
            font-weight: 700;
            color: #0a3d62;
            margin-bottom: 0;
        }
        /* Email container */
        .email-container {
            background: white;
            border: 1px solid #d1d8e0;
            border-radius: 8px;
            padding: 20px 25px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            transition: box-shadow 0.3s ease;
        }
        .email-container:hover {
            box-shadow: 0 5px 15px rgba(0,0,0,0.15);
        }
        /* Subject style */
        .subject {
            font-size: 1.2rem;
            font-weight: 600;
            color: #1e3799;
            margin-bottom: 6px;
        }
        /* Sender style */
        .sender {
            font-size: 0.9rem;
            font-style: italic;
            color: #57606f;
            margin-bottom: 12px;
        }
        /* Summary style */
        .summary {
            font-size: 1rem;
            color: #30336b;
            line-height: 1.5;
            white-space: pre-wrap;
        }
        /* Info box */
        .stAlert {
            background-color: #dff9fb !important;
            border-left: 6px solid #22a6b3 !important;
            color: #1e272e !important;
            font-size: 1rem !important;
            margin-bottom: 40px !important;
        }
        </style>
        """, unsafe_allow_html=True
    )

    st.markdown('<h1 class="title">Email Summarizer</h1>', unsafe_allow_html=True)
    st.info("Fetching your latest emails and summarizing them.")

    with st.spinner("Loading emails and generating summaries..."):
        emails = fetch_emails(5)

    for i, mail in enumerate(emails):
        summary = summarize(mail["body"])
        subject_escaped = html.escape(mail["subject"])
        from_escaped = html.escape(mail["from"])
        summary_escaped = html.escape(summary)
        date_escaped = html.escape(mail.get("date", "Unknown Date"))

        html_content = f'''
        <div class="email-container">
            <div class="subject">Received: {date_escaped}</div>
            <p><strong>Subject:</strong> {subject_escaped}</p>
            <p><strong>From:</strong> {from_escaped}</p>
            <p><strong>Summary:</strong> {summary_escaped}</p>
        </div>
        '''
        st.markdown(html_content, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
