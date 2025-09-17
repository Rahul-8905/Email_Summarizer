import imaplib
import email
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
from dotenv import load_dotenv
from email.header import decode_header

# 1. Load credentials

env_path = r".env"
load_dotenv(dotenv_path=env_path)

EMAIL_ACCOUNT = os.getenv("EMAIL")
APP_PASSWORD = os.getenv("APPPASSWORD")

#Load Fine-tuned Model

MODEL_PATH = r"fine_tuned_model"  # path where train.py saved the model

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

#Gmail IMAP Login

IMAP_SERVER = "imap.gmail.com"

def decode_subject(subject_raw):
    """Decode encoded email subjects"""
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

        emails.append({
            "subject": subject,
            "from": sender,
            "body": body
        })
    mail.logout()
    return emails

#Summarization

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

#Streamlit UI

def main():
    st.title("AI Mail Assistant (Fine-tuned)")
    st.info("Fetching your latest emails and summarizing them with your fine-tuned FLAN-T5 model.")

    emails = fetch_emails(5)

    for i, mail in enumerate(emails):
        st.subheader(f"Email {i+1}: {mail['subject']}")
        st.write(f"**From:** {mail['from']}")
        summary = summarize(mail["body"])
        st.success(summary)

main()
