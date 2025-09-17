# Email_Summarizer
A Streamlit-based AI assistant that connects to Gmail via IMAP, fetches recent emails, and generates concise summaries using a fine-tuned FLAN-T5 model with parameter-efficient LoRA training.

# AI Email Summarizer – Architecture Document

## Components

1. **Gmail Fetcher**
   - Uses IMAP to connect to Gmail.
   - Authenticates securely with `.env` credentials (EMAIL, APPPASSWORD).
   - Retrieves the last N emails (subject, sender, body).
   - Outputs structured JSON for downstream tasks.

2. **Fine-tuned Summarizer**
   - Backbone: `google/flan-t5-small`.
   - Fine-tuned using LoRA (Low-Rank Adaptation) on an email summarization dataset.
   - Input: Email subject + body.
   - Output: Concise human-readable summary.
   - Hosted locally inside the Streamlit app.

3. **Streamlit User Interface**
   - Simple web-based UI for interaction.
   - Displays fetched emails with subject + sender.
   - Provides a "Summarize" button for each email.
   - Shows generated summaries inline.
