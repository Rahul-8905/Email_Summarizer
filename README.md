# Email_Summarizer
A Streamlit-based AI assistant that connects to Gmail via IMAP, fetches recent emails, and generates concise summaries using a fine-tuned FLAN-T5 model with parameter-efficient LoRA training.

**Author:** Boddepalli Rahul  
**Roll No.:** ME23B162 
**Department:** Mechanical Engineering  
**Institute:** Indian Institute of Technology, Madras 


# AI Email Summarizer - Architecture Document

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
### Why FLAN-T5 and LoRA?

**FLAN-T5** was chosen as the backbone because it is a strong text-to-text model that can handle a variety of NLP tasks, including summarization. Its instruction-tuned design makes it especially good at following commands like "Summarize this email," producing concise and readable summaries without extensive task-specific training.

**LoRA (Low-Rank Adaptation)** was used for fine-tuning to efficiently adapt the model to the email summarization task. Instead of updating all of FLAN-T5's parameters-which would be computationally expensive-LoRA injects a small number of trainable parameters into the attention layers. This allows the model to **specialize in summarizing emails** while keeping training fast, memory-efficient, and stable.  

3. **Streamlit User Interface**
   - Simple web-based UI for interaction.
   - Displays fetched emails with subject + sender.
   - Shows generated summaries inline.
# Flow Diagram
```mermaid
flowchart TD
    A["User"] -->|Login with Gmail| B["Gmail Fetcher"]
    B -->|Email JSON| C["Summarizer: Fine-tuned FLAN-T5 with LoRA"]
    C -->|Summary| D["Streamlit UI"]
    D -->|Display Summaries| A
