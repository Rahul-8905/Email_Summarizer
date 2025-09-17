# Data Science Report

## 1. Dataset Description
- Source: Kaggle email dataset (email-thread-summary-dataset)
- Already Processed Dataset 
- Size: 21,684 emails.

---

## 2. Fine-Tuning Setup
- **Base model**: FLAN-T5 small  
- **Method**: LoRA (Low-Rank Adaptation)  
- **Parameters**:
  - r = 8  
  - lora_alpha = 16  
  - target_modules = ["q", "k", "v"]  
  - dropout = 0.05  
- **Training**:
  - Batch size = 16  
  - Optimizer = AdamW (default in Hugging Face Trainer)  
  - Epochs = 3  
  - Learning rate = 2e-4

---

## 3. Results

- **Evaluation Metrics**:
  -Steps vs Training Loss values have been uploaded 

- **Qualitative Examples**:
  | Email Snippet | Generated Summary |
  |---------------|------------------|
  | Long email body ... | "Meeting rescheduled to 3 PM tomorrow." |

---

## 4. Discussion
- LoRA fine-tuning made the model specialize in **email summarization** without retraining the entire FLAN-T5 model.  
- Compared to base model, summaries are **more concise and context-aware**.  
- Limitations: struggles with very long threads; some loss of nuance.  
- Future work:  
  - Try multi-task training with classification (urgent vs. non-urgent).  
  - Use RAG to add context for older emails.
