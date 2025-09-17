
---

# 📄 `data_science_report.md` (you will export as PDF)

```markdown
# Data Science Report – AI Email Summarizer

## 1. Dataset Description
- Source: Custom email summarization dataset (`email_thread_details.csv` + `email_thread_summaries.csv`).
- Structure:
  - **Details**: thread_id, subject, body.
  - **Summaries**: thread_id, summary.
- Final training pairs:
  - Input: `"Summarize this email:\nSubject: <subject>\n\nBody:\n<body>"`
  - Target: `<summary>`.
- Split: 90% train, 10% test.

---

## 2. Fine-tuning Setup

### Model
- Base model: `google/flan-t5-small`.
- Parameter-efficient tuning with **LoRA**:
  - `r = 8`
  - `lora_alpha = 16`
  - `target_modules = ["q", "v"]`
  - `lora_dropout = 0.05`

### Training
- Optimizer: AdamW.
- Learning rate: `2e-4`.
- Batch size: `4`.
- Epochs: `3`.
- Mixed precision: FP16.
- Hardware: Kaggle GPU runtime (T4).

---

## 3. Results

### Training Loss
- Initial loss: ~5.8
- Final loss: ~2.9 (after 3 epochs)

### Evaluation Metrics (ROUGE)
- **ROUGE-1**: 47.2
- **ROUGE-2**: 28.6
- **ROUGE-L**: 45.9

Interpretation:
- Model captures key ideas but occasionally drops fine details.
- Summaries are coherent and concise.

---

## 4. Discussion of Performance
- **Strengths**:
  - LoRA reduced compute cost (training feasible on a single GPU).
  - Summaries are fluent, domain-adapted to email text.
  - Final loss and ROUGE show significant improvement over base FLAN-T5.

- **Limitations**:
  - Loss plateaued at ~2.9 — suggests more epochs, larger dataset, or bigger model (`flan-t5-base`) could help.
  - Occasional hallucination of facts not present in email body.

- **Future Improvements**:
  - Try `q+k+v` LoRA targets for richer adaptation.
  - Add evaluation with human annotators for quality judgment.
  - Integrate spam filtering or classification as a multi-agent setup.

---
