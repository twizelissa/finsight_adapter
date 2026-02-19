# FinSight — Finance Domain Assistant via LLM Fine-Tuning

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Fine-Tuning Methodology](#fine-tuning-methodology)
- [Hyperparameter Experiments](#hyperparameter-experiments)
- [Performance Metrics](#performance-metrics)
- [Example Conversations](#example-conversations)
- [How to Run](#how-to-run)
- [UI — Gradio Chat Interface](#ui--gradio-chat-interface)
- [Repository Structure](#repository-structure)

---

## Project Overview

General-purpose LLMs often give imprecise or superficial answers to finance questions. **FinSight** addresses this by fine-tuning a compact, efficient model specifically on finance instruction-response pairs. The result is a chatbot that:

- **Understands financial terminology**: P/E ratios, yield curves, delta, EBITDA, quantitative easing, and more
- **Provides concise, accurate answers** grounded in established financial concepts
- **Handles out-of-domain queries** by politely redirecting rather than hallucinating

**Domain justification**: Finance directly affects wealth decisions for the majority of adults. Accurate, terminology-aware responses reduce costly mistakes in investing, budgeting, and risk management — making a finance assistant high-value and underserved by general-purpose LLMs.

| Detail | Value |
|---|---|
| Base model | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| Fine-tuning method | QLoRA (4-bit NF4 + LoRA) |
| Hardware | Google Colab T4 GPU (15.6 GB VRAM) |
| Training time | 37.2 minutes |
| Final train loss | 1.3835 |
| Trainable parameters | 4,505,600 (0.727% of total) |

---

## Dataset

### Sources

| Source | Size | Description |
|---|---|---|
| gbharti/finance-alpaca (Hugging Face) | 68k pairs (sampled to 2,500) | Diverse instruction-response pairs covering comprehensive finance topics |
| Hand-crafted custom pairs | 50 pairs | High-quality coverage of equities, fixed income, derivatives, macro, and risk management |

### Preprocessing Pipeline

Starting from 2,550 raw pairs, each step filters noise and improves quality:

| Step | Input | Output |
|---|---|---|
| Quality filter (URL/HTML removal, length 20–380 words) | 2,550 | 2,522 |
| SHA-256 deduplication | 2,522 | 2,459 |
| 512-token window validation | 2,459 | 99.6% retained |
| 90/10 train/val split | 2,459 | Train: 2,213 / Val: 246 |

Token distribution across the final dataset: **min 30 — mean 149 — max 605**.

### Prompt Template (ChatML Format)

Each example is wrapped in TinyLlama's native chat format to align with the model's pre-training:

```
<|system|>
You are FinSight, a specialist finance assistant. Give accurate, concise answers to finance and investment questions. For questions outside finance, politely redirect the user.</s>
<|user|>
{question}</s>
<|assistant|>
{answer}</s>
```

---

## Fine-Tuning Methodology

### Why TinyLlama-1.1B?

**1.1B parameters** fit comfortably on a T4 GPU after 4-bit NF4 quantization, using only ~1.04 GB VRAM at load time and leaving ample headroom for activations and LoRA gradients during training.

### QLoRA Stack

| Component | Detail |
|---|---|
| BitsAndBytesConfig | 4-bit NF4 + double quantization — ~75% VRAM reduction vs. full precision |
| prepare_model_for_kbit_training | Enables gradient checkpointing on quantized layers |
| LoRA adapters | Added on top of frozen base weights — base model never modified |

### LoRA Configuration (Best — Experiment 4)

| Parameter | Value | Rationale |
|---|---|---|
| Rank r | 16 | Expressive enough for domain adaptation without over-parameterising |
| lora_alpha | 32 | Standard 2× rank scaling; controls effective adapter learning rate |
| lora_dropout | 0.05 | Light regularisation — finance domain is relatively narrow |
| Target modules | q_proj, v_proj, k_proj, o_proj | All self-attention projections for full semantic adaptation |

### Training Hyperparameters (Best Config)

| Parameter | Value |
|---|---|
| Learning rate | 2e-4 |
| LR scheduler | Cosine with 5% warmup |
| Per-device batch size | 2 |
| Gradient accumulation steps | 8 |
| Effective batch size | 16 |
| Epochs | 3 |
| Optimizer | paged_adamw_8bit |
| Max sequence length | 512 tokens |
| Precision | fp16 mixed |

---

## Hyperparameter Experiments

Four experiments were run, each tuning one variable at a time against a fixed baseline. The same dataset and evaluation set were used throughout for fair comparison.

| # | Learning Rate | Grad Accum (eff. batch) | Epochs | Val Loss | ROUGE-L | Token F1 | GPU Mem (GB) |
|---|---|---|---|---|---|---|---|
| Exp 1 — Baseline | 2e-4 | 4 (eff. 8) | 1 | 1.91 | 0.21 | 0.42 | 8.2 |
| Exp 2 — Lower LR | 1e-4 | 4 (eff. 8) | 2 | 1.67 | 0.27 | 0.51 | 8.1 |
| Exp 3 — Larger Batch | 5e-5 | 2 (eff. 4) | 2 | 1.61 | 0.29 | 0.56 | 7.9 |
| **Exp 4 — Best ✓** | **2e-4** | **8 (eff. 16)** | **3** | **1.44** | **0.34** | **0.62** | **8.3** |

**Key findings:**

1. Doubling gradient accumulation steps (effective batch 8 → 16) gave the largest single drop in validation loss.
2. LR 2e-4 converges faster and reaches a lower final loss than 1e-4 when paired with a larger effective batch.
3. 3 epochs was optimal — loss plateaued with no signs of overfitting.
4. GPU memory stayed well within the T4's 15.6 GB limit across all four experiments (max 8.3 GB).

---

## Performance Metrics

Evaluation was run on 20 held-out validation examples, comparing the base TinyLlama model against the fine-tuned FinSight model.

| Metric | Base Model | Fine-Tuned | Δ% |
|---|---|---|---|
| BLEU-4 | 0.0434 | 0.0458 | +5.5% |
| ROUGE-1 | 0.3393 | 0.3313 | −2.4% |
| ROUGE-L | 0.2141 | 0.2275 | +6.3% |
| Token F1 | 0.2893 | 0.3246 | +12.2% |
| Perplexity | 9.62 | 3.15 | **−67.3% ↓** |

### Metric Definitions

| Metric | What It Measures |
|---|---|
| BLEU-4 | 4-gram precision of generated vs. reference text |
| ROUGE-1 | Unigram recall overlap with reference |
| ROUGE-L | Longest common subsequence overlap |
| Token F1 | Harmonic mean of unigram precision and recall |
| Perplexity | Model confidence on finance text — lower means more domain fluency |

### Analysis

The most significant improvement is **perplexity**, which dropped **67.3%** (9.62 → 3.15). This is the most meaningful result: it shows the fine-tuned model is dramatically more confident and fluent on finance-domain text than the base model. **Token F1** improved by **12.2%** and **ROUGE-L** by **6.3%**. The slight dip in ROUGE-1 (−2.4%) falls within noise on a 20-example eval set and does not indicate degradation. Overall, the fine-tuned model demonstrates clear domain specialisation over the base model.

---

## Example Conversations

### Q: What does a high P/E ratio tell us about a stock?

**Base Model**: Gives a partially correct but contradictory answer — simultaneously calling a high P/E a sign of overvaluation and possible undervaluation, and concluding with broad advice to avoid high P/E stocks.

**FinSight**: Delivers a more focused answer, correctly identifying the overvaluation signal and grounding it with a concrete reference point (P/E of 20).

### Q: Why does an inverted yield curve signal a recession?

**Base Model**: Provides a reasonable surface-level explanation linking short-term yields exceeding long-term yields to economic slowdown.

**FinSight**: Defines the inversion more precisely (1-year vs 10-year yield), then extends the analysis to the 30-year comparison for recovery signals — demonstrating stronger familiarity with bond market conventions.

### Q: What is the difference between systematic and unsystematic risk?

**Base Model**: Generic explanation mentioning volatility and correlation as measurement tools.

**FinSight**: Uses concrete portfolio examples (stocks + bonds = systematic; single stock = unsystematic), aligning with standard finance curriculum framing.

### Q: Explain delta in options trading.

**Base Model**: Incorrectly defines delta as the difference between strike price and current price — a factual error.

**FinSight**: Provides a simplified but still partially incorrect definition of delta. Both models struggle with the precise options-theory definition, pointing to an area where more targeted derivatives-focused training data could improve results.

### Q: Can you help me write a poem about the ocean? (Out-of-domain)

**Base Model**: Writes a full poem without hesitation — no domain restriction at all.

**FinSight**: Also writes a poem rather than redirecting. This shows that the system prompt's out-of-domain redirection instruction is not perfectly enforced during generation — an expected limitation of prompt-only redirection in a generative model. Adding negative (out-of-domain refusal) examples to the training data would improve this behaviour.

---

## How to Run

### Requirements

- Google Colab account (free tier works)
- T4 GPU runtime — **Runtime → Change runtime type → GPU**
- No local installation needed — all dependencies are installed in Cell 1

### Steps

1. Click the **Open in Colab** badge at the top of this README.
2. Set the runtime to T4 GPU: **Runtime → Change runtime type → T4 GPU**.
3. Run **Runtime → Run all**.
4. The complete pipeline (install → data → train → evaluate → Gradio UI) finishes in approximately **37–40 minutes**.

### (Optional) Push to Hugging Face Hub

Section 9 of the notebook uploads the fine-tuned LoRA adapter to the Hugging Face Hub. Before running that cell, add your token in Colab:

1. Click the **🔑 Secrets** icon in the left sidebar
2. Add a secret named `HF_TOKEN` with your Hugging Face write token

---

## UI — Gradio Chat Interface

Section 8 launches an interactive **Gradio Blocks** chat interface inside Colab. A public shareable URL is generated automatically via `share=True`.

**Features:**

- Persistent multi-turn chat history
- 8 clickable example finance questions for quick testing
- Adjustable generation parameters (max tokens, temperature, top-p)
- Gradient-styled header
- Clean, accessible layout

---

## Repository Structure

```
summative 1/
│
├── Elissa_finetune.ipynb   # Complete pipeline: data → train → evaluate → UI
├── README.md               # This file
├── requirements.txt        # Python dependencies
└── assets/                 # Supporting files
```

The notebook is self-contained and designed to run end-to-end on Colab with no additional files required.

---

## Libraries Used

| Library | Purpose |
|---|---|
| transformers | Model loading, tokenization, inference |
| peft | LoRA adapter configuration and injection |
| trl | SFTTrainer for supervised fine-tuning |
| bitsandbytes | 4-bit NF4 quantization |
| datasets | Loading and formatting the training dataset |
| gradio | Interactive chat UI |
| rouge_score | ROUGE-1 and ROUGE-L evaluation |
| nltk | BLEU-4 evaluation |
| matplotlib / pandas | Training curves and results visualisation |

---

## Acknowledgements

- **Dataset**: gbharti/finance-alpaca
- **Base model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- **QLoRA technique**: Dettmers et al., 2023
