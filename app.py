import sys
import types

# ── Python 3.13 compatibility: mock missing 'audioop' ──────────────────────
audioop_mock = types.ModuleType("audioop")
sys.modules["audioop"] = audioop_mock

# ── Fix HfFolder missing in newer huggingface_hub ──────────────────────────
import huggingface_hub as _hf_hub
if not hasattr(_hf_hub, "HfFolder"):
    class _HfFolder:
        @staticmethod
        def get_token(): return None
    _hf_hub.HfFolder = _HfFolder

# ── Fix gradio_client schema parsing bug ──────────────────────────────────
try:
    import gradio_client.utils as _gc_utils
    _orig_json_schema = _gc_utils._json_schema_to_python_type
    def _safe_json_schema(schema, defs=None):
        if not isinstance(schema, dict):
            return "any"
        return _orig_json_schema(schema, defs)
    _gc_utils._json_schema_to_python_type = _safe_json_schema
except Exception:
    pass

# ── Main imports ───────────────────────────────────────────────────────────
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ── Config ─────────────────────────────────────────────────────────────────
BASE_MODEL   = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_WEIGHTS = "twizelissa/finsight_adapter"
device       = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load model ─────────────────────────────────────────────────────────────
print("Loading tokenizer…")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print("Loading base model…")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.float32,
    trust_remote_code=True,
)

print("Loading LoRA adapters…")
model = PeftModel.from_pretrained(base_model, LORA_WEIGHTS)
model.eval()
print("Model ready!")

# ── Generation helper ──────────────────────────────────────────────────────
def generate_response(question: str, max_new: int = 260, temperature: float = 0.7) -> str:
    prompt = f"### Instruction:\n{question.strip()}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=480).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=temperature,
            top_p=0.92,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### Response:" in full:
        return full.split("### Response:")[-1].strip()
    return full.strip()

# ── Gradio chat function ───────────────────────────────────────────────────
def chat(message: str, history: list):
    if not message.strip():
        return "", history
    reply = generate_response(message.strip())
    history = (history or []) + [
        {"role": "user",      "content": message.strip()},
        {"role": "assistant", "content": reply},
    ]
    return "", history

# ── UI ─────────────────────────────────────────────────────────────────────
EXAMPLES = [
    ["What is a P/E ratio and what does a high value indicate?"],
    ["Explain the yield curve and what an inversion signals."],
    ["What is the difference between growth and value investing?"],
    ["How does quantitative easing affect equity markets?"],
    ["What is delta in options trading?"],
    ["Should I hold bonds when interest rates are rising?"],
    ["What is free cash flow and why does it matter?"],
    ["Explain the Sharpe ratio."],
]

CSS = """
.gradio-container { max-width: 860px; margin: auto; }
footer { display: none !important; }
"""

with gr.Blocks(title="FinSight — Finance AI", css=CSS) as demo:
    gr.HTML("""
        <div style='text-align:center;padding:18px 24px;
                    background:linear-gradient(135deg,#1e3a5f,#2563eb);
                    border-radius:12px;margin-bottom:8px;'>
            <h1 style='color:#fff;margin:0;font-size:1.9em;'>FinSight</h1>
            <p style='color:#bfdbfe;margin:4px 0 0;font-size:.95em;'>
                Finance Domain Assistant &mdash; QLoRA fine-tuned TinyLlama-1.1B
            </p>
        </div>
    """)

    chatbot = gr.Chatbot(
        value=[],
        height=420,
        type="messages",
        show_label=False,
        bubble_full_width=False,
    )
    state = gr.State([])

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask a finance or investment question…",
            show_label=False,
            scale=8,
            lines=2,
        )
        btn = gr.Button("Send", variant="primary", scale=1, min_width=80)

    btn.click(chat, [msg, state], [msg, chatbot])
    msg.submit(chat, [msg, state], [msg, chatbot])

    gr.Examples(examples=EXAMPLES, inputs=msg, label="Try an example", examples_per_page=4)
    gr.Markdown(
        "---\n**Scope:** personal finance · equities · fixed income · derivatives · macro\n\n"
        "*AI-generated responses are for educational purposes only. Not financial advice.*"
    )

demo.launch()
