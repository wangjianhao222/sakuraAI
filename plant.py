import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import wikipedia
from typing import Optional

st.set_page_config(page_title="Encyclopedia AI", layout="wide")

st.title("Encyclopedia AI (Hugging Face Model, No API Key Required)")
st.markdown(
    """
A simple Streamlit app using a publicly available Hugging Face model locally.  
- Optional Wikipedia context enhancement.  
- Note: The first run will download the model, which may take a while.
"""
)

# ---- Sidebar ----
st.sidebar.header("Settings")
MODEL_CHOICES = {
    "google/flan-t5-small (instruction-tuned, lightweight)": "google/flan-t5-small",
    "google/flan-t5-base (stronger, slower)": "google/flan-t5-base",
    "t5-small (general purpose, small)": "t5-small",
}
model_name = st.sidebar.selectbox("Choose model", list(MODEL_CHOICES.keys()), index=0)
model_id = MODEL_CHOICES[model_name]

use_wiki = st.sidebar.checkbox("Use Wikipedia context (requires internet)", value=True)
max_wiki_sentences = st.sidebar.slider("Max Wikipedia sentences", 1, 6, 3)
max_length = st.sidebar.number_input("Max generated tokens", 32, 1024, 256)
temperature = st.sidebar.slider("Temperature (higher = more creative)", 0.0, 1.0, 0.2)
clear_button = st.sidebar.button("Clear conversation history")

device = 0 if torch.cuda.is_available() else -1

@st.cache_resource(show_spinner=True)
def load_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        framework="pt"
    )
    return pipe

with st.spinner(f"Loading model {model_id} ..."):
    try:
        generator = load_model(model_id)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

# ---- Session state for chat history ----
if "history" not in st.session_state:
    st.session_state.history = []

if clear_button:
    st.session_state.history = []
    st.success("Conversation history cleared.")

# ---- Helper functions ----
def fetch_wikipedia_snippet(query: str, max_sentences: int = 3) -> Optional[str]:
    try:
        search_results = wikipedia.search(query)
        if not search_results:
            return None
        page_title = search_results[0]
        summary = wikipedia.summary(page_title, sentences=max_sentences, auto_suggest=False)
        return f"**Wikipedia snippet (source: {page_title})**:\n{summary}"
    except Exception:
        return None

def build_prompt(question: str, wiki_ctx: Optional[str]) -> str:
    """
    Construct prompt for instruction-tuned models like Flan-T5.
    """
    system = (
        "You are a professional AI encyclopedia assistant. "
        "Answer accurately, concisely, and in English. "
        "If unsure, indicate uncertainty and provide possible directions or references."
    )
    prompt = system + "\n\n"
    if wiki_ctx:
        prompt += f"{wiki_ctx}\n\n"
    prompt += f"Question: {question}\n\nAnswer:"
    return prompt

def generate_answer(prompt: str):
    gen = generator(
        prompt,
        max_length=max_length,
        do_sample=True if temperature > 0 else False,
        temperature=float(temperature),
        top_p=0.95,
        num_return_sequences=1
    )
    return gen[0].get("generated_text", "").strip()

# ---- UI: Input and conversation ----
with st.form("query_form", clear_on_submit=False):
    user_input = st.text_area("Enter your question (e.g., 'What is the history of the Eiffel Tower?')", height=120)
    submitted = st.form_submit_button("Ask")

if submitted and user_input and user_input.strip():
    q = user_input.strip()
    wiki_ctx = None
    if use_wiki:
        with st.spinner("Fetching Wikipedia snippet..."):
            try:
                wiki_ctx = fetch_wikipedia_snippet(q, max_sentences=max_wiki_sentences)
            except Exception:
                wiki_ctx = None
    prompt = build_prompt(q, wiki_ctx)
    with st.spinner("Generating answer..."):
        try:
            answer = generate_answer(prompt)
        except Exception as e:
            st.error(f"Generation failed: {e}")
            answer = "Sorry, there was an error generating the answer."

    st.session_state.history.append((q, answer))

# Display conversation history
st.subheader("Conversation History")
if not st.session_state.history:
    st.info("No questions yet. Enter a question above to start.")
else:
    for i, (user_q, bot_a) in enumerate(reversed(st.session_state.history)):
        idx = len(st.session_state.history) - i
        st.markdown(f"**User [{idx}]:** {user_q}")
        st.markdown(f"**Encyclopedia AI:**\n{bot_a}")
        st.divider()

# Export conversation
if st.session_state.history:
    import json, io
    if st.button("Export conversation as JSON"):
        buf = io.BytesIO()
        buf.write(json.dumps(st.session_state.history, ensure_ascii=False, indent=2).encode("utf-8"))
        buf.seek(0)
        st.download_button("Download JSON", data=buf, file_name="encyclopedia_ai_history.json", mime="application/json")

st.markdown("---")
st.caption("Tip: If your machine has limited resources, choose a smaller model (e.g., flan-t5-small) for smoother performance.")
