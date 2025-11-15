import streamlit as st
from transformers import pipeline
import wikipedia
import json, io

st.set_page_config(page_title="Encyclopedia AI", layout="wide")

st.title("Encyclopedia AI (Streamlit Cloud Ready)")
st.markdown(
    """
Ask questions about any topic in English.  
Uses a Hugging Face model locally and optionally fetches Wikipedia summaries for context.
"""
)

# --- Sidebar ---
st.sidebar.header("Settings")
MODEL_CHOICES = {
    "t5-small (lightweight, always available)": "t5-small",
    "google/flan-t5-small (instruction-tuned)": "google/flan-t5-small",
}
model_name = st.sidebar.selectbox("Choose model", list(MODEL_CHOICES.keys()))
model_id = MODEL_CHOICES[model_name]

use_wiki = st.sidebar.checkbox("Use Wikipedia context", value=True)
max_wiki_sentences = st.sidebar.slider("Max Wikipedia sentences", 1, 6, 3)
max_length = st.sidebar.number_input("Max generated tokens", 32, 512, 256)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2)
clear_button = st.sidebar.button("Clear conversation history")

# --- Load model ---
@st.cache_resource(show_spinner=True)
def load_generator(model_id):
    return pipeline("text2text-generation", model=model_id)

generator = load_generator(model_id)

# --- Session state ---
if "history" not in st.session_state:
    st.session_state.history = []

if clear_button:
    st.session_state.history = []
    st.success("Conversation history cleared.")

# --- Helper functions ---
def fetch_wikipedia_snippet(query, max_sentences=3):
    try:
        results = wikipedia.search(query)
        if not results:
            return None
        page_title = results[0]
        summary = wikipedia.summary(page_title, sentences=max_sentences, auto_suggest=False)
        return f"**Wikipedia snippet (source: {page_title})**:\n{summary}"
    except:
        return None

def build_prompt(question, wiki_ctx):
    prompt = f"You are an AI encyclopedia assistant. Answer concisely in English.\n\n"
    if wiki_ctx:
        prompt += wiki_ctx + "\n\n"
    prompt += f"Question: {question}\nAnswer:"
    return prompt

def generate_answer(prompt):
    gen = generator(prompt, max_length=max_length, do_sample=temperature>0, temperature=temperature)
    return gen[0]["generated_text"].strip()

# --- UI: Input ---
with st.form("query_form", clear_on_submit=False):
    user_input = st.text_area("Enter your question:", height=120)
    submitted = st.form_submit_button("Ask")

if submitted and user_input.strip():
    q = user_input.strip()
    wiki_ctx = fetch_wikipedia_snippet(q, max_wiki_sentences) if use_wiki else None
    prompt = build_prompt(q, wiki_ctx)
    answer = generate_answer(prompt)
    st.session_state.history.append((q, answer))

# --- Display history ---
st.subheader("Conversation History")
if st.session_state.history:
    for i, (user_q, bot_a) in enumerate(reversed(st.session_state.history)):
        idx = len(st.session_state.history) - i
        st.markdown(f"**User [{idx}]:** {user_q}")
        st.markdown(f"**Encyclopedia AI:**\n{bot_a}")
        st.divider()
else:
    st.info("No questions yet. Enter a question above to start.")

# --- Export conversation ---
if st.session_state.history and st.button("Export conversation as JSON"):
    buf = io.BytesIO()
    buf.write(json.dumps(st.session_state.history, ensure_ascii=False, indent=2).encode("utf-8"))
    buf.seek(0)
    st.download_button("Download JSON", data=buf, file_name="encyclopedia_ai_history.json", mime="application/json")
