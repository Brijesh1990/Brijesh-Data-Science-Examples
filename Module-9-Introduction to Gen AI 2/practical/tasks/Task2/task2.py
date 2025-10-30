import re
import streamlit as st
from transformers import pipeline

# -----------------------------
# Setup (cached so the model loads once)
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_model():
    # Small instruct-style model that works well for concise generation
    # You can switch to 'google/flan-t5-small' if you need even lighter
    return pipeline(
        task="text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=128,
        temperature=0.2,
        do_sample=False,
    )

llm = load_model()

# -----------------------------
# Helpers
# -----------------------------
SENTENCE_END_RE = re.compile(r"([.!?])+")

def normalize_fact(text: str) -> str:
    """Keep only the first sentence, trim, and cap length to ~25 words."""
    # Split by common sentence enders
    parts = SENTENCE_END_RE.split(text.strip())
    if len(parts) >= 2:
        first_sentence = (parts[0] + parts[1]).strip()
    else:
        first_sentence = text.strip()

    # Remove leading bullets/numbers
    first_sentence = re.sub(r"^[-*\d\.\)\s]+", "", first_sentence)

    # Limit to ~25 words
    words = first_sentence.split()
    if len(words) > 25:
        first_sentence = " ".join(words[:25]).rstrip(",;:") + "."

    # Ensure ends with a period
    if not re.search(r"[.!?]$", first_sentence):
        first_sentence += "."

    return first_sentence

def make_prompt(topic: str) -> str:
    return (
        "You are a precise, factual assistant. "
        "Given a topic that is an animal, a bird, or a place, "
        "produce exactly three distinct, interesting facts. "
        "Each fact must be ONE sentence, concise (<25 words), factual, and non-overlapping. "
        "Do not include numbering or extra commentary. Output each fact on a new line.\n\n"
        f"Topic: {topic}\nFacts:"
    )


def generate_facts(topic: str):
    prompt = make_prompt(topic)
    raw = llm(prompt)[0]["generated_text"]

    # Split on newlines, filter empties, keep first 3
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    if not lines:
        lines = [raw.strip()] if raw.strip() else []

    facts = []
    for line in lines:
        if len(facts) >= 3:
            break
        facts.append(normalize_fact(line))

    # If the model didn't give 3 clear lines, try splitting the raw text by sentence enders
    if len(facts) < 3:
        # Create provisional sentences
        sentences = re.split(r"(?<=[.!?])\s+", raw.strip())
        for s in sentences:
            s = s.strip()
            if s and s not in facts:
                facts.append(normalize_fact(s))
            if len(facts) >= 3:
                break

    # Final guard: deduplicate and cap to 3
    unique = []
    for f in facts:
        if f and f not in unique:
            unique.append(f)
        if len(unique) == 3:
            break

    return unique

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Fun Fact Generator", page_icon="✨")
st.title("✨ Fun Fact Generator")
st.write("Enter the name of an **animal**, **bird**, or **place**, and I'll return three interesting one‑sentence facts.")

with st.form("fact_form"):
    topic = st.text_input("Topic (animal / bird / place)", placeholder="e.g., Tiger, Sparrow, Kyoto")
    submitted = st.form_submit_button("Generate Facts")

if submitted:
    topic = (topic or "").strip()
    if not topic:
        st.warning("Please enter a topic.")
    else:
        with st.spinner("Thinking up concise facts..."):
            facts = generate_facts(topic)
        if not facts:
            st.error("Sorry, I couldn't generate facts. Try a different topic.")
        else:
            st.subheader(f"Three facts about {topic}:")
            for i, f in enumerate(facts, start=1):
                st.markdown(f"**{i}.** {f}")

st.caption("Model: google/flan-t5-base · Generation kept deterministic and concise. You can swap to 'google/flan-t5-small' for faster loads.")