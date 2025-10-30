from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
import nltk
from nltk.corpus import wordnet
import streamlit as st

nltk.download("wordnet")
nltk.download("omw-1.4")

@st.cache_resource
def load_model():
    generator = pipeline("text2text-generation", model="google/flan-t5-large")
    return HuggingFacePipeline(pipeline=generator)

llm = load_model()

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)[:10]  # limit to

def word_helper(word):
    definition_prompt = f"Give a simple dictionary-style definition of the word '{word}'."
    example_prompt = f"Write a short example sentence using the word '{word}'."

    definition = llm.invoke(definition_prompt)
    example = llm.invoke(example_prompt)
    synonyms = get_synonyms(word)

    return {
        "Word": word,
        "Definition": definition,
        "Example Sentence": example,
        "Synonyms": synonyms if synonyms else ["No synonyms found"]
    }

st.set_page_config(page_title="Word Helper", page_icon="📖")

st.title("📖 Word Helper App")
st.write("Get **definitions, example sentences, and synonyms** of any word using AI + WordNet.")

# User input
word = st.text_input("Enter a word:", "")

if st.button("Get Word Info") and word.strip() != "":
    with st.spinner("Fetching word info..."):
        word_info = word_helper(word.strip())

    # Display results
    st.subheader(f"Word: {word_info['Word']}")
    st.write(f"**Definition:** {word_info['Definition']}")
    st.write(f"**Example Sentence:** {word_info['Example Sentence']}")

    st.write("**Synonyms:**")
    st.write(", ".join(word_info["Synonyms"]))

