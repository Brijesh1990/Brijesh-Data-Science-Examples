import streamlit as st
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

@st.cache_resource  # cache models to avoid reloading every time
def load_translators():
    translator_fr = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
    translator_es = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")
    translator_de = pipeline("translation", model="Helsinki-NLP/opus-mt-en-de")

    llm_fr = HuggingFacePipeline(pipeline=translator_fr)
    llm_es = HuggingFacePipeline(pipeline=translator_es)
    llm_de = HuggingFacePipeline(pipeline=translator_de)
    return llm_fr, llm_es, llm_de

llm_fr, llm_es, llm_de = load_translators()

st.title("🌍 Multi-Language Translator")
st.write("Translate English text into **French, Spanish, and German** using Hugging Face models and LangChain.")

# User Input
text = st.text_area("✍️ Enter English text:", "Hello, how are you today? I am learning Generative AI with LangChain and Hugging Face.")

# Input text
#text = "Hello, how are you today? I am learning Generative AI with LangChain and Hugging Face."

if st.button("Translate"):
    if text.strip():
        # Run translations
        output_fr = llm_fr.invoke(text)
        output_es = llm_es.invoke(text)
        output_de = llm_de.invoke(text)

        # Display results
        st.subheader("Translations")
        st.markdown(f"**🇫🇷 French:** {output_fr}")
        st.markdown(f"**🇪🇸 Spanish:** {output_es}")
        st.markdown(f"**🇩🇪 German:** {output_de}")
    else:
        st.warning("⚠️ Please enter some English text to translate.")

