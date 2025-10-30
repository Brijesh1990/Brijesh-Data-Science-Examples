import streamlit as st
from transformers import pipeline
from PIL import Image

# Load Hugging Face model (BLIP for VQA + captioning)
vqa_pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
qa_pipeline = pipeline("visual-question-answering", model="Salesforce/blip-vqa-base")

st.title("🖼️ Image Q&A and Description App")
st.write("Upload an image and ask a question. If no question is asked, the app will describe the image.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Ask a question
    user_question = st.text_input("Ask a question about the image (or leave blank for description):")

    if st.button("Analyze Image"):
        if user_question.strip():
            # If user asked a question → run VQA
            result = qa_pipeline(image, user_question)
            st.subheader("Answer to your question:")
            st.write(result[0]['answer'])
        else:
            # If no question → provide description
            result = vqa_pipeline(image)
            st.subheader("Image Description:")
            st.write(result[0]['generated_text'])
