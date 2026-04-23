import streamlit as st
import os
from src.preprocess import DocumentPreProcessor
from src.extractor import DataExtractor
from src.brain import DocumentBrain
import json

st.set_page_config(page_title="Logistic Intelligence OCR", layout="wide")

st.title("📦 Logistics Document Intelligence")
st.write("Upload a shipping document to extract structured data automatically.")

uploaded_file = st.file_uploader(
    "Choose an invoice or BOL image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp_upload.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Document")
        st.image(uploaded_file, use_container_width=True)

    with col2:
        with st.spinner("🔄 Processing Document..."):
            # 1. Preprocess
            processor = DocumentPreProcessor("temp_upload.jpg")
            processor.rescale_image()
            cleaned_img = processor.apply_cleaning()

            # 2. Extract
            extractor = DataExtractor()
            raw_results = extractor.extract_text(
                "temp_upload.jpg")  # For simplicity in the UI demo

            # 3. Brain
            brain = DocumentBrain()
            structured_data = brain.extract_structured_data(raw_results)

            st.subheader("🤖 Extracted Data (JSON)")
            # Try to format the output as JSON for beauty
            try:
                # Remove markdown code blocks if AI included them
                clean_json = structured_data.replace(
                    "```json", "").replace("```", "")
                st.json(json.loads(clean_json))
            except:
                st.write(structured_data)

    st.success("Extraction Complete!")
