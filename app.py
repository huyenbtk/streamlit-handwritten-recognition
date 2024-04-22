import streamlit as st
from PIL import Image
import pytesseract

def ocr_core(image):
    text = pytesseract.image_to_string(image)
    return text

st.title('OCR Streamlit App')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Processing...")
    extracted_text = ocr_core(image)
    st.write(extracted_text)
