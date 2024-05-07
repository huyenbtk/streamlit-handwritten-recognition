import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import numpy as np
import pyperclip

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

def extract_text(image):
    return "Text here"

st.sidebar.title("Settings")

st.title('Handwritting recognition')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    image_np = np.array(image.convert('RGB'))
    extracted_text = extract_text(image_np)
    
    st.text_area("Extracted Text", extracted_text, height=150)
    
    # Copy text
    if st.button('Copy Text'):
        pyperclip.copy(extracted_text)
        st.success("Text copied successfully!")
else:
    st.header("Please upload an image.")
