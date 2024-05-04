import streamlit as st
from PIL import Image

def upload_image_page():
    st.title('Upload Image')

    # File uploader widget
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Display the uploaded image
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

def main():
    # Nhúng file CSS bằng phương thức st.html()
    st.markdown("""
        <link rel="stylesheet" href="sidebar.css">
    """, unsafe_allow_html=True)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ("Home", "Upload Image", "Settings"))

    if page == "Home":
        st.title("Home Page")
        st.write("This is the home page.")
    elif page == "Upload Image":
        upload_image_page()
    elif page == "Settings":
        st.title("Settings Page")
        st.write("This is the settings page.")

if __name__ == "__main__":
    main()
