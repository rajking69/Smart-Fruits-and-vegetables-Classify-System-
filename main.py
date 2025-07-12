import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("model.h5")
    image = Image.open(test_image)
    image = image.resize((64, 64))  # Resize to match model input
    input_arr = np.array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Contact"])

# Main Page
if app_mode == "Home":
    st.header("FRUITS & VEGETABLES RECOGNITION SYSTEM")
    image_path = "Cover Photo.jpg"
    st.image(image_path)

    # Prediction Feature on Home Page
    st.header("Model Prediction")
    test_image = st.file_uploader("Choose an Image:")

    if test_image is not None:
        try:
            # Check file type
            if test_image.type not in ["image/jpeg", "image/png"]:
                st.error("Please upload a valid image (JPEG or PNG).")
            else:
                # Open and display the image
                image = Image.open(test_image)
                if image.format not in ['PNG', 'JPEG']:
                    image = image.convert("RGB")  # Convert to RGB if needed
                st.image(image, use_column_width=True)

        except Exception as e:
            st.error(f"Error loading image: {e}")

    # Predict button
    if st.button("Predict"):
        try:
            st.snow()
            st.write("Our Prediction")
            result_index = model_prediction(test_image)
            with open("labels.txt") as f:
                content = f.readlines()
            label = [i.strip() for i in content]
            st.success(f"Model is predicting it's a {label[result_index]}")
        except Exception as e:
            st.error(f"Error: {e}")

elif app_mode == "About Project":
    # About Project Section
    st.header("About Project")

    # Project Overview
    st.subheader("Project Overview")
    st.text("""
    The "Fruits & Vegetables Recognition System" is a machine learning project 
    that uses a Convolutional Neural Network (CNN) model to classify various 
    fruits and vegetables. The model is trained on a diverse dataset containing 
    images of food items such as bananas, apples, cucumbers, carrots, and more. 
    This system allows users to upload images of fruits or vegetables and receive 
    predictions on the food item.
    """)

    # About the Dataset
    st.subheader("About Dataset")
    st.text("""
    The dataset used for this project consists of images of various fruits and vegetables, 
    which are grouped into categories: fruits and vegetables. Each category includes images 
    of different food items that the model learns to recognize and classify accurately.
    """)

    # Developer Details Section
    st.subheader("Developer Details")

    # Creating two columns for side-by-side display of developer details
    col1, col2 = st.columns(2)

    # Developer: Sheikh Mohammad Rajking
    with col1:
        # Resize image using PIL
        developer_image_path = "rajking.JPG"  # Change this to the path of the photo
        developer_image = Image.open(developer_image_path)
        developer_image = developer_image.resize((150, 200))  # Resize to desired size
        st.image(developer_image)  # Display the resized image
        st.markdown("<h3 style='color: #0077B6; font-weight: bold;'>Sheikh Md. Rajking</h3>", unsafe_allow_html=True)
        st.text("Contact: rajking4457@gmail.com")
        st.text("GitHub: https://github.com/rajking")

    # Developer: Adri Shikar Barua
    with col2:
        # Resize image using PIL
        adri_image_path = "adri.png"  # Change this to the path of the photo
        adri_image = Image.open(adri_image_path)
        adri_image = adri_image.resize((150, 200))  # Resize to desired size
        st.image(adri_image)  # Display the resized image
        st.markdown("<h3 style='color: #0077B6; font-weight: bold;'>Adri Shikar Barua</h3>", unsafe_allow_html=True)
        st.text("Contact: adrishikharbarua77452@gmail.com")
        st.text("GitHub: https://github.com/rajking")

    # Prevent horizontal scroll
    st.markdown(
        """
        <style>
        body {
            overflow-x: hidden;  /* Disables horizontal scrolling */
        }
        </style>
        """, unsafe_allow_html=True
    )

    # About Dataset Content
    st.subheader("Dataset Content")
    st.text("""
    This dataset consists of three folders:
    1. train (100 images each)
    2. test (10 images each)
    3. validation (10 images each)
    """)

elif app_mode == "Contact":
    st.title("Contact Information")
    st.text("Contact us at: smrajking4457@gmail.com")
