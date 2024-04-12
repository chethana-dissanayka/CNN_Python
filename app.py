import os
from keras.models import load_model
import streamlit as st 
import tensorflow as tf
import numpy as np

# Function to classify images using the loaded model
def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    flower_name = flower_names[np.argmax(result)]
    score = np.max(result) * 100
    outcome = f"<span style='font-size: x-large'>The image belongs to: <span style='color:yellow'>{flower_name}</span><br>Accuracy is: <span style='color:yellow'>{score:.2f}%</span></span>"

    return outcome


st.header('Flower Classification CNN Model')
flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Load the pre-trained model
model = load_model('Flower_Recog_Model.keras')

# Check if the 'upload' directory exists, and create it if it doesn't
upload_dir = 'upload'
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

# File upload
uploaded_file = st.file_uploader('Upload an Image')
if uploaded_file is not None:
    # Save the uploaded file to the 'upload' directory
    with open(os.path.join(upload_dir, uploaded_file.name), 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image
    st.image(uploaded_file, width=200)

    # Classify the uploaded image and display the result
    st.markdown(classify_images(os.path.join(upload_dir, uploaded_file.name)), unsafe_allow_html=True)


