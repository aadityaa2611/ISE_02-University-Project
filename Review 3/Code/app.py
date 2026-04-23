import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import json
import os
import traceback

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Indian Cow Breed Classifier",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    padding: 2rem 3rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    min-height: 100vh;
}

.header-container {
    text-align: center;
    padding: 2.5rem 0;
    background: linear-gradient(135deg, #ffffff 0%, #f0f4ff 100%);
    border-radius: 20px;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    margin-bottom: 2rem;
    border: 3px solid #667eea;
}

.main-title {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3.2rem;
    font-weight: bold;
}

.subtitle {
    color: #5a6c7d;
    font-size: 1.2rem;
    font-weight: 500;
}

.result-card {
    color: white;
    padding: 2rem;
    border-radius: 20px;
    text-align: center;
    margin: 1rem 0;
    box-shadow: 0 8px 25px rgba(0,0,0,0.3);
}

.conf-high { background: linear-gradient(135deg, #00b894, #00cec9); }
.conf-medium { background: linear-gradient(135deg, #fdcb6e, #e17055); }
.conf-low { background: linear-gradient(135deg, #d63031, #e84393); }

.footer {
    text-align: center;
    padding: 2rem;
    background: #ffffff;
    border-radius: 15px;
    margin-top: 3rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="header-container">
    <h1 class="main-title">Indian Cow Breed Classifier</h1>
    <p class="subtitle">Densenet-Based Recognition System (17 Indigenous Cow Breeds)</p>
</div>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_classifier():
    model_path = 'indian_livestock_model.h5'
    classes_path = 'class_names.json'

    try:
        if not os.path.exists(model_path):
            st.error("Model file not found.")
            return None, None

        if not os.path.exists(classes_path):
            st.error("Class names file not found.")
            return None, None

        model = load_model(model_path, compile=False)

        with open(classes_path, 'r', encoding='utf-8') as f:
            class_names = json.load(f)

        # verify that the number of class names matches the model's output layer
        try:
            output_units = model.output_shape[-1]
        except Exception:
            output_units = None

        if output_units is not None and output_units != len(class_names):
            st.error(
                f"Loaded model predicts {output_units} classes, "
                f"but {len(class_names)} names were provided."
            )
            return None, None

        return model, class_names

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        with st.expander("Error Details"):
            st.code(traceback.format_exc())
        return None, None


# ---------------- PREPROCESS ----------------
def preprocess_image(img, img_size=224):
    try:
        img = img.convert("RGB")
        img = img.resize((img_size, img_size))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        raise Exception(f"Preprocessing failed: {str(e)}")


# ---------------- PREDICTION ----------------
def predict_breed(model, img_array, class_names):
    try:
        predictions = model.predict(img_array, verbose=0)[0]

        if len(predictions) != len(class_names):
            raise ValueError("Mismatch between model output and class names")

        predicted_index = np.argmax(predictions)
        predicted_breed = class_names[predicted_index]
        confidence = float(predictions[predicted_index] * 100)

        return predicted_breed, confidence

    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")


# ---------------- MAIN APP ----------------
def main():

    model, class_names = load_classifier()

    if model is None:
        st.warning("Model not loaded properly.")
        return

    st.success(f"Model Loaded Successfully | Total Cow Breeds: {len(class_names)}")

    tab1, tab2 = st.tabs(["Classify Image", "Model Info"])

    # -------- TAB 1 --------
    with tab1:
        st.markdown("### Upload Cow Image")

        uploaded_file = st.file_uploader(
            "Choose an image",
            type=['jpg', 'jpeg', 'png']
        )

        if uploaded_file is not None:

            img = Image.open(uploaded_file)

            col1, col2 = st.columns(2)

            with col1:
                st.image(img, use_column_width=True)

            with col2:
                if st.button("Predict Breed", use_container_width=True):

                    with st.spinner("Analyzing image..."):

                        img_array = preprocess_image(img)
                        breed, confidence = predict_breed(model, img_array, class_names)

                    if confidence >= 80:
                        conf_class = "conf-high"
                        conf_text = "High Confidence"
                    elif confidence >= 60:
                        conf_class = "conf-medium"
                        conf_text = "Moderate Confidence"
                    else:
                        conf_class = "conf-low"
                        conf_text = "Low Confidence"

                    st.markdown(f"""
                    <div class="result-card {conf_class}">
                        <h2>{breed}</h2>
                        <h3>{confidence:.2f}%</h3>
                        <p>{conf_text}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    if confidence < 60:
                        st.warning("Low confidence. Try a clearer image.")

    # -------- TAB 2 --------
    with tab2:
        st.markdown("### Model Information")

        st.markdown(f"""
        - **Architecture:** DenseNet121  
        - **Technique:** Transfer Learning  
        - **Input Size:** 224x224  
        - **Total Cow Breeds:** {len(class_names)}  
        - **Framework:** TensorFlow / Keras  
        """)

        st.markdown("#### Recognized Breeds")
        cols = st.columns(4)

        for i, breed in enumerate(class_names):
            cols[i % 4].markdown(f"- {breed}")

    # -------- FOOTER --------
    st.markdown("""
    <div class="footer">
        <p><strong>Review 3 Prototype Version</strong></p>
        <p>Indian Indigenous Cow Breed Classification System</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
