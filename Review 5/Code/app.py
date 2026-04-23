import streamlit as st
import torch
from PIL import Image
from model import CattleBreedClassifier
from utils import load_class_names, predict
import os
import traceback

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
    padding: 1.5rem;
    border-radius: 15px;
    text-align: center;
    margin: 0.5rem 0;
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
    <p class="subtitle">EfficientNet-Based Recognition System (50 Indigenous Cow Breeds)</p>
</div>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_classifier():
    model_path = 'best_cattle_classifier_v2.pth'
    classes_path = 'class_names.json'

    try:
        if not os.path.exists(model_path):
            st.error("Model file not found.")
            return None, None

        if not os.path.exists(classes_path):
            st.error("Class names file not found.")
            return None, None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        class_names = load_class_names()
        num_classes = len(class_names)

        model = CattleBreedClassifier(num_classes)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        return model, class_names, device

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        with st.expander("Error Details"):
            st.code(traceback.format_exc())
        return None, None, None

# ---------------- MAIN APP ----------------
def main():
    model, class_names, device = load_classifier()

    if model is None:
        st.warning("Model not loaded properly.")
        return

    st.success(f"Model Loaded Successfully | Total Cow Breeds: {len(class_names)}")

    tab1, tab2 = st.tabs(["Classify Image", "Model Info"])

    # -------- TAB 1 ---------
    with tab1:
        st.markdown("### Upload Cow Image")

        uploaded_file = st.file_uploader(
            "Choose an image",
            type=['jpg', 'jpeg', 'png']
        )

        if uploaded_file is not None:
            img = Image.open(uploaded_file).convert("RGB")

            col1, col2 = st.columns(2)

            with col1:
                st.image(img, use_column_width=True)

            with col2:
                if st.button("Predict Breed", use_container_width=True):
                    with st.spinner("Analyzing image..."):
                        results = predict(img, model, class_names, device)

                    st.markdown("### Top Predictions:")

                    for i, (breed, prob) in enumerate(results):
                        confidence = prob * 100
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
                            <h3>{i+1}. {breed}</h3>
                            <h4>{confidence:.2f}%</h4>
                            <p>{conf_text}</p>
                        </div>
                        """, unsafe_allow_html=True)

    # -------- TAB 2 ---------
    with tab2:
        st.markdown("### Model Information")

        st.markdown(f"""
        - **Architecture:** EfficientNet V2 Small  
        - **Technique:** Transfer Learning  
        - **Input Size:** 224x224  
        - **Total Cow Breeds:** {len(class_names)}  
        - **Framework:** PyTorch  
        """)

        st.markdown("#### Recognized Breeds")
        cols = st.columns(4)

        for i, breed in enumerate(class_names):
            cols[i % 4].markdown(f"- {breed}")

    # -------- FOOTER ---------
    st.markdown("""
    <div class="footer">
        <p><strong>Cattle Breed Classification System</strong></p>
        <p>Indian Indigenous Cow Breed Recognition</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()