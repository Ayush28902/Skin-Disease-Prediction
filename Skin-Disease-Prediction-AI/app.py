import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# === CONFIG ===
st.set_page_config(page_title="Automated Diagnosis of Skin Diseases", layout="wide")

# === CUSTOM CSS ===
st.markdown("""
    <style>
        .stApp {
            background-color: #c98653;
        }
        .css-18e3th9 {  /* Sidebar background */
            background-color: #f3edf7 !important;
        }
        .css-1d391kg {  /* Main content area background */
            background-color: #f3edf7 !important;
        }
    </style>
""", unsafe_allow_html=True)


MODEL_PATH = "./model/skin_disease_model.h5"
LABEL_MAP = {
    'mel': 'melanoma',
    'nv': 'nevus',
    'bkl': 'keratosis',
    'bcc': 'basal cell carcinoma',
    'akiec': 'actinic keratoses',
    'df': 'dermatofibroma',
    'vasc': 'vascular lesion',
    'eczema': 'eczema',
    'psoriasis': 'psoriasis'
}
CLASSES = list(LABEL_MAP.keys())

@st.cache_resource
def load_predictor():
    return load_model(MODEL_PATH)

model = load_predictor()

def predict_image(img):
    image_resized = img.resize((299, 299))
    image_array = img_to_array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    preds = model.predict(image_array)[0]
    ranked = sorted(zip(CLASSES, preds), key=lambda x: x[1], reverse=True)
    return ranked

# === SIDEBAR ===
# === SIDEBAR WITH UNIFORM BUTTONS ===
st.sidebar.markdown("""
    <style>
        .sidebar-button {
            background-color: #6bdaed;
            border: 1px solid #ddd;
            border-radius: 6px;
            padding: 12px;
            margin-bottom: 10px;
            text-align: center;
            font-weight: 600;
            font-size: 16px;
            transition: background-color 0.3s ease;
            cursor: pointer;
        }
        .sidebar-button:hover {
            background-color: #8d6bed;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

menu_items = ["Home", "Downloads", "Demo Video", "About", "Contact Us"]

for item in menu_items:
    st.sidebar.markdown(f"<div class='sidebar-button'>{item}</div>", unsafe_allow_html=True)


# === MAIN CONTENT ===
st.markdown("<h1 style='text-align:center;'>Automated Diagnosis of Skin Diseases with Image Recognition 🖥️🩺</h1>", unsafe_allow_html=True)
st.markdown("### <div style='text-align:center;'>We'd love to know more about you!</div>", unsafe_allow_html=True)

# === USER FEEDBACK FORM ===
with st.form("user_feedback_form"):
    col1, col2 = st.columns(2)
    with col1:
        who = st.selectbox("Who are you?", ["Select an option", "Student", "Doctor", "Researcher", "Other"])
    with col2:
        source = st.selectbox("How did you find our website?", ["Select an option", "Google", "Social Media", "Friend", "Other"])
    
    feedback = st.text_area("If you'd like to share any thoughts, please write here:")
    submit_button = st.form_submit_button("Submit")

# === DISCLAIMER ===
st.markdown("<p style='color:red; font-weight:bold;'>"
            "Please note that although our model achieves an accuracy rate of 85%, its predictions should be considered with a limited guarantee. "
            "Determining the precise type of skin lesion should be done by a qualified doctor for an accurate diagnosis."
            "</p>", unsafe_allow_html=True)

# === CONSENT CHECKBOX ===
agree = st.checkbox("I understand and accept")

# === IMAGE UPLOAD ===
# === IMAGE UPLOAD ===
if agree:
    uploaded_file = st.file_uploader("Drag and drop or browse a skin image (PNG, JPG, JPEG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')

        # Resize the uploaded image
        img_resized = img.resize((120 , 70))  # Resize to a more manageable size
        col1, col2 = st.columns(2)  # Two columns for side-by-side layout

        # Show the image on the left column
        with col1:
            st.image(img_resized, caption='📸 Uploaded Image', use_column_width=True)

        # Show the prediction results on the right column
        with col2:
            st.write("⏳ Analyzing the image... Please wait...")

            results = predict_image(img)

            st.success("✅ Analysis complete!")
            st.subheader("📊 Top 5 Predicted Skin Conditions")
            for i, (label, prob) in enumerate(results[:5]):
                full_label = LABEL_MAP.get(label, label)
                st.markdown(f"<p style='font-size:18px;'>👉 <strong>{i+1}. {full_label.title()}</strong> — "
                            f"<span style='color:#1f77b4;'>{prob*100:.2f}%</span></p>", unsafe_allow_html=True)


else:
    st.warning("✅ Please accept the terms before uploading an image.")

# === FOOTER ===
st.markdown("""<hr><div style='text-align:center; font-size: 18px; color: #36030a;'>Made with ❤️ by AYUSH | Powered by TensorFlow & Streamlit</div>""", unsafe_allow_html=True)
