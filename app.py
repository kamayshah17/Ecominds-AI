import streamlit as st
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image
import random

# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(
    page_title="EcoMind AI",
    layout="wide"
)

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_waste_model():
    return load_model("waste_classifier_binary.h5")

model = load_waste_model()

# ----------------------------
# Waste Prediction
# ----------------------------
def predict_waste(img):
    img = img.convert("RGB")  
    img = img.resize((224,224))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x)[0][0]
    return "Non-Recyclable" if pred > 0.5 else "Recyclable"

# ----------------------------
# Energy Recommendations
# ----------------------------
def energy_recommendation(appliance, hours):
    suggestions = []
    if appliance.lower() == "fan" and hours > 1:
        suggestions.append("Switch off the fan after 30 minutes to save electricity.")
    if appliance.lower() == "washing machine":
        suggestions.append("Use eco-mode to reduce energy and water consumption.")
    if appliance.lower() == "light" and hours > 2:
        suggestions.append("Turn off lights when not in use.")
    if not suggestions:
        suggestions.append("Your usage looks efficient.")
    return suggestions

# ----------------------------
# Gamification - Eco Score
# ----------------------------
if "score" not in st.session_state:
    st.session_state.score = 0

def update_score(points):
    st.session_state.score += points

# ----------------------------
# Custom CSS for Cleaner UI
# ----------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
        padding: 20px;
    }
    .stButton button {
        background-color: #2b8a3e;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton button:hover {
        background-color: #237032;
    }
    .stMetric {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ----------------------------
# Streamlit App Layout
# ----------------------------
st.title("EcoMind AI – Smart Energy & Waste Awareness Assistant")
st.write("An AI-powered tool for sustainable living. Classify waste, get energy-saving tips, and track your eco-friendly progress.")

# Display Random Eco Tip
if "eco_tip" not in st.session_state:
    with open("tips.txt", "r", encoding="utf-8") as f:
        eco_tips = f.readlines()
    eco_tips = [tip.strip().strip('",') for tip in eco_tips if tip.strip()]
    st.session_state.eco_tip = random.choice(eco_tips)

st.header("Eco Tip of the Day")
st.info(st.session_state.eco_tip)
# st.info(f"Eco Tip of a Day :\n\n{st.session_state.eco_tip}")

# Waste Classification Section
st.header("Waste Classification")
uploaded_file = st.file_uploader("Upload a photo of a waste item", type=["jpg","jpeg","png"])

# Initialize session state
if "last_filename" not in st.session_state:
    st.session_state.last_filename = None

if uploaded_file:
    img = Image.open(uploaded_file)

    # Show a button to open "popup"
    if st.button("View Image"):
        with st.expander("Uploaded Image Preview", expanded=True):
            st.image(img, caption="Uploaded Waste Item", use_column_width=True)

    # Prediction
    category = predict_waste(img)
    st.success(f"Classified as: **{category}**")

    # Update score only if new file
    if uploaded_file.name != st.session_state.last_filename:
        if category == "Recyclable":
            st.info("Recommendation: Place this item in the recycling bin.")
            update_score(10)
        else:
            st.warning("Recommendation: Dispose this in general waste.")
            update_score(-5)

        # Save filename in session state
        st.session_state.last_filename = uploaded_file.name
    else:
        st.error("This file was already uploaded, score not updated.")


# Energy Recommendations Section
st.header("Energy Usage Suggestions")
col1, col2 = st.columns(2)
with col1:
    appliance = st.text_input("Enter appliance (e.g., Fan, Light, Washing Machine)")
with col2:
    hours = st.number_input("Enter hours of usage", min_value=0.0, step=0.5)

if st.button("Get Recommendation"):
    if appliance.strip() == "" or hours <= 0:
        st.error("Please enter a valid appliance and hours of usage.")
    else:
        current_input = f"{appliance.lower()}-{hours}"
        if "last_input" not in st.session_state or st.session_state.last_input != current_input:
            recs = energy_recommendation(appliance, hours)
            st.subheader("Suggestions")
            for rec in recs:
                st.write("- " + rec)
            update_score(5)
            st.session_state.last_input = current_input
        else:
            st.error("You have already submitted this input. Please change the input to get new recommendations.")

# Eco Score Section
st.header("Your Sustainability Score")
st.metric(label="Eco Score", value=st.session_state.score)
st.progress(min(st.session_state.score / 100, 1.0))

# Footer
st.markdown("---")
st.caption("EcoMind AI © 2025 | Encouraging small steps for a sustainable future.")
