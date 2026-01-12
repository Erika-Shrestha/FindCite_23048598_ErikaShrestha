import os
import streamlit as st
from PIL import Image
import random
import io
import base64
import pickle
import numpy as np


# Custom styles
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(
            to bottom,
            rgba(182, 94, 186, 0.2),
            rgba(218, 235, 253, 0.5)
        );
    }

    .title-block {
        background-color: #4B0082;
        width: 250px;
        height: 50px;
        margin: 20px auto;
        border-radius: 8px;
    }

    div.stButton > button {
        background-color: #4B0082; 
        color: white;               
        height: 50px;              
        width: 150px;               
        border-radius: 12px;       
        border:none;
        font-size: 18px;             
        font-weight: bold;
        margin-left: 170px;
        transition: all 0.3s ease;
        
    }

    div.stButton > button:hover {
        background-color: #6A0DAD;  
        transform: scale(1.008);     
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Example labels
possible_labels = ["Background", "Method", "Result"]

# Load image
BASE_DIR = os.path.dirname(__file__)
img_path = os.path.join(BASE_DIR, "resources", "book.png")
img = Image.open(img_path)
rotated_img = img.rotate(15, expand=True)
buffered = io.BytesIO()
rotated_img.save(buffered, format="PNG")
img_str = base64.b64encode(buffered.getvalue()).decode()

# Centered title
st.markdown(
    """
    <h1 style='text-align: center; color: #4B0082; margin-bottom: 30px; margin-top: -40px'>
        Citation Intent Classification
    </h1>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns([1, 5.5, 1])

with col1:
    st.markdown(
        f"""
        <div style="margin-left: -150px">
            <img src="data:image/png;base64,{img_str}">
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    # Text input
    citation_text = st.text_area(
        "Enter a citation sentence:",
        height=150,
        placeholder="Example: This method follows the approach proposed by Smith et al. (2020)."
    )

    if st.button("Predict Intent"):
        if citation_text.strip() == "":
            st.warning("Please enter a citation sentence.")
        else:
            prediction = log_model.predict([citation_text])[0]

            probs = log_model.predict_proba([citation_text])[0]
            confidence = np.max(probs)

            st.subheader("Prediction from Both Models")
            # Create a table for clear comparison
            pred_df = {
                "Model": ["Logistic Regression", "SciBERT"],
                "Predicted Label": [prediction],
                "Confidence Score": [round(confidence, 2)]
            }
            st.table(pred_df)

with col3:
    st.markdown(
        f"""
        <div style="margin-left: 50px; margin-top: 150px; width: 100px">
            <img src="data:image/png;base64,{img_str}">
        </div>
        """,
        unsafe_allow_html=True
    )


