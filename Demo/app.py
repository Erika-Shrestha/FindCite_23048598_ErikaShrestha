import os
import streamlit as st
from PIL import Image
import io
import base64
import numpy as np
import joblib
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

#FUNCTION to preprocess string columns into cleaned texts for logistic regression model training
def preprocess_text_for_logreg(text):

    #TOKENIZES
    words = text.split()

    #REMOVES space sparse
    cleaned_words = []
    for word in words:
        word = word.strip()
        if word != '':
            cleaned_words.append(word)

    #CONVERTS to lowercase
    lowercase_words = []
    for word in cleaned_words:
        lowercase_words.append(word.lower())

    #REMOVES stopwords
    filtered_words = []
    for word in lowercase_words:
        if word not in stop_words:
            filtered_words.append(word)

    #LEMMATIZES
    lemmatized_words = []
    for word in filtered_words:
        lemma = lemmatizer.lemmatize(word)
        lemmatized_words.append(lemma)

    #STORES back to strings
    cleaned_text = ' '.join(lemmatized_words)

    return cleaned_text

#FUNCTION to preprocess string columns into cleaned texts for scibert model training
def preprocess_text_for_scibert(text):

    #REMOVES space sparse
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

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

label_map = {0: "Background", 1: "Method", 2: "Result"}

log_model = joblib.load('logreg_model.pkl')
scibert_tokenizer = AutoTokenizer.from_pretrained('scibert_model')
scibert_model = TFAutoModelForSequenceClassification.from_pretrained('scibert_model')

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
            clean_text_lr = preprocess_text_for_logreg(citation_text)
            log_prediction_idx = log_model.predict([clean_text_lr])[0]
            log_prediction = label_map.get(log_prediction_idx, "Unknown")
            log_probs = log_model.predict_proba([clean_text_lr])[0]
            log_confidence = np.max(log_probs)

            clean_text_scibert = preprocess_text_for_scibert(citation_text)
            inputs = scibert_tokenizer(clean_text_scibert, return_tensors="tf", padding=True, truncation=True, max_length=512)
            scibert_outputs = scibert_model(inputs)
            scibert_logits = scibert_outputs.logits
            scibert_probs = tf.nn.softmax(scibert_logits, axis=-1).numpy()[0]
            scibert_prediction_idx = np.argmax(scibert_probs)
            scibert_prediction = label_map.get(scibert_prediction_idx, "Unknown")
            scibert_confidence = np.max(scibert_probs)

            st.subheader("Prediction from Both Models")
            # Create a table for clear comparison
            pred_df = {
                "Model": ["Logistic Regression", "SciBERT"],
                "Predicted Label": [log_prediction, scibert_prediction],
                "Confidence Score": [round(log_confidence, 2), round(scibert_confidence, 2)]
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


