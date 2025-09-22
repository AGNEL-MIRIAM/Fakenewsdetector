import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

# ✅ Local model directory
MODEL_DIR = r"C:\Users\Akshay\Downloads\bert_model (1)\content\drive\MyDrive\FakeNewsProject\bert_fakenews_model"

# ✅ Load tokenizer and model
@st.cache_resource
def load_model():
    try:
        tokenizer = BertTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
        model = BertForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

tokenizer, model = load_model()

# ✅ UI
st.title("Fake News Detection using BERT")
st.write("Enter a news article or headline below to check if it's likely **Fake** or **Real**.")

text = st.text_area("Enter News Text", height=200)

if st.button("Check"):
    if not text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits).item()
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            confidence = probabilities[0][predicted_class].item()
            label = "Fake" if predicted_class == 0 else "Real"
            st.success(f"Prediction: **{label}**")
            st.info(f"Confidence: {confidence:.2%}")

# ✅ Examples
st.subheader("Try example texts:")
col1, col2 = st.columns(2)

with col1:
    if st.button("Example: Fake News"):
        st.text_area("Example Text", "Scientists discover that drinking 10 cups of coffee daily can make you live forever.", height=100, key="fake_example")

with col2:
    if st.button("Example: Real News"):
        st.text_area("Example Text", "The stock market closed higher today as investors reacted positively to the latest economic data.", height=100, key="real_example")
