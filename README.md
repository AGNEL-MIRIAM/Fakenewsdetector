# 📰 AI-Based Fake News Detector using BERT

A web app that classifies news articles as **real or fake** using a fine-tuned BERT model and a clean Streamlit interface. Model was trained in Google Colab and deployed locally.

---

## 🚀 Features
- ⚡ Real-time fake news prediction
- 🤖 Fine-tuned BERT model (via Hugging Face Transformers)
- 💻 Streamlit-powered web interface
- 🔬 Trained on a labeled news dataset

---

## 🧠 Tech Stack
- Python 3
- BERT (bert-base-uncased)
- Hugging Face Transformers 🤗
- Streamlit
- Pandas, NumPy

---

## 🔗 Hosted Model

The BERT model used in this fake news detection project is trained and hosted on Hugging Face Hub:

👉 [View the Model on Hugging Face](https://huggingface.co/miriamagnel/bert-fake-news-detector)

---

## 💻 Installation & Usage

1. **Clone this repository**
```bash
git clone https://github.com/AGNEL-MIRIAM/Fakenewsdetector.git
cd Fakenewsdetector

# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

---

## 📁 **Project Structure**

```
fakenewsdetector/
├── app.py                 # Streamlit frontend
├── requirements.txt       # Dependencies
├── bert_model/            # Saved BERT model directory
├── README.md              # This file
└── .gitignore             # Ignore rules
```

---

##📄 **License**
Licensed under the MIT License

---

## 🙋‍♀️ Author

**Agnel Miriam**

