import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# Load the combined dataset
df = pd.read_csv("combined_fakenewsnet.csv")

# Split features and labels
X = df['text']
y = df['label']

# Split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

# Train PassiveAggressiveClassifier
model = PassiveAggressiveClassifier(max_iter=1000)
model.fit(x_train_vec, y_train)

# Evaluate the model
y_pred = model.predict(x_test_vec)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model trained with accuracy: {acc:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Save model and vectorizer
pickle.dump(model, open("models/model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))
print("✅ Model and vectorizer saved to 'models/' folder")
