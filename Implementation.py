import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
import joblib
import streamlit as st

# ---------------------------
# NLTK setup
# ---------------------------
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# ---------------------------
# Train or load model
# ---------------------------
def train_model():
    df = pd.read_csv(r"C:\Users\oaksh.DESKTOP-A3EJ8OV\Internship Task ,Assignments and Projects\Flipkart - Sentiment Analysis\YONEX MAVIS 350 Nylon Shuttle.csv")
    df = df.dropna(subset=["Review text"])
    df["Review Title"] = df["Review Title"].fillna("")

    def label_sentiment(rating):
        if rating <= 2:
            return "Negative"
        elif rating == 3:
            return "Neutral"
        else:
            return "Positive"

    df["sentiment"] = df["Ratings"].apply(label_sentiment)
    df["full_review"] = df["Review Title"] + " " + df["Review text"]

    X = df["full_review"]
    y = df["sentiment"]

    pipeline = Pipeline([
        ("vectorizer", CountVectorizer(preprocessor=clean_text, max_features=5000)),
        ("classifier", LinearSVC(C=1))
    ])

    pipeline.fit(X, y)
    joblib.dump(pipeline, "sentiment_pipeline.pkl")
    return pipeline

# Load trained model if exists
try:
    pipeline = joblib.load("sentiment_pipeline.pkl")
except FileNotFoundError:
    pipeline = train_model()

# ---------------------------
# Prediction function
# ---------------------------
def predict_sentiment(review_text):
    return pipeline.predict([review_text])[0]


# ---------------------------
# Streamlit App
# ---------------------------
st.title("Flipkart Shuttlecock Sentiment Analyzer")
st.write("Enter a product review to predict its sentiment (Positive / Neutral / Negative).")

review_input = st.text_area("Your Review Here:")

if st.button("Predict Sentiment"):
    if review_input.strip() == "":
        st.warning("Please enter a review first!")
    else:
        sentiment = predict_sentiment(review_input)
        st.success(f"Predicted Sentiment: {sentiment}")


# Sample review
st.write("---")
st.subheader("Example Prediction")
sample_review = "The shuttlecock is durable and has good flight. Very happy with the purchase."
if st.button("Predict Example Review"):
    example_sentiment = predict_sentiment(sample_review)
    st.info(f"Sample Review: '{sample_review}'")
    st.success(f"Predicted Sentiment: {example_sentiment}")
