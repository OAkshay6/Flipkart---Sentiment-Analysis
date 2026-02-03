# Flipkart Sentiment Analysis

A machine learning web application that predicts the sentiment of product reviews (Positive, Negative, or Neutral). The application is deployed on AWS EC2 and built using Streamlit.

## Live Demo
URL: http://16.170.246.113:8501
(Note: This link works as long as the AWS EC2 instance is active)

## Model Development Process
This project involved a rigorous experimentation phase to identify the most effective classification model:

1. Text Preprocessing: Experimented with both Bag of Words (BoW) and TF-IDF vectorization techniques.
2. Model Selection: Trained and evaluated 6 different classification algorithms.
3. Performance Metric: Used the Macro F1 Score to evaluate performance, ensuring the model handled class imbalance effectively.
4. Best Model: The Support Vector Machine (SVM) combined with Bag of Words (BoW) achieved the highest Macro F1 Score.
5. Optimization: Performed hyperparameter tuning on the SVM model to maximize accuracy.
6. Pipeline: Implemented a scikit-learn pipeline to streamline preprocessing and prediction.

## Technologies Used
* Python
* Streamlit (Frontend UI)
* Scikit-learn (Machine Learning)
* Pandas (Data Processing)
* AWS EC2 (Cloud Deployment)

## Project Structure
* Implementation.py: The main Streamlit application script.
* sentiment_pipeline.pkl: The trained machine learning model and pipeline.
* requirements.txt: List of Python dependencies required to run the app.

## How to Run Locally
Follow these steps to run the application on your local machine using Anaconda Prompt or a standard terminal:

1. Clone the repository
   git clone https://github.com/OAkshay6/Flipkart---Sentiment-Analysis.git

2. Open Anaconda Prompt (or Terminal)
   Open your prompt and change the directory to the folder where you cloned the code:
   cd path/to/your/folder

3. Install requirements
   pip install -r requirements.txt

4. Run the application
   streamlit run Implementation.py
