import streamlit as st
import pandas as pd
from answer_predictor import AnswerPredictor
from data_preprocessor import load_tokenizer

# Load resources with error handling
try:
    data = pd.read_csv('data/data.csv')
    tokenizer = load_tokenizer('models/tokenizer.json')
    max_length = 100
    predictor = AnswerPredictor('models/chatbot_model.keras', tokenizer, max_length, data)
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

# Streamlit app UI
st.title("Exoplanet Q&A Chatbot")
st.write("Ask any question about exoplanets!")

# Text input for the question
user_question = st.text_input("Your Question", "")

# Display the answer when the user submits a question
if st.button('Get Answer'):
    if user_question:
        answer = predictor.predict_answer(user_question)
        st.write(f"**Answer**: {answer}")
    else:
        st.write("Please enter a question.")
