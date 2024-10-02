import streamlit as st
import numpy as np
import re
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

class AnswerPredictor:
    def __init__(self, model, tokenizer, max_length):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def preprocess_question(self, text):
        text = text.lower()
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def predict_answer(self, question):
        question_seq = self.tokenizer.texts_to_sequences([self.preprocess_question(question)])
        question_padded = pad_sequences(question_seq, maxlen=self.max_length, padding='post')
        prediction = self.model.predict(question_padded)
        predicted_sequence = np.argmax(prediction, axis=-1)
        predicted_answer = [self.tokenizer.index_word.get(idx, '') for idx in predicted_sequence[0] if idx > 0]
        return ' '.join(predicted_answer).strip()

# Load the model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model = load_model('chatbot_model.keras')
    with open('tokenizer.pickle', 'rb') as f:
        tokenizer = pickle.load(f)
    max_length = 20  # This should match the max_length used during training
    return model, tokenizer, max_length

# Streamlit UI
st.title("Exoplanet Q&A Chatbot")

# Load model and tokenizer
model, tokenizer, max_length = load_model_and_tokenizer()

# Initialize the AnswerPredictor
predictor = AnswerPredictor(model, tokenizer, max_length)

# Input form
question = st.text_input("Ask a question about exoplanets:")
if st.button("Get Answer"):
    if question.strip() != "":
        predicted_answer = predictor.predict_answer(question)
        st.write(f"Predicted Answer: {predicted_answer}")
    else:
        st.write("Please enter a question.")
