import re
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model

class AnswerPredictor:
    def __init__(self, model_path, tokenizer, max_length, data):
        try:
            self.model = load_model(model_path)
        except Exception as e:
            raise ValueError(f"Error loading the model: {e}")
        
        if tokenizer is None:
            raise ValueError("Tokenizer cannot be None.")
        if data is None or data.empty:
            raise ValueError("Data cannot be None or empty.")
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = data
        self.vectorizer = TfidfVectorizer().fit(data['Question'].tolist())
        self.question_vectors = self.vectorizer.transform(data['Question'].tolist())

    def preprocess_question(self, text):
        text = text.lower()
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def complete_question(self, question):
        if not question.endswith('?'):
            question += '?'
        return question

    def find_best_match(self, question):
        question_vector = self.vectorizer.transform([question])
        similarities = cosine_similarity(question_vector, self.question_vectors)
        best_match_index = similarities.argmax()
        
        # Check for a similarity threshold (optional)
        if similarities[0, best_match_index] < 0.1:  # Adjust threshold as necessary
            return None  # No suitable match found
        
        return best_match_index

    def predict_answer(self, question):
        processed_question = self.complete_question(self.preprocess_question(question))
        best_match_index = self.find_best_match(processed_question)

        if best_match_index is None:
            return "Sorry, I couldn't find an answer to that question."

        return self.data['Answer'].iloc[best_match_index].strip()
