import re
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model

class AnswerPredictor:
    def __init__(self, model_path, tokenizer, max_length, data):
        self.model = load_model(model_path)
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
        return best_match_index

    def predict_answer(self, question):
        processed_question = self.complete_question(self.preprocess_question(question))
        best_match_index = self.find_best_match(processed_question)
        return self.data['Answer'].iloc[best_match_index].strip()
