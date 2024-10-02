import pandas as pd
import re
import json
from sklearn.utils import resample
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from transformers import T5Tokenizer, T5ForConditionalGeneration

class DataPreprocessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.tokenizer = Tokenizer()
        self.paraphrase_tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.paraphrase_model = T5ForConditionalGeneration.from_pretrained("t5-small")
        self.max_length = 0
        self.vocab_size = 0

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def load_and_preprocess_data(self):
        data = pd.read_csv(self.filepath)
        data['Question'] = data['Question'].apply(self.preprocess_text)
        data['Answer'] = data['Answer'].apply(self.preprocess_text)
        return data

    def oversample_data(self, data, n_samples=1200):
        return resample(data, replace=True, n_samples=n_samples, random_state=42)

    def tokenize_and_pad(self, data):
        self.tokenizer.fit_on_texts(data['Question'].tolist() + data['Answer'].tolist())
        questions_seq = self.tokenizer.texts_to_sequences(data['Question'])
        answers_seq = self.tokenizer.texts_to_sequences(data['Answer'])
        self.max_length = max(max(len(seq) for seq in questions_seq), max(len(seq) for seq in answers_seq))
        questions_padded = pad_sequences(questions_seq, maxlen=self.max_length, padding='post')
        answers_padded = pad_sequences(answers_seq, maxlen=self.max_length, padding='post')
        return questions_padded, answers_padded

    def prepare_labels(self, answers_padded):
        answers_shifted = answers_padded[:, 1:]
        self.vocab_size = len(self.tokenizer.word_index) + 1
        return to_categorical(answers_shifted, num_classes=self.vocab_size)

    def augment_data(self, data):
        augmented_data = []
        for _, row in data.iterrows():
            question, answer = row['Question'], row['Answer']
            input_ids = self.paraphrase_tokenizer.encode("paraphrase: " + question, return_tensors="pt")
            outputs = self.paraphrase_model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)
            paraphrase = self.paraphrase_tokenizer.decode(outputs[0], skip_special_tokens=True)
            augmented_data.append({'Question': paraphrase, 'Answer': answer})
        return pd.concat([data, pd.DataFrame(augmented_data)], ignore_index=True)

    def save_tokenizer(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.tokenizer.word_index, f)

    def preprocess(self):
        data = self.load_and_preprocess_data()
        augmented_data = self.augment_data(data)
        data_oversampled = self.oversample_data(augmented_data)
        questions_padded, answers_padded = self.tokenize_and_pad(data_oversampled)
        labels = self.prepare_labels(answers_padded)
        return questions_padded, labels, self.vocab_size, self.max_length

def load_tokenizer(filename):
    # Load the tokenizer's word index
    with open(filename, 'r') as f:
        word_index = json.load(f)
    
    tokenizer = Tokenizer()
    tokenizer.word_index = word_index
    tokenizer.index_word = {index: word for word, index in word_index.items()}
    return tokenizer
