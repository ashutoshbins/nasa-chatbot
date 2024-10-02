import tensorflow as tf

class ModelTrainer:
    def __init__(self, vocab_size, max_length, embedding_dim=100):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.model = None

    def build_model(self):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_length))
        self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)))
        self.model.add(tf.keras.layers.Dense(self.vocab_size, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return self.model

    def train(self, questions_padded, labels, epochs=180, batch_size=16, validation_split=0.2):
        self.model.fit(questions_padded, labels, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    def save_model(self, model_path='models/chatbot_model.keras'):
        self.model.save(model_path)  # Save model in .keras format
