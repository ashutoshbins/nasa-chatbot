import streamlit as st
import pandas as pd
from answer_predictor import AnswerPredictor
from data_preprocessor import load_tokenizer

# Function to load data and model with error handling
def load_resources():
    try:
        data = pd.read_csv('data/data.csv')
    except FileNotFoundError:
        st.error("Data file not found. Please ensure 'data/data.csv' exists.")
        return None, None
    except pd.errors.EmptyDataError:
        st.error("Data file is empty. Please provide valid data.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred while loading data: {e}")
        return None, None

    try:
        tokenizer = load_tokenizer('models/tokenizer.json')
    except FileNotFoundError:
        st.error("Tokenizer file not found. Please ensure 'models/tokenizer.json' exists.")
        return None, None
    except json.JSONDecodeError:
        st.error("Error decoding tokenizer JSON file. Please check the file format.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred while loading the tokenizer: {e}")
        return None, None

    max_length = 100  # Set this based on your actual max length
    try:
        predictor = AnswerPredictor('models/chatbot_model.keras', tokenizer, max_length, data)
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None, None

    return data, predictor

# Load the data and model
data, predictor = load_resources()
if predictor is None:
    st.stop()  # Stop the app if loading failed

# Streamlit app UI
st.title("Exoplanet Q&A Chatbot")
st.write("Ask any question about exoplanets!")

# Text input for the question
user_question = st.text_input("Your Question", "")

# Display the answer when the user submits a question
if st.button('Get Answer'):
    if user_question:
        try:
            with st.spinner("Generating answer..."):
                answer = predictor.predict_answer(user_question)
                st.write(f"**Answer**: {answer}")
        except Exception as e:
            st.error(f"Error predicting answer: {e}")
    else:
        st.write("Please enter a question.")

# General exception handler for runtime errors
try:
    pass  # Main code logic would be placed here
except Exception as e:
    st.error(f"I DONT KNOW THE ANSWER YET")
