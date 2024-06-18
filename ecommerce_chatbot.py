!python -m spacy download en_core_web_sm
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load the data from the CSV file
data = pd.read_csv('/content/drive/MyDrive/data.csv')

# Display the first few rows of the dataset to understand its structure
data.head()

code = """
# Load a pre-trained NLP model
nlp = spacy.load('en_core_web_sm')

def get_intent(user_input):
    doc = nlp(user_input)
    # Simple intent detection based on keywords
    if 'cream' in user_input and 'dry skin' in user_input:
        return 'find_product', 'cream for dry skin'
    # Add more intents as needed
    return 'unknown', ''

# Sample product descriptions
product_descriptions = data['description']

# Vectorize the product descriptions
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(product_descriptions)

def find_closest_products(query, top_n=5):
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarity.argsort()[-top_n:][::-1]
    return data.iloc[top_indices]

def chatbot_response(user_input):
    intent, query = get_intent(user_input)

    if intent == 'find_product':
        products = find_closest_products(query)
        response = "Here are some products you might like:\n"
        for idx, row in products.iterrows():
            response += f"{row['name']}: {row['description']}\n"
        response += "Need more details about any of these products?"
    else:
        response = "I'm not sure what you're looking for. Could you please clarify?"

    # Add some wit
    response += "\n(And don't worry, I'm here to help you find the perfect product!)"

    return response


# User Interface with Streamlit
st.title('E-commerce Chatbot')

user_input = st.text_input('You:', '')

if user_input:
    response = chatbot_response(user_input)
    st.text_area('Bot:', value=response, height=200, max_chars=None, key=None)"""

file_path = '/content/drive/MyDrive/ecommerce_chatbot.py'
with open(file_path, 'w') as f:
    f.write(code)

# Run the Streamlit app
!streamlit run '/content/drive/MyDrive/ecommerce_chatbot.py'
