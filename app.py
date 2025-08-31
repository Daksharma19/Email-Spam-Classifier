import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# Load the fitted vectorizer and trained model
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message")

def transform_text(text):
    # 1) Lower Case
    text = text.lower()

    # 2) Tokenization
    text = nltk.word_tokenize(text)

    # 3) Removing Special Characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    # 4) Removing stop words and punctutations
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    text = y[:]
    return " ".join(y)

if st.button("Predict"):
    # 1) Preprocess
    transformed_sms = transform_text(input_sms)

    # 2) Vectorize
    vector_input = vectorizer.transform([transformed_sms])

    # 3) Predict
    result = model.predict(vector_input)[0]

    # 4) Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
