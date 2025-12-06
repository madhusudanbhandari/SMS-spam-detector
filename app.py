import streamlit as st
import pickle
import sklearn
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text=text.lower()  ##to lower casa
    text=nltk.word_tokenize(text) ##tokens

    y=[] ##removing special words
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text=y[:]
    y.clear()
    for i in text:
        ps=PorterStemmer()
        y.append(ps.stem(i))

    return ' '.join(y)

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title("Email/SMS-Spam Detector")

input_sms=st.text_input("Enter the message")

if st.button("Predict"):

    #1 Preprocess
    transformed_sms=transform_text(input_sms)

    #2 Vectorize
    vector_input=tfidf.transform([transformed_sms])
    #3 predict
    result=model.predict(vector_input)[0]
    #4 display
    if result==1:
        st.header("spam")
    else:
        st.header("not spam")