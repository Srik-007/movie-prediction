import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense
from tensorflow.keras.datasets import imdb
import tensorboard as tensorboard
import datetime
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

word_index=imdb.get_word_index()
reverse_index={value:key for key,value in word_index.items()}
model = load_model('simplernn_imdb.h5')
def decode_review(encoded_review):
    decoded_review=' '.join([reverse_index.get(i-3,'?') for i in encoded_review])
    return decoded_review
def preproccess_test(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review
def predict_sentiment(review):
    preproccess_input=preproccess_test(review)
    prediction=model.predict(preproccess_input)
    sentiment='Positive' if prediction[0][0] >0.5 else 'Negative'
    return sentiment, prediction[0][0]
import streamlit as st
st.title('Movie review sentiment analysis')
st.write('Write your review about any movie, we will classify your review as positive or negative review')
input=st.text_area('Write your review here')
if st.button('Classify'):
    input=preproccess_test(input)
    prediction=model.predict(input)
    sentiment='Positive' if prediction[0][0] >0.5 else 'Negative'
    st.write(f"Sentiment:{sentiment}")
    st.write(f"Your review score:{prediction}")
else:
    st.write("Give your review")


