import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

#load the saved tokenizer
with open("tokenizer.pkl","rb") as f:
    tokenizer = pickle.load(f)

#load the saved model
model = load_model("model.h5")



#define a function to prredict sentiment
def predict_sentiment(review):
    #tokenize and pad the review
    sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequence, maxlen=200)
    prediction = model.predict(padded_sequence)
    sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
    return sentiment

#streamlit app
st.title("Sentiment Analysis")
st.write("Enter a movie review and find out its sentiment!")


#Input review from the user
user_review = st.text_area("Enter your review here:")


#predict sentiment on button click
if st.button("Predict Sentiment"):
    if user_review.strip():
        sentiment = predict_sentiment(user_review)
        st.write(f"the sentiment of the review is **{sentiment}**.")

    else:
        st.write("please enter a valid review")