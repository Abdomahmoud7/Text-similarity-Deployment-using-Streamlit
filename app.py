from controller.LoadModel import LoadModel
from controller.GetPredection import GetPredection

import streamlit as st

st.title("AI Twitter Sentiment Classifier")
st.write("Enter text to analyze sentiment (positive or bad):")

user_input = st.text_area("Insert tweet here:")

model = LoadModel("sentiment_model.pkl")

if st.button("Sentiment analysis"):
    predection = GetPredection(model, user_input)

    st.write(f"the predict is: {predection}")
