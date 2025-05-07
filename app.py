import streamlit as st
from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")

st.title("Sentiment Analysis Tool")
st.write("Enter a sentence and I'll tell you the sentiment!")

user_input = st.text_area("Text to analyze", "I love AI projects!")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        result = sentiment_pipeline(user_input)[0]
        st.success(f"**Sentiment:** {result['label']} \n\n**Confidence:** {round(result['score'] * 100, 2)}%")
        