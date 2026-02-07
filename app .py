
import streamlit as st
import pickle

st.title("Spam Detection NLP App")

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

msg = st.text_input("Enter a message")

if st.button("Predict"):
    if msg:
        result = model.predict([msg])[0]
        st.error("SPAM") if result == 1 else st.success("NOT SPAM")
