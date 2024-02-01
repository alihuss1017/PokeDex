import streamlit as st
from about import about_page
from predict import predict_page
st.set_page_config("Ali's Pokedex")

st.title("Ali's Pokedex")

page = st.sidebar.selectbox("Navigate", ["About", "Predict"])

if page == "About":
    about_page()

if page == "Predict":
    predict_page()