import streamlit as st
from about import about_page
from predict import predict_page

st.set_page_config("Ali's Pokédex")

url = "https://slackmojis.com/emojis/1155-pokeball/download"
html_content = f"""
<div style = "display: flex; align-items: center;">
<h1 style = "margin-right: 1px;">Pokémon Classifier</h1>
<img src = "{url}" alt = "Pokeball GIF" style = "width: 75px; height: 75px;">
</div>
"""
st.markdown(html_content, unsafe_allow_html = True)
page = st.sidebar.selectbox("Navigate", ["About", "Predict"])

if page == "About":
    about_page()

if page == "Predict":
    predict_page()