import streamlit as st

def about_page():
    with open("README.md", "r", encoding = 'utf-8') as md_file:
        md_text = md_file.read()

    st.markdown(md_text, unsafe_allow_html=True)