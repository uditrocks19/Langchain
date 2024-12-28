import langchain_helper as lch
import streamlit as st
import textwrap
st.title("Youtube video Summary bot")

with st.sidebar:
    with st.form(key = 'my_form'):
        youtube_url = st.sidebar.text_area(
            label= "what is your video url?",
            max_chars=100
        )
        query = st.sidebar.text_area(label="Ask me about the video",
                                     max_chars=50 ,
                                     key = "query")
        submit_button = st.form_submit_button(label = "Submit")

if query and youtube_url:
    db = lch.create_vector_db_from_youtube_url(youtube_url)
    response = lch.get_response_from_query(db , query)
    st.subheader("Answer")
    st.subheader(textwrap.fill(response , width = 80))
