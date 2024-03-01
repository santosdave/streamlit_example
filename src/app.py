import streamlit as  st


# respond to user input
def get_response(user_input):
    return ("Not sure")

st.set_page_config(page_title="Chat with  websites", page_icon="🤖")

st.title("Chat with websites")

# APp sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

# App Chat Input
user_input = st.chat_input("Type your message here...")

# Starter message
with st.chat_message("AI"):
    st.write("Hello, how may I help you")

if user_input is not None and user_input != "":
    response = get_response(user_input)
    with st.chat_message("Human"):
        st.write(user_input)

    with st.chat_message("AI"):
        st.write(response)






    