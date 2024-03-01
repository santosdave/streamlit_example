import streamlit as  st
from langchain_core.messages import AIMessage, HumanMessage

# respond to user input
def get_response(user_input):
    return ("Not sure")


# Streamlit Config
st.set_page_config(page_title="Chat with  websites", page_icon="🤖")
st.title("Chat with websites")

# Streamlit sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

# Streamlit Chat Input
user_input = st.chat_input("Type your message here...")

# Chat History Stateful
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        # Starter message
        AIMessage(content="Hello, how may I help you?"),
    ]


# User Input
if user_input is not None and user_input != "":
    response = get_response(user_input)
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=response))

# Chat Mesages
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message,HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)







    