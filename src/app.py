import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from dotenv import load_dotenv

# load the OPENAI_API_KEY from the environment variable
load_dotenv()



# create vector store from url
def get_vector_store(url):
    # get text in documment form
    loader = WebBaseLoader(url)
    document = loader.load()

    # split document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    # create vector store from chunks
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())

    return vector_store


# get context
def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation generate a search query to look up in order to get information relevant to the conversation")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain

# get conversation rag chain


def get_conversation_rag_chain(retriever_chain):
    llm = ChatOpenAI()

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Answer the user's question based on the belowe context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


# respond to user input
def get_response(user_input):
    # create conversation chain
    retriever_chain = get_context_retriever_chain(
        st.session_state.vectore_store)

    # get conversation rag chain
    conversation_rag_chain = get_conversation_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input": user_input,
        })
    return response['answer']

# Streamlit Config
st.set_page_config(page_title="Chat with  websites", page_icon="🤖")
st.title("Chat with websites")

# Streamlit sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")


if website_url is None or website_url == "":
    st.info("Please enter a website URL")

else:

    # Chat History Stateful
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            # Starter message
            AIMessage(content="Hello, how may I help you?"),
        ]
    # Vector store Stateful
    if "vectore_store" not in st.session_state:
        st.session_state.vectore_store = get_vector_store(website_url)

    

    # Streamlit Chat Input
    user_input = st.chat_input("Type your message here...")

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
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
