from streamlit_chat import message
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
import os
import pickle
import tempfile

from datetime import datetime


def initialize_session_state():

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about "]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! "]


def conversation_chat(query, chain, history):
    # Chunk the conversation history into smaller segments (adjust chunk_size as needed)
    chunk_size = 512
    chunks = [history[i:i + chunk_size] for i in range(0, len(history), chunk_size)]

    # Generate responses for each chunk and concatenate them
    responses = []
    for chunk in chunks:
        combined_chunk = " ".join([item[0] for item in chunk])  # Combine prompts
        result = chain({"question": query, "chat_history": combined_chunk})
        responses.append(result["answer"])

    final_answer = " ".join(responses)
    history.append((query, final_answer))
    return final_answer


def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your PDF", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):

                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                st.sidebar.write("question starting time is", current_time)

                output = conversation_chat(user_input, chain, st.session_state['history'])

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")


def create_conversational_chain(vector_store):
    # Create llm
    llm = LlamaCpp(
        streaming=True,
        model_path="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        temperature=0.75,
        top_p=1,
        verbose=True,
        n_ctx=4096
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                 retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
                                                 memory=memory)

    return chain

def main():
    initialize_session_state()
    st.title("PDF ChatBot using Mistral-7B-Instruct :books:")
    st.subheader("Documents Uploading")
    uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)

    if uploaded_files:
        text = []
        for file in uploaded_files:
            file_extension = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            loader = None
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)
    
            if loader:
                text.extend(loader.load())
                os.remove(temp_file_path)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
        text_chunks = text_splitter.split_documents(text)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                           model_kwargs={'device': 'cpu'})

        store_name = uploaded_files[0].name[:-4]
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                vector_store = pickle.load(f)
            st.write('Embeddings Loaded from the Disk')
        else:
            vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(vector_store, f)
            st.write('Embeddings Computed Success')

        chain = create_conversational_chain(vector_store)

        display_chat_history(chain)


if __name__ == "__main__":
    main()
