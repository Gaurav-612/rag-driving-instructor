import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time

import sys
import warnings

if 'warnings' not in sys.modules:
    sys.modules['warnings'] = warnings

# Rest of your imports and code...

st.title("RAG-Based Driving Instructor")

with st.expander("About this app"):
    st.markdown("""
    #### RAG App to know about driving in Ontario, based on NVIDIA NIM

    This application uses Retrieval Augmented Generation (RAG) to answer questions about driving in Ontario. It's powered by NVIDIA NIM (NVIDIA Inference Microservices).

    **How it works:**
    1. The app uses a PDF of the [Ontario Driver's Handbook](https://www.ontario.ca/document/official-mto-drivers-handbook) as its knowledge base.
    2. When you ask a question, the app searches through this knowledge base to find relevant information.
    3. It then uses NVIDIA's advanced language model to generate a response based on the relevant information and your question.

    **Features:**
    - Real-time answers to your driving-related questions
    - Information sourced directly from official Ontario driving documentation
    - Powered by state-of-the-art AI technology from NVIDIA

    To get started, enter your NVIDIA API key in the sidebar and ask a question about driving in Ontario!
    """)

def get_embedding():
    if 'vectors' not in st.session_state:
        with st.spinner("Processing documents..."):
            try:
                st.session_state.embeddings = NVIDIAEmbeddings(verbose=False)
                st.session_state.loader = PyPDFDirectoryLoader('./pdfs')
                st.session_state.docs = st.session_state.loader.load()
                st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=700,chunk_overlap=50)
                st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
                print("Documents have been split")
                
                st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)
                st.sidebar.success("FAISS vector store is ready")
            except Exception as e:
                st.sidebar.error(f"Error in get_embedding: {str(e)}")


# initialize the session state for the API key and question
if 'NVIDIA_API_KEY' not in st.session_state:
    st.session_state.NVIDIA_API_KEY = ''

if 'question' not in st.session_state:
    st.session_state.question = ''

# Add sidebar content
st.sidebar.title("API Key")
st.sidebar.markdown("Enter your NVIDIA API Key below. If you don't have one, you can get it from [NVIDIA AI Playground](https://build.nvidia.com/explore/discover).")

# session state to keep the API key in the input field
NVIDIA_API_KEY = st.sidebar.text_input("NVIDIA API Key", type="password", value=st.session_state.NVIDIA_API_KEY)
submit_button = st.sidebar.button("Submit API Key")

# Initialize the LLM and run get_embedding() when API key is submitted
if submit_button:
    if NVIDIA_API_KEY:
        # Save the API key to session state
        st.session_state.NVIDIA_API_KEY = NVIDIA_API_KEY
        os.environ["NVIDIA_API_KEY"] = NVIDIA_API_KEY
        try:
            llm = ChatNVIDIA(model_name="meta/llama3-70b-instruct", verbose=False)
            st.session_state.llm = llm
            st.sidebar.success("API Key submitted successfully")
            get_embedding()
        except Exception as e:
            st.sidebar.error(f"Error initializing ChatNVIDIA: {str(e)}")
    else:
        st.sidebar.error("Please enter your NVIDIA API Key before submitting.")

prompt=ChatPromptTemplate.from_template(
"""
You are a driving instructor with multiple years of experience in Ontario, Canada.
Answer the questions based on the provided context only.
Provide the most accurate response based on the question
<context>
{context}
</context>
Questions:{input}
"""
)

prompt = st.text_input("Ask any question about the driving authority", value=st.session_state.question)
ask_button = st.button("Ask me")

if ask_button:
    if st.session_state.NVIDIA_API_KEY and 'vectors' in st.session_state and 'llm' in st.session_state:
        if prompt:
            # Update the session state whenever the input changes
            if prompt != st.session_state.question:
                st.session_state.question = prompt

            try:
                document_chain=create_stuff_documents_chain(st.session_state.llm, prompt)
                retriever=st.session_state.vectors.as_retriever()
                retrieval_chain=create_retrieval_chain(retriever,document_chain)
                start=time.process_time()
                with st.spinner("Thinking..."):
                    response=retrieval_chain.invoke({'input':prompt})
                print("Response time :",time.process_time()-start)
                st.write(response['answer'])

                with st.expander("Similarity Results"):
                    for i, doc in enumerate(response["context"]):
                        st.write(doc.page_content)
                        st.write("--------------------------------")
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
        else:
            st.warning("Please enter a question before clicking 'Ask me'.")
    elif not st.session_state.NVIDIA_API_KEY:
        st.error("Please enter your NVIDIA API Key in the sidebar and click 'Submit API Key' before asking questions.")
    elif 'llm' not in st.session_state:
        st.error("LLM initialization failed. Please check your API key and try again.")
    else:
        st.error("Document embeddings are not ready. Please wait for the process to complete after submitting your API key.")
