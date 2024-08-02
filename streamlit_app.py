import streamlit as st
import os
import requests
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
import os
import pickle
load_dotenv()

## load the GROQ And OpenAI API KEY 
groq_api_key=os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")

st.title("Military Monk Chat Bot 🎖️🛦🪖")

llm=ChatGroq(groq_api_key=groq_api_key,
            model_name="llama3-8b-8192")
# Llama3-8b-8192 

prompt=ChatPromptTemplate.from_template(
"""
You are a Q&A assistant. Your goal is to answer questions as
accurately as possible based on the instructions and context provided only.
<context>
{context}
<context>
Questions: {input}
"""
)

def load_embeddings():
    if "vectors" not in st.session_state:
        if os.path.exists("vectors.pkl"):
            try:
                with open("vectors.pkl", "rb") as f:
                    st.session_state.vectors = pickle.load(f)
                st.success("Loaded vectors from pickle file.")
            except pickle.UnpicklingError:
                st.error("Failed to load vectors from pickle file. Please check the file format and content.")
                st.stop()  # Stop execution if loading fails


load_embeddings()
prompt1=st.chat_input("Enter Your Question")


# if st.button("Tap to talk with Bot"):
#     vector_embedding()
#     st.write("Go........! 🚀")

import time

with st.sidebar:
    st.title("Military Monk")
    st.subheader("This app lets you clear doubts [👉]")
    add_vertical_space(2)
    st.write("Reach out to me on [LinkedIn](https://www.linkedin.com/in/prasad-kumbhar-/)")
    add_vertical_space(2)
    st.write("Give Feedback [Google Form](https://forms.gle/YEvpBio2TVRDQoYFA)")
    add_vertical_space(2)
    st.write("Our Blog [Website](https://military-monk.blogspot.com/)")
    add_vertical_space(2)
    st.write("Tip: Be specific while asking question")

                
if prompt1:
    # vector_embedding()
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()
    response=retrieval_chain.invoke({'input':prompt1})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

    # With a streamlit expander
    # with st.expander("Document Similarity Search"):
    #     # Find the relevant chunks
    #     for i, doc in enumerate(response["context"]):
    #         st.write(doc.page_content)
    #         st.write("--------------------------------")

# Thanks to tutorial from Krish Naik
