import streamlit as st
from streamlit import session_state as ss
import streamlit as st
import json
import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredHTMLLoader,PyPDFLoader
import shutil
from langchain_community.vectorstores import FAISS
import os
import time
from langchain_community.vectorstores import Chroma

if('db_created' not in st.session_state):
    st.session_state['db_created']=False
if('subject' not in st.session_state):
    st.session_state['subject']=None
if('grade' not in st.session_state):
    st.session_state['grade']=None
   
st.title("History and Economics Chatbot")


subject = st.selectbox("Select Subject", ["Economics", "History"],index=0)

if subject == "Economics":
    grade = st.selectbox("Select Grade", ["11th", "12th"],index=0)
else:
    grade = st.selectbox("Select Grade", ["10th", "11th", "12th"],index=0)

if(st.button("Create Database")):
    folder_path = f"D:\\ML\\superkalam\\RAG-with-citations\\database\\{subject}\\{grade}"
    if os.path.exists(folder_path):
        st.success(f"The database folder for {subject} {grade} grade already exists.")
    else:
        with st.spinner("Creating database folder..."):
            # time.sleep(3)  
            loader=DirectoryLoader(f"D:\\ML\\superkalam\\RAG-with-citations\\docs\\{subject}\\{grade}",loader_cls=PyPDFLoader)
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
            docs = text_splitter.split_documents(documents)
            embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
            # FAISS
            db = FAISS.from_documents(docs, embedding_function)
            db.save_local(f"D:\\ML\\superkalam\\RAG-with-citations\\database\\{subject}\\{grade}")
            # Chroma
            # db = Chroma.from_documents(docs, embedding_function,persist_directory='db')
            # db = Chroma(persist_directory=f"D:\\ML\\superkalam\\database_chroma\\{st.session_state['subject']}\\{st.session_state['grade']}", embedding_function=embedding_function)
            st.success(f"The database folder for {subject} {grade} has been created.")
    st.session_state['db_created']=True
    st.session_state['subject']=subject
    st.session_state['grade']=grade

